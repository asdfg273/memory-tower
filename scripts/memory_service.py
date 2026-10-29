"""
memory_service.py —— 基于 SQLite 的记忆存储服务
==============================================

职责边界（很重要）
------------------
本服务**只做持久化与检索**，不训练、不修改任何模型权重，也不做语义判断。
判断由 model（提议）+ rules（护栏）负责，本服务只执行最终指令。

表结构（需求字段全覆盖）
------------------------
    memories
      memory_id     TEXT PRIMARY KEY     记忆 ID（形如 m_xxxxxxxx）
      user_id       TEXT NOT NULL        用户
      task_id       TEXT                 任务/项目
      raw_source    TEXT NOT NULL        原始片段（第一版只存原文，不存归一化摘要）
      memory_type   TEXT NOT NULL        short / medium / long
      scope         TEXT NOT NULL        turn / session / task / user
      created_at    REAL NOT NULL        Unix 时间戳（秒，float）
      updated_at    REAL NOT NULL
      expires_at    REAL                 NULL = 不过期
      confidence    REAL NOT NULL        0~1
      version       INTEGER NOT NULL     当前版本号，从 1 开始
      deleted_at    REAL                 NULL = 未删除（软删除）
      metadata      TEXT                 额外信息 JSON（来源、规则审计等）
    memory_versions
      memory_id / version / 上面各业务字段的**历史快照**，用于审计与回滚

接口
----
    create()     新建记忆
    update()     更新（写新版本，旧版本进历史表）
    delete()     软删除（可 hard=True 物理删除）
    merge()      合并多条重复记忆（保留主体，其余软删除，raw_source 拼接）
    get_relevant() 按 user/task/scope + 文本相似度检索相关记忆
    expire_now() 清理到期记忆（可定时调用）

用法
----
    from memory_service import MemoryService
    svc = MemoryService("data/memory.db")
    mid = svc.create("以后这个项目都用中文回答", user_id="u1", task_id="projA",
                     memory_type="long", scope="user", confidence=0.93)
    svc.get_relevant(user_id="u1", task_id="projA", query="中文回答")

CLI：
    python memory_service.py --db data/memory.db --demo     # 跑一遍回归演示
    python memory_service.py --db data/memory.db --stats    # 查看统计
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common import (
    MEMORY_TYPE_DEFAULT_SCOPE,
    MEMORY_TYPE_RANK,
    Memory,
    SCOPE_LABELS,
    TTL_SECONDS,
    normalize_text,
    now_ts,
    span_key,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id   TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    task_id     TEXT,
    raw_source  TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    scope       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    expires_at  REAL,
    confidence  REAL NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    deleted_at  REAL,
    metadata    TEXT
);
CREATE INDEX IF NOT EXISTS idx_mem_user      ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_mem_task      ON memories(user_id, task_id);
CREATE INDEX IF NOT EXISTS idx_mem_scope     ON memories(user_id, scope);
CREATE INDEX IF NOT EXISTS idx_mem_alive     ON memories(user_id, deleted_at);
CREATE INDEX IF NOT EXISTS idx_mem_expires   ON memories(expires_at);

CREATE TABLE IF NOT EXISTS memory_versions (
    memory_id   TEXT NOT NULL,
    version     INTEGER NOT NULL,
    user_id     TEXT NOT NULL,
    task_id     TEXT,
    raw_source  TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    scope       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    expires_at  REAL,
    confidence  REAL NOT NULL,
    change_kind TEXT NOT NULL,      -- create / update / delete / merge_out / merge_in
    note        TEXT,
    recorded_at REAL NOT NULL,
    PRIMARY KEY (memory_id, version)
);
"""

# 业务字段（用于生成 SQL，避免手写重复列名）
BUSINESS_FIELDS = (
    "user_id", "task_id", "raw_source", "memory_type", "scope",
    "created_at", "updated_at", "expires_at", "confidence", "version",
    "deleted_at", "metadata",
)


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    """一条完整记忆记录（含审计字段）。"""

    memory_id: str
    user_id: str
    task_id: Optional[str]
    raw_source: str
    memory_type: str
    scope: str
    created_at: float
    updated_at: float
    expires_at: Optional[float]
    confidence: float
    version: int = 1
    deleted_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_alive(self) -> bool:
        return self.deleted_at is None

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= now_ts()

    def to_memory(self) -> Memory:
        """转成跨模块使用的 Memory（供规则层/服务层传递）。"""
        return Memory(
            action="create", memory_type=self.memory_type, scope=self.scope,
            source_span=self.raw_source, confidence=self.confidence,
            memory_id=self.memory_id, user_id=self.user_id, task_id=self.task_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "memory_id": self.memory_id, "user_id": self.user_id, "task_id": self.task_id,
            "raw_source": self.raw_source, "memory_type": self.memory_type, "scope": self.scope,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "expires_at": self.expires_at, "confidence": self.confidence,
            "version": self.version, "deleted_at": self.deleted_at,
            "is_expired": self.is_expired, "metadata": self.metadata,
        }
        return d

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "MemoryRecord":
        meta_raw = row["metadata"] if "metadata" in row.keys() else None
        try:
            meta = json.loads(meta_raw) if meta_raw else {}
        except (TypeError, json.JSONDecodeError):
            meta = {"_raw_metadata": meta_raw}
        return cls(
            memory_id=row["memory_id"], user_id=row["user_id"], task_id=row["task_id"],
            raw_source=row["raw_source"], memory_type=row["memory_type"], scope=row["scope"],
            created_at=float(row["created_at"]), updated_at=float(row["updated_at"]),
            expires_at=(float(row["expires_at"]) if row["expires_at"] is not None else None),
            confidence=float(row["confidence"]), version=int(row["version"]),
            deleted_at=(float(row["deleted_at"]) if row["deleted_at"] is not None else None),
            metadata=meta,
        )


class MemoryServiceError(RuntimeError):
    """服务层异常（找不到记忆、参数非法等）。"""


# ---------------------------------------------------------------------------
# 检索打分（轻量、可解释）
# ---------------------------------------------------------------------------

def _bigrams(text: str) -> set:
    key = span_key(text)
    if len(key) < 2:
        return {key} if key else set()
    return {key[i:i + 2] for i in range(len(key) - 1)}


def text_score(query: str, text: str) -> float:
    """0~1 的粗粒度文本相关度（bigram 余弦，够用且无需依赖额外模型）。

    实现复用 common.char_similarity，避免在多处各写一份相似度代码。
    """
    from common import char_similarity

    return char_similarity(query, text)


def _char_grams(text: str, n: int) -> List[str]:
    key = span_key(text)
    return [key[i:i + n] for i in range(max(0, len(key) - n + 1))]


def _term_freq(text: str, n: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for gram in _char_grams(text, n):
        out[gram] = out.get(gram, 0.0) + 1.0
    return out


def _cosine_idf(a: Dict[str, float], b: Dict[str, float],
                idf: Dict[str, float]) -> float:
    num = na = nb = 0.0
    for gram in set(a) | set(b):
        w = idf.get(gram, 1.0)
        x, y = a.get(gram, 0.0) * w, b.get(gram, 0.0) * w
        num += x * y
        na += x * x
        nb += y * y
    return num / (math.sqrt(na) * math.sqrt(nb)) if na and nb else 0.0


def text_score_idf(query: str, text: str, idf1: Dict[str, float],
                   idf2: Dict[str, float], w_unigram: float = 0.5) -> float:
    """idf 加权的一元 + 二元 char 余弦。

    比朴素 bigram 余弦更抗改写：中文改写常换词序/换搭配，二元组匹配不上时
    一元组仍能命中，而 idf 会压掉"的/了/是"这类到处都有的字。
    实测（38 条 query，`reports/memory_retrieval_ab.md`）：pooled hit@1 .605→.684，
    零重叠改写组 .333→.556，一元权重在 0.3~0.5 之间是平台区。
    """
    return (w_unigram * _cosine_idf(_term_freq(query, 1), _term_freq(text, 1), idf1)
            + (1.0 - w_unigram) * _cosine_idf(_term_freq(query, 2), _term_freq(text, 2), idf2))


def recency_score(updated_at: float, half_life_days: float = 14.0) -> float:
    """时间衰减：半衰期默认 14 天，返回 0~1。"""
    age_days = max(0.0, (now_ts() - updated_at) / 86400.0)
    return float(0.5 ** (age_days / max(0.5, half_life_days)))


# ---------------------------------------------------------------------------
# 服务
# ---------------------------------------------------------------------------

class MemoryService:
    """SQLite 记忆服务。

    线程安全说明：所有写操作都用 `with self._conn` 事务包裹；连接使用
    `check_same_thread=False` 以方便在 DataLoader / Web 线程里复用，
    但不保证多线程并发写的顺序性（本项目单写入者场景足够）。
    """

    # 检索时各 scope 的基础权重：越"持久/越全局"的记忆越优先被召回
    SCOPE_WEIGHT: Dict[str, float] = {"user": 1.0, "task": 0.85, "session": 0.7, "turn": 0.5}

    def __init__(self, db_path: str = os.path.join("data", "memory.db"), init: bool = True) -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # idf 表随库内容变化重建（键 = 条数 + 最新 updated_at）
        self._idf: Optional[Tuple[Dict[str, float], Dict[str, float]]] = None
        self._idf_key: Optional[Tuple[int, float]] = None
        # WAL 提升并发读写体验（内存库不支持，忽略异常）
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        except sqlite3.Error:
            pass
        if init:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ---------------- 生命周期 ----------------
    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "MemoryService":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------------- 内部工具 ----------------
    def _idf_tables(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """按当前存活记忆算一元/二元 idf（条数或最新 updated_at 变了才重算）。"""
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(updated_at), 0) FROM memories "
            "WHERE deleted_at IS NULL"
        ).fetchone()
        key = (int(row[0] or 0), float(row[1] or 0.0))
        if self._idf is not None and self._idf_key == key:
            return self._idf
        texts = [r["raw_source"] for r in self._conn.execute(
            "SELECT raw_source FROM memories WHERE deleted_at IS NULL")]
        n = max(1, len(texts))
        df1: Dict[str, float] = {}
        df2: Dict[str, float] = {}
        for text in texts:
            for gram in set(_char_grams(text, 1)):
                df1[gram] = df1.get(gram, 0.0) + 1.0
            for gram in set(_char_grams(text, 2)):
                df2[gram] = df2.get(gram, 0.0) + 1.0
        idf1 = {g: math.log((n + 1) / (c + 0.5)) for g, c in df1.items()}
        idf2 = {g: math.log((n + 1) / (c + 0.5)) for g, c in df2.items()}
        self._idf = (idf1, idf2)
        self._idf_key = key
        return self._idf

    @staticmethod
    def new_id() -> str:
        return "m_" + uuid.uuid4().hex[:12]

    @staticmethod
    def _normalize_fields(memory_type: str, scope: str) -> Tuple[str, str]:
        memory_type = (memory_type or "medium").lower()
        scope = (scope or MEMORY_TYPE_DEFAULT_SCOPE.get(memory_type, "task")).lower()
        if scope not in SCOPE_LABELS:
            scope = "task"
        if memory_type not in MEMORY_TYPE_RANK:
            memory_type = "medium"
        # 一致性修正：不允许 scope 超出其持久度上限
        cap = {"turn": "short", "session": "short", "task": "medium", "user": "long"}[scope]
        if MEMORY_TYPE_RANK[memory_type] > MEMORY_TYPE_RANK[cap]:
            memory_type = cap
        return memory_type, scope

    @staticmethod
    def _compute_expiry(scope: str, ttl_seconds: Optional[int], expires_at: Optional[float]) -> Optional[float]:
        if expires_at is not None:
            return float(expires_at)
        if ttl_seconds is not None:
            return now_ts() + float(ttl_seconds) if ttl_seconds > 0 else None
        ttl = TTL_SECONDS.get(scope)
        return None if ttl is None else now_ts() + float(ttl)

    def _record_version(self, record: MemoryRecord, change_kind: str, note: str = "") -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO memory_versions
               (memory_id, version, user_id, task_id, raw_source, memory_type, scope,
                created_at, updated_at, expires_at, confidence, change_kind, note, recorded_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (record.memory_id, record.version, record.user_id, record.task_id, record.raw_source,
             record.memory_type, record.scope, record.created_at, record.updated_at,
             record.expires_at, record.confidence, change_kind, note, now_ts()),
        )

    def _get_row(self, memory_id: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            raise MemoryServiceError(f"记忆不存在：{memory_id}")
        return row

    # ---------------- CRUD ----------------
    def create(
        self,
        raw_source: str,
        user_id: str = "default",
        task_id: Optional[str] = None,
        memory_type: str = "medium",
        scope: str = "task",
        confidence: float = 0.0,
        expires_at: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
        _conn_managed: bool = False,
    ) -> str:
        """新建一条记忆，返回 memory_id。

        `raw_source` 就是 source_span 原文；第一版**不做归一化摘要**。
        `expires_at` 未指定时按 scope 的 TTL_SECONDS 自动计算（user 级默认永久）。
        """
        raw_source = normalize_text(raw_source)
        if not raw_source:
            raise MemoryServiceError("raw_source 不能为空")
        memory_type, scope = self._normalize_fields(memory_type, scope)
        confidence = min(1.0, max(0.0, float(confidence)))
        ts = now_ts()
        mid = memory_id or self.new_id()
        expiry = self._compute_expiry(scope, ttl_seconds, expires_at)

        record = MemoryRecord(
            memory_id=mid, user_id=user_id or "default", task_id=task_id, raw_source=raw_source,
            memory_type=memory_type, scope=scope, created_at=ts, updated_at=ts,
            expires_at=expiry, confidence=confidence, version=1, deleted_at=None,
            metadata=dict(metadata or {}),
        )
        try:
            self._conn.execute(
                f"""INSERT INTO memories (memory_id, {', '.join(BUSINESS_FIELDS)})
                    VALUES ({', '.join(['?'] * (len(BUSINESS_FIELDS) + 1))})""",
                (record.memory_id, record.user_id, record.task_id, record.raw_source,
                 record.memory_type, record.scope, record.created_at, record.updated_at,
                 record.expires_at, record.confidence, record.version, record.deleted_at,
                 json.dumps(record.metadata, ensure_ascii=False)),
            )
            self._record_version(record, "create", "新建记忆")
            if not _conn_managed:
                self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise MemoryServiceError(f"memory_id 冲突：{mid}（{exc}）") from exc
        return mid

    def update(
        self,
        memory_id: str,
        raw_source: Optional[str] = None,
        confidence: Optional[float] = None,
        memory_type: Optional[str] = None,
        scope: Optional[str] = None,
        task_id: Optional[str] = None,
        expires_at: Optional[float] = None,
        note: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        _conn_managed: bool = False,
    ) -> int:
        """更新一条记忆 -> 写入新版本，返回新版本号。

        未提供的字段保持不变；`version` 自增；旧值已由 memory_versions 保留。
        """
        row = self._get_row(memory_id)
        old = MemoryRecord.from_row(row)

        new_source = normalize_text(raw_source) if raw_source is not None else old.raw_source
        if not new_source:
            raise MemoryServiceError("更新后的 raw_source 不能为空")
        new_type, new_scope = self._normalize_fields(
            memory_type or old.memory_type, scope or old.scope
        )
        new_conf = old.confidence if confidence is None else min(1.0, max(0.0, float(confidence)))
        new_version = old.version + 1
        ts = now_ts()
        new_expiry = old.expires_at if expires_at is None else expires_at
        new_meta = {**old.metadata, **(metadata or {})}
        new_meta.setdefault("history", []).append(
            {"version": old.version, "raw_source": old.raw_source, "at": ts}
        )

        try:
            self._conn.execute(
                """UPDATE memories
                   SET raw_source=?, memory_type=?, scope=?, updated_at=?, expires_at=?,
                       confidence=?, version=?, task_id=?, metadata=?, deleted_at=NULL
                   WHERE memory_id=?""",
                (new_source, new_type, new_scope, ts, new_expiry, new_conf, new_version,
                 task_id if task_id is not None else old.task_id,
                 json.dumps(new_meta, ensure_ascii=False), memory_id),
            )
            updated = MemoryRecord(
                memory_id=memory_id, user_id=old.user_id,
                task_id=task_id if task_id is not None else old.task_id,
                raw_source=new_source, memory_type=new_type, scope=new_scope,
                created_at=old.created_at, updated_at=ts, expires_at=new_expiry,
                confidence=new_conf, version=new_version, deleted_at=None, metadata=new_meta,
            )
            self._record_version(updated, "update", note or f"由 v{old.version} 更新")
            if not _conn_managed:
                self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise MemoryServiceError(f"更新失败：{exc}") from exc
        return new_version

    def delete(
        self,
        memory_id: Optional[str] = None,
        raw_source: Optional[str] = None,
        user_id: Optional[str] = None,
        hard: bool = False,
        note: str = "",
        _conn_managed: bool = False,
    ) -> List[str]:
        """删除记忆：默认软删除（deleted_at），hard=True 时物理删除。

        支持两种定位方式：memory_id 精确删除；raw_source 走相似度兜底匹配
        （用于模型只给出 delete + 文本、没给 id 的情况）。返回被删除的 id 列表。
        """
        targets: List[MemoryRecord] = []
        if memory_id:
            targets = [MemoryRecord.from_row(self._get_row(memory_id))]
        elif raw_source:
            candidates = self.find_by_text(raw_source, user_id=user_id, limit=3, min_score=0.30)
            targets = [c for c, _ in candidates]
            if not targets:
                raise MemoryServiceError(f"未找到与 {raw_source!r} 匹配的记忆，无法删除")
        else:
            raise MemoryServiceError("delete 需要 memory_id 或 raw_source")

        ts = now_ts()
        deleted_ids: List[str] = []
        try:
            for rec in targets:
                if hard:
                    self._conn.execute("DELETE FROM memories WHERE memory_id=?", (rec.memory_id,))
                    change = "hard_delete"
                else:
                    self._conn.execute(
                        "UPDATE memories SET deleted_at=?, updated_at=? WHERE memory_id=?",
                        (ts, ts, rec.memory_id),
                    )
                    rec.deleted_at, rec.updated_at = ts, ts
                    change = "delete"
                self._record_version(rec, change, note or f"删除（{change}）")
                deleted_ids.append(rec.memory_id)
            if not _conn_managed:
                self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise MemoryServiceError(f"删除失败：{exc}") from exc
        return deleted_ids

    def merge(
        self,
        memory_ids: Sequence[str],
        keep_id: Optional[str] = None,
        separator: str = "；",
        new_confidence: Optional[float] = None,
        note: str = "",
    ) -> Dict[str, Any]:
        """合并多条重复记忆。

        规则：keep_id（默认第一条）作为主体，raw_source 去重后拼接，
        version+1，其余记忆软删除并在 metadata 里互相记录。
        """
        ids = [i for i in memory_ids if i]
        if len(ids) < 2:
            raise MemoryServiceError("merge 至少需要 2 条记忆")
        keep = keep_id or ids[0]
        if keep not in ids:
            raise MemoryServiceError("keep_id 必须出现在 memory_ids 中")

        records = [MemoryRecord.from_row(self._get_row(i)) for i in ids]
        keep_rec = next(r for r in records if r.memory_id == keep)
        others = [r for r in records if r.memory_id != keep]

        # 去重拼接：保留语义顺序（按 created_at 排序）
        seen = set()
        parts: List[str] = []
        for rec in sorted(records, key=lambda r: r.created_at):
            body = rec.raw_source
            key = span_key(body)
            if key and key not in seen:
                seen.add(key)
                parts.append(body)
            else:
                # 文本高度重复时，保留更长的那个
                for i, existing in enumerate(parts):
                    if span_key(body) in span_key(existing) and len(body) > len(existing):
                        parts[i] = body
        merged_source = separator.join(parts)

        conf = new_confidence
        if conf is None:
            conf = max(r.confidence for r in records)

        try:
            self.update(
                keep, raw_source=merged_source, confidence=conf,
                note=note or f"合并 {len(ids)} 条记忆：{ids}", _conn_managed=True,
            )
            merged_meta = {"merged_from": ids, "merge_at": now_ts()}
            self._conn.execute(
                "UPDATE memories SET metadata=? WHERE memory_id=?",
                (json.dumps({**keep_rec.metadata, **merged_meta}, ensure_ascii=False), keep),
            )
            now = now_ts()
            for rec in others:
                self._conn.execute(
                    "UPDATE memories SET deleted_at=?, updated_at=?, metadata=? WHERE memory_id=?",
                    (now, now,
                     json.dumps({**rec.metadata, "merged_into": keep}, ensure_ascii=False),
                     rec.memory_id),
                )
                rec.deleted_at, rec.updated_at = now, now
                self._record_version(rec, "merge_in", f"被合并进 {keep}")
            self._record_version(
                MemoryRecord.from_row(self._get_row(keep)), "merge_out", f"合并了 {ids}"
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise MemoryServiceError(f"合并失败：{exc}") from exc

        return {"keep_id": keep, "merged_ids": ids, "raw_source": merged_source,
                "deleted": [r.memory_id for r in others], "confidence": conf}

    def execute(self, instruction: Memory, user_id: str = "default",
                task_id: Optional[str] = None) -> Dict[str, Any]:
        """执行一条最终写入指令（rules.RuleDecision.final）。

        这是"模型/规则 -> 存储"的唯一入口，方便统一审计。
        """
        action = (instruction.action or "ignore").lower()
        if action == "ignore":
            return {"action": "ignore", "executed": False}
        metadata = {"origin": "memory_controller", "notes": instruction.notes,
                    "reason": instruction.reason}
        extra = getattr(instruction, "metadata", None)
        if isinstance(extra, dict):
            metadata.update({k: v for k, v in extra.items() if v is not None})
        if action == "create":
            mid = self.create(
                instruction.source_span, user_id=user_id, task_id=task_id,
                memory_type=instruction.memory_type, scope=instruction.scope,
                confidence=instruction.confidence, metadata=metadata,
            )
            return {"action": "create", "executed": True, "memory_id": mid}
        if action == "update":
            target = instruction.target_memory_id
            if not target:
                # 没给 id：按文本相似度兜底
                found = self.find_by_text(instruction.source_span, user_id=user_id,
                                          limit=1, min_score=0.40)
                target = found[0][0].memory_id if found else None
            if target:
                ver = self.update(target, raw_source=instruction.source_span,
                                  confidence=instruction.confidence,
                                  memory_type=instruction.memory_type, scope=instruction.scope,
                                  note=instruction.reason, metadata=metadata)
                return {"action": "update", "executed": True, "memory_id": target,
                        "version": ver}
            # 没有可更新的目标 -> 退化为新建（并在返回值里说明）
            mid = self.create(
                instruction.source_span, user_id=user_id, task_id=task_id,
                memory_type=instruction.memory_type, scope=instruction.scope,
                confidence=instruction.confidence,
                metadata={**metadata, "fallback_from": "update"},
            )
            return {"action": "create", "executed": True, "memory_id": mid,
                    "note": "update 未匹配到旧记忆，已退化为 create"}
        if action == "delete":
            ids = self.delete(memory_id=instruction.target_memory_id,
                              raw_source=instruction.source_span if not instruction.target_memory_id else None,
                              user_id=user_id, note=instruction.reason)
            return {"action": "delete", "executed": True, "deleted": ids}
        raise MemoryServiceError(f"未知 action：{action}")

    # ---------------- 查询 ----------------
    def get(self, memory_id: str, include_deleted: bool = False) -> Optional[MemoryRecord]:
        row = self._conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
        if row is None:
            return None
        rec = MemoryRecord.from_row(row)
        if rec.deleted_at is not None and not include_deleted:
            return None
        return rec

    def history(self, memory_id: str) -> List[Dict[str, Any]]:
        """返回一条记忆的完整版本历史（审计用）。"""
        rows = self._conn.execute(
            "SELECT * FROM memory_versions WHERE memory_id=? ORDER BY version ASC", (memory_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def text_similarity(self, query: str, text: str) -> float:
        """检索打分的唯一入口：idf 加权 1+2 元，与 get_relevant 内部同口径。

        两套打分器并存会让产品行为漂移：同一批 38 条 query 上，朴素 2-gram 的
        pooled hit@1 是 .605，本函数是 .658（para 组 .333→.556）。所以
        find_by_text 与 chat_app.retrieve 的二段重排都必须走这里。
        阈值按新尺度重新标定过（_ab_tmp/_thr_fit.py，1064 对 query×memory）：
        去重 0.25→0.30、定位已有记忆去改/删 0.25→0.40、注入门限 0.08→0.14。
        """
        if not query or not text:
            return 0.0
        idf1, idf2 = self._idf_tables()
        return text_score_idf(query, text, idf1, idf2)

    def find_by_text(
        self,
        text: str,
        user_id: Optional[str] = None,
        limit: int = 5,
        min_score: float = 0.3,
    ) -> List[Tuple[MemoryRecord, float]]:
        """按文本相似度找记忆（规则层冲突检测 / delete 兜底用）。"""
        rows = self._conn.execute(
            """SELECT * FROM memories WHERE deleted_at IS NULL
               AND (user_id = ? OR ? IS NULL)""",
            (user_id, user_id),
        ).fetchall()
        scored = [(MemoryRecord.from_row(r), self.text_similarity(text, r["raw_source"]))
                  for r in rows]
        scored = [(rec, s) for rec, s in scored if s >= min_score]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def get_relevant(
        self,
        user_id: str,
        task_id: Optional[str] = None,
        query: str = "",
        limit: int = 8,
        scopes: Optional[Sequence[str]] = None,
        include_expired: bool = False,
        min_score: float = 0.0,
    ) -> List[MemoryRecord]:
        """检索与当前上下文相关的记忆。

        打分 = 0.55 * 文本相关度 + 0.2 * scope 权重 + 0.15 * 置信度 + 0.1 * 时间新鲜度
        其中 task 级记忆只在 task_id 匹配时参与打分（避免跨项目污染）。
        结果按分数降序返回。
        """
        now = now_ts()
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE deleted_at IS NULL AND user_id = ?", (user_id,)
        ).fetchall()

        allowed_scopes = set(scopes) if scopes else set(SCOPE_LABELS)
        scored: List[Tuple[MemoryRecord, float]] = []
        for row in rows:
            rec = MemoryRecord.from_row(row)
            if rec.scope not in allowed_scopes:
                continue
            if rec.is_expired and not include_expired:
                continue
            if rec.scope == "task" and task_id and rec.task_id and rec.task_id != task_id:
                continue
            sim = 0.0
            if query:
                idf1, idf2 = self._idf_tables()
                sim = text_score_idf(query, rec.raw_source, idf1, idf2)
            score = (
                0.55 * sim
                + 0.20 * self.SCOPE_WEIGHT.get(rec.scope, 0.5)
                + 0.15 * rec.confidence
                + 0.10 * recency_score(rec.updated_at)
            )
            if score >= min_score:
                scored.append((rec, score))
        scored.sort(key=lambda x: (x[1], x[0].updated_at), reverse=True)
        return [rec for rec, _ in scored[:limit]]

    def candidates_for_rules(
        self, user_id: str, task_id: Optional[str], query: str, limit: int = 8
    ) -> List[Dict[str, Any]]:
        """产出可直接喂给 rules.detect_conflicts 的候选列表（带 id 与元信息）。"""
        return [
            {"memory_id": r.memory_id, "raw_source": r.raw_source, "memory_type": r.memory_type,
             "scope": r.scope, "confidence": r.confidence}
            for r in self.get_relevant(user_id=user_id, task_id=task_id, query=query, limit=limit)
        ]

    # ---------------- 维护 ----------------
    def expire_now(self) -> List[str]:
        """把所有已到期的记忆软删除，返回被清理的 id。"""
        ts = now_ts()
        rows = self._conn.execute(
            "SELECT memory_id FROM memories WHERE deleted_at IS NULL AND expires_at IS NOT NULL "
            "AND expires_at <= ?", (ts,),
        ).fetchall()
        ids = [r["memory_id"] for r in rows]
        try:
            for mid in ids:
                self._conn.execute(
                    "UPDATE memories SET deleted_at=?, updated_at=? WHERE memory_id=?",
                    (ts, ts, mid),
                )
                rec = MemoryRecord.from_row(self._get_row(mid))
                self._record_version(rec, "expire", "TTL 到期自动清理")
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise MemoryServiceError(f"清理过期记忆失败：{exc}") from exc
        return ids

    def stats(self) -> Dict[str, Any]:
        """库统计信息（按类型/作用域/存活状态分组）。"""
        def group(col: str) -> Dict[str, int]:
            rows = self._conn.execute(
                f"SELECT {col} AS k, COUNT(*) AS c FROM memories WHERE deleted_at IS NULL GROUP BY {col}"
            ).fetchall()
            return {str(r["k"]): int(r["c"]) for r in rows}

        total = self._conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"]
        alive = self._conn.execute(
            "SELECT COUNT(*) AS c FROM memories WHERE deleted_at IS NULL").fetchone()["c"]
        expired = self._conn.execute(
            "SELECT COUNT(*) AS c FROM memories WHERE deleted_at IS NULL AND expires_at IS NOT NULL "
            "AND expires_at <= ?", (now_ts(),)).fetchone()["c"]
        versions = self._conn.execute("SELECT COUNT(*) AS c FROM memory_versions").fetchone()["c"]
        return {
            "db_path": self.db_path,
            "total_rows": int(total),
            "alive": int(alive),
            "deleted": int(total) - int(alive),
            "expired_but_alive": int(expired),
            "version_rows": int(versions),
            "by_memory_type": group("memory_type"),
            "by_scope": group("scope"),
        }


# ---------------------------------------------------------------------------
# 演示 / 自检
# ---------------------------------------------------------------------------

def _demo(db_path: str) -> int:
    """跑一遍完整生命周期，验证 create/update/merge/delete/get_relevant。"""
    path = db_path
    if path != ":memory:" and os.path.exists(path):
        os.remove(path)          # 演示用，确保干净（仅删除本演示指定的库文件）
    svc = MemoryService(path)
    print("=" * 78)

    mid1 = svc.create("以后这个项目都用中文回答", user_id="u1", task_id="projA",
                      memory_type="long", scope="user", confidence=0.93)
    mid2 = svc.create("回答要简洁一点", user_id="u1", task_id="projA",
                      memory_type="medium", scope="task", confidence=0.88)
    mid3 = svc.create("回答尽量简洁", user_id="u1", task_id="projA",
                      memory_type="medium", scope="task", confidence=0.8)
    print(f"[create] {mid1} / {mid2} / {mid3}")

    ver = svc.update(mid1, raw_source="以后这个项目和 projB 都用中文回答", confidence=0.95)
    print(f"[update] {mid1} -> v{ver}")
    assert svc.get(mid1).version == 2
    assert len(svc.history(mid1)) >= 2

    merged = svc.merge([mid2, mid3])
    print(f"[merge ] keep={merged['keep_id']} merged_source={merged['raw_source']!r} "
          f"deleted={merged['deleted']}")
    assert svc.get(mid3) is None, "被合并的记忆应不可见"

    hits = svc.get_relevant(user_id="u1", task_id="projA", query="中文 简洁")
    print("[query ] " + " | ".join(f"{r.memory_id}:{r.raw_source[:14]}({r.scope})" for r in hits))
    assert hits, "get_relevant 应返回结果"

    found = svc.find_by_text("以后都用中文回答", user_id="u1")
    print(f"[match ] {[(r.memory_id, round(s, 3)) for r, s in found]}")

    deleted = svc.delete(raw_source="简洁", user_id="u1")
    print(f"[delete] 软删除 {deleted}")

    # TTL：创建一条 60 秒后过期的记忆，再手动回拨 expires_at 验证清理
    mid_exp = svc.create("临时：这次先用英文", user_id="u1", task_id="projA",
                         memory_type="short", scope="turn", confidence=0.7, ttl_seconds=60)
    svc._conn.execute("UPDATE memories SET expires_at=? WHERE memory_id=?",
                      (now_ts() - 1, mid_exp))
    svc._conn.commit()
    expired = svc.expire_now()
    print(f"[expire] 清理 {expired}")
    assert mid_exp in expired

    print("[stats ]", json.dumps(svc.stats(), ensure_ascii=False, indent=2))
    svc.close()
    print("=" * 78)
    print("[demo] 全部断言通过")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="SQLite 记忆服务")
    p.add_argument("--db", type=str, default=os.path.join("data", "memory.db"))
    p.add_argument("--demo", action="store_true", help="跑一遍完整生命周期演示（会清空该库文件）")
    p.add_argument("--stats", action="store_true", help="打印库统计信息")
    p.add_argument("--expire", action="store_true", help="清理到期记忆")
    args = p.parse_args()

    if args.demo:
        raise SystemExit(_demo(args.db))
    svc = MemoryService(args.db)
    if args.expire:
        print("已清理：", svc.expire_now())
    if args.stats or not args.expire:
        print(json.dumps(svc.stats(), ensure_ascii=False, indent=2))
    svc.close()


if __name__ == "__main__":
    main()

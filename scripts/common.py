"""
common.py —— 记忆写入控制器（Memory Write Controller）公共定义
================================================================

本文件被其余 7 个脚本共享，集中放置：

1. 标签空间定义（action / memory_type / scope）—— 保证 generate / preprocess /
   train / infer / rules / evaluate 全链路使用**同一份**标签与顺序，
   任何一处顺序错乱都会造成静默的标签漂移，因此只允许在这里定义。
2. 记忆生命周期常量（TTL、长期记忆阈值、类型等级）。
3. 纯文本工具函数（对话拼接、中文正则归一化、JSON 读写）。
4. `Memory` 数据类 —— 模型输出 / 记忆服务 / 规则层之间传递的统一结构。

注意：本模块**不依赖 torch / transformers**，因此可以在裸环境下被
rules.py、memory_service.py 直接导入并跑单元测试。
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

# ---------------------------------------------------------------------------
# 0. 版本 / 路径
# ---------------------------------------------------------------------------

__version__ = "0.1.0"

# 项目根目录（本文件所在目录），供其它脚本拼接默认路径使用
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 基座模型：SmolLM2-135M-Instruct（极小，8GB 显存绰绰有余，全量微调可行）
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "HuggingFaceTB/SmolLM2-135M-Instruct")

# 默认最大输入长度。135M 模型 max_position_embeddings=8192，但 8GB 显存下
# 512 长度的全量微调已经足够覆盖本项目的数据形态，且显存/速度最稳。
MAX_LENGTH = 512

# ---------------------------------------------------------------------------
# 1. 标签空间（顺序即 label id，禁止随意重排！）
# ---------------------------------------------------------------------------

# 动作：模型只在这 4 类里选。
#   ignore = 不写入
#   create = 新建一条记忆
#   update = 覆盖/修正已有记忆（数据生成阶段的 "merge" 软例也归一到这里）
#   delete = 删除/作废已有记忆
ACTION_LABELS: List[str] = ["ignore", "create", "update", "delete"]
ACTION2ID: Dict[str, int] = {name: i for i, name in enumerate(ACTION_LABELS)}
ID2ACTION: Dict[int, str] = {i: name for name, i in ACTION2ID.items()}

# 记忆类型：短期 / 中期 / 长期
MEMORY_TYPE_LABELS: List[str] = ["short", "medium", "long"]
MEMORY_TYPE2ID: Dict[str, int] = {name: i for i, name in enumerate(MEMORY_TYPE_LABELS)}
ID2MEMORY_TYPE: Dict[int, str] = {i: name for name, i in MEMORY_TYPE2ID.items()}

# 作用域：单轮 / 会话 / 任务 / 用户
SCOPE_LABELS: List[str] = ["turn", "session", "task", "user"]
SCOPE2ID: Dict[str, int] = {name: i for i, name in enumerate(SCOPE_LABELS)}
ID2SCOPE: Dict[int, str] = {i: name for name, i in SCOPE2ID.items()}

NUM_ACTIONS = len(ACTION_LABELS)
NUM_MEMORY_TYPES = len(MEMORY_TYPE_LABELS)
NUM_SCOPES = len(SCOPE_LABELS)

# 中文说明（用于日志 / 报告可读性）
ACTION_ZH = {"ignore": "忽略", "create": "新建", "update": "更新", "delete": "删除"}
MEMORY_TYPE_ZH = {"short": "短期", "medium": "中期", "long": "长期"}
SCOPE_ZH = {"turn": "单轮", "session": "会话", "task": "任务", "user": "用户"}

# 记忆类型"持久度"等级：数值越大越持久，用于 downgrade（降级写入）
MEMORY_TYPE_RANK = {"short": 0, "medium": 1, "long": 2}
RANK2MEMORY_TYPE = {v: k for k, v in MEMORY_TYPE_RANK.items()}

# ---------------------------------------------------------------------------
# 2. 策略阈值
# ---------------------------------------------------------------------------

# 推理阶段：低于该置信度 -> 转 ignore（或要求用户确认）。
# 取值来源：evaluate.py 的阈值扫描（报告中 "R7 置信度阈值扫描" 一节）。
# 在真实数据上 0.70 会把 10 条本应写入的记忆压成 ignore（其中 6 条模型本来判断正确），
# 而真正需要拦截的低质量写入置信度都在 0.9+，所以 0.70 属于过度保守。
# 扫描结果：0.55 在"误写率不变（0.0078）"的前提下把 action 准确率从 0.9526 提到 0.9723。
CONFIDENCE_THRESHOLD = 0.55

# 规则层：长期记忆必须同时满足 action in {create,update} 且 confidence > 0.85
LONG_TERM_CONFIDENCE = 0.85

# 规则层：冲突检测时，新记忆置信度高于该值才允许"覆盖"旧记忆
CONFLICT_OVERWRITE_CONFIDENCE = 0.80

# 各 scope 的默认存活时间（秒）。None = 永不自动过期。
TTL_SECONDS: Dict[str, Optional[int]] = {
    "turn": 30 * 60,               # 单轮：30 分钟
    "session": 24 * 3600,          # 会话：1 天
    "task": 30 * 24 * 3600,        # 任务：30 天
    "user": None,                  # 用户级：永久
}

# 记忆类型 -> 默认 scope（当模型给出的组合不合理时用于兜底修正）
MEMORY_TYPE_DEFAULT_SCOPE = {"short": "turn", "medium": "task", "long": "user"}

# source_span 与对话原文的最短/最长匹配长度（字符）
MIN_SPAN_CHARS = 2
MAX_SPAN_CHARS = 200

# ---------------------------------------------------------------------------
# 3. 纯文本工具
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
# 归一化时保留：中日韩文字、字母数字、下划线
_KEEP_RE = re.compile(r"[^\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+", re.UNICODE)


def normalize_text(text: Optional[str]) -> str:
    """NFKC 归一化 + 压缩空白，用于展示与比较。

    例：'以后  这个项目\\n都用中文' -> '以后 这个项目 都用中文'
    """
    if not text:
        return ""
    # 全角转半角、兼容字符统一
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\u200b", "")  # 去零宽空格
    return _WS_RE.sub(" ", text).strip()


def span_key(text: Optional[str]) -> str:
    """把文本压成"只保留有效字符"的键，用于 source_span 与原文的宽松匹配。

    这样 '以后这个项目都用中文回答。' 与 '以后 这个项目都用中文回答' 的键相同。
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return _KEEP_RE.sub("", text)


def truncate(text: Optional[str], max_chars: int = MAX_SPAN_CHARS) -> str:
    """按字符截断（source_span 第一版不训练生成头，仅规则截取，需限长）。"""
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


# ---------------------------------------------------------------------------
# 4. 对话 / 输入文本构造（训练与推理**必须**共用，否则分布不一致）
# ---------------------------------------------------------------------------

def format_candidates(candidate_memories: Optional[Sequence[Any]]) -> str:
    """把候选（已存在的）记忆列表压成一行文本。

    支持两种形态：
      - 字符串：["用户喜欢中文"]
      - 字典：  [{"memory_id": "m1", "raw_source": "...", "memory_type": "long"}]
    """
    if not candidate_memories:
        return "（无）"
    parts: List[str] = []
    for idx, cand in enumerate(candidate_memories, start=1):
        if isinstance(cand, dict):
            # 兼容若干字段名，尽量鲁棒
            text = (
                cand.get("raw_source")
                or cand.get("text")
                or cand.get("content")
                or ""
            )
            meta_bits = []
            if cand.get("memory_type"):
                meta_bits.append(str(cand["memory_type"]))
            if cand.get("scope"):
                meta_bits.append(str(cand["scope"]))
            if cand.get("confidence") is not None:
                try:
                    meta_bits.append(f"conf={float(cand['confidence']):.2f}")
                except (TypeError, ValueError):
                    pass
            if cand.get("memory_id"):
                meta_bits.append(str(cand["memory_id"]))
            meta = f" [{'/'.join(meta_bits)}]" if meta_bits else ""
            parts.append(f"{idx}. {normalize_text(text)}{meta}")
        else:
            parts.append(f"{idx}. {normalize_text(str(cand))}")
    return " ; ".join(parts)


def build_prompt(
    conversation: Sequence[Dict[str, str]],
    current_task: str = "",
    candidate_memories: Optional[Sequence[Any]] = None,
    max_history_turns: int = 12,
) -> str:
    """把一条样本拼成模型输入文本（训练 / 推理 / 评估统一入口）。

    模板（与需求文档一致）：

        [当前任务] {current_task}
        [对话历史]
        user: ...
        assistant: ...
        [候选记忆] ...
        [请判断是否写入记忆]
    """
    lines: List[str] = []
    lines.append(f"[当前任务] {normalize_text(current_task) or '（未指定）'}")
    lines.append("[对话历史]")

    convo = list(conversation or [])
    # 只保留最近 max_history_turns 轮，避免 padding 吃掉全部 512 token
    if max_history_turns and len(convo) > max_history_turns:
        convo = convo[-max_history_turns:]

    if not convo:
        lines.append("（无对话）")
    for msg in convo:
        role = normalize_text(str(msg.get("role", "user"))).lower() or "user"
        content = normalize_text(str(msg.get("content", "")))
        lines.append(f"{role}: {content}")

    lines.append(f"[候选记忆] {format_candidates(candidate_memories)}")
    lines.append("[请判断是否写入记忆]")
    return "\n".join(lines)


def conversation_user_texts(conversation: Sequence[Dict[str, str]]) -> List[str]:
    """返回对话里所有 user 消息的 content（source_span 的主要候选来源）。"""
    return [
        normalize_text(str(m.get("content", "")))
        for m in (conversation or [])
        if str(m.get("role", "")).lower() == "user"
    ]


def split_sentences(text: str) -> List[str]:
    """按中英文标点切句，供 source_span 规则截取使用。"""
    text = normalize_text(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[。！？!?；;\n])", text)
    return [p.strip() for p in pieces if p and p.strip()]


def bigrams(text: str) -> set:
    """有效字符的双字组合集合（中文无分词器时的轻量特征）。"""
    key = span_key(text)
    if len(key) < 2:
        return {key} if key else set()
    return {key[i:i + 2] for i in range(len(key) - 1)}


def char_similarity(a: str, b: str) -> float:
    """bigram 余弦相似度（0~1）。放在 common 里供 model/rules/memory_service 共用，
    避免这些模块之间互相 import。"""
    q, t = bigrams(a), bigrams(b)
    if not q or not t:
        return 0.0
    import math

    inter = len(q & t)
    return inter / math.sqrt(len(q) * len(t))


# ---------------------------------------------------------------------------
# 5. JSON / JSONL 读写
# ---------------------------------------------------------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """读 JSONL，跳过空行与坏行（坏行打印警告但不中断）。"""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] {path}:{lineno} JSON 解析失败，已跳过：{exc}")
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]], append: bool = False) -> int:
    """写 JSONL（ensure_ascii=False 保留中文），返回写入条数。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with open(path, mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def now_ts() -> float:
    """当前时间戳（秒，float）。集中在此便于测试时打桩。"""
    return time.time()


# ---------------------------------------------------------------------------
# 6. Memory 数据类（模块间传递的统一结构）
# ---------------------------------------------------------------------------

@dataclass
class Memory:
    """一条结构化记忆写入指令 / 一条记忆记录。

    第一版**不生成 value（归一化摘要）**，只保存 raw_source（原文片段）。
    """

    action: str = "ignore"                 # ignore / create / update / delete
    memory_type: str = "short"             # short / medium / long
    scope: str = "turn"                    # turn / session / task / user
    source_span: str = ""                  # 从对话中截取的原文片段
    confidence: float = 0.0                # 0~1
    # 以下字段由记忆服务/规则层填充，模型不直接预测
    memory_id: Optional[str] = None
    user_id: Optional[str] = None
    task_id: Optional[str] = None
    target_memory_id: Optional[str] = None  # update / delete 作用的旧记忆
    reason: str = ""                        # 规则层给出的可解释理由
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加信息（如检索键 key_text）

    # ---- 基本规整 -------------------------------------------------------
    def __post_init__(self) -> None:
        self.action = str(self.action).strip().lower()
        if self.action not in ACTION2ID:
            self.action = "ignore"
        self.memory_type = str(self.memory_type).strip().lower()
        if self.memory_type not in MEMORY_TYPE2ID:
            self.memory_type = "medium"
        self.scope = str(self.scope).strip().lower()
        if self.scope not in SCOPE2ID:
            self.scope = "task"
        try:
            self.confidence = float(self.confidence)
        except (TypeError, ValueError):
            self.confidence = 0.0
        # 裁剪到 [0,1]，避免模型/数据里的越界值污染下游
        self.confidence = min(1.0, max(0.0, self.confidence))
        self.source_span = truncate(self.source_span)

    # ---- 便捷属性 -------------------------------------------------------
    @property
    def should_write(self) -> bool:
        """是否真的需要落库（delete 也算一种"要执行"的动作）。"""
        return self.action != "ignore"

    def as_target(self) -> Dict[str, Any]:
        """输出成需求文档里的 target 结构（评测 / 数据校验用）。"""
        return {
            "action": self.action,
            "memory_type": self.memory_type,
            "scope": self.scope,
            "source_span": self.source_span,
            "confidence": round(self.confidence, 4),
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_target(cls, target: Dict[str, Any], **extra: Any) -> "Memory":
        """从 target 字典构造（忽略未知字段，缺字段用默认值）。"""
        target = target or {}
        return cls(
            action=target.get("action", "ignore"),
            memory_type=target.get("memory_type", "medium"),
            scope=target.get("scope", "task"),
            source_span=target.get("source_span", ""),
            confidence=target.get("confidence", 0.0),
            **extra,
        )

    def downgrade(self, reason: str) -> "Memory":
        """把一条记忆降级为 ignore，并记录原因（保留原始预测供审计）。"""
        self.notes.append(f"降级为 ignore：{reason}")
        self.reason = reason
        self.action = "ignore"
        return self

    def demote_type(self, reason: str) -> "Memory":
        """把长期记忆降级为 medium（规则层"长期记忆门槛"不满足时使用）。"""
        self.notes.append(f"记忆类型由 {self.memory_type} 降级为 medium：{reason}")
        self.memory_type = "medium"
        return self


# ---------------------------------------------------------------------------
# 7. 自检
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_convo = [
        {"role": "user", "content": "以后这个项目都用中文回答"},
        {"role": "assistant", "content": "好的"},
    ]
    print(build_prompt(demo_convo, "项目A技术讨论", []))
    print("-" * 60)
    print(span_key("以后 这个项目都用中文回答。") == span_key("以后这个项目都用中文回答"))
    m = Memory.from_target({"action": "CREATE", "memory_type": "long", "scope": "user",
                            "source_span": "以后这个项目都用中文回答", "confidence": 1.4})
    print(m.as_target())
    print(m.downgrade("测试").as_target())

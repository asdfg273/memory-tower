"""记忆塔 -> 代理自己的长期记忆：HTTP 服务 + stdio MCP 服务 + 命令行。

三种用法（同一套存储）：

  1) 命令行（最快，写完就能用）
     venv\\Scripts\\python.exe tools_tower_memory.py --db data\\tower_mem\\agent.jsonl remember "闭式写入的中位耗时是 0.19 ms"
     venv\\Scripts\\python.exe tools_tower_memory.py --db data\\tower_mem\\agent.jsonl recall "闭式写入有多快？"

  2) HTTP（给网页/其它程序）
     venv\\Scripts\\python.exe tools_tower_memory.py --serve --port 8801
     POST /remember {"text": "...", "key": "...", "questions": ["..."]}
     POST /recall   {"query": "...", "k": 3}
     GET  /stats | /list | POST /forget {"id": "..."}

  3) MCP（给 dsh 之类的宿主当插件加载，stdio）
     venv\\Scripts\\python.exe tools_tower_memory.py --mcp

存储 = 记忆塔本体：只存文本，加载时重算向量；写入走闭式岭回归
W = (KᵀK+λI)⁻¹KᵀZ，读出 r = q@W，再和值矩阵做余弦取 top-k。
（**默认裸嵌入**；只有显式加 `--proj` 才过塔的 w_k/w_v 随机投影——实测投影会掉点：
零重叠问法 hit@1 .500 → .250，见 reports/TOWER_MEM_PLUGIN.md 与 PAPER.md §6.95。）

设计取舍：
- **一条记忆默认只挂原文一个键**。追加问法当键会稀释读出（28→56→137 键单调变差，
  para 组 .111→.000），A′ 已证伪；要加键必须逐条验证收益。
- **只存文本**，向量在加载时重算（十几秒，含模型加载），这样文件可读、可 diff、可手改。
- 检索另给一路字符 bigram 分数（词面通道），融合方式 `fusion="gated"|"rrf"|"tower"`
  （默认 `gated`：有词面命中就按词面排，全零才问塔）。这条是实测结论，不是拍脑袋。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_DB = os.path.join("data", "tower_mem", "agent.jsonl")
TOWER_CACHE = os.path.join("checkpoints_tower")

# ----------------------------------------------------------------------------
# 底座：懒加载记忆塔
# ----------------------------------------------------------------------------


class Encoder:
    """把文本变成塔里的键/值向量。"""

    def __init__(self, m: int = 256, device: str = "cpu", proj: bool = True,
                 lam: float = 1e-2, verbose: bool = True) -> None:
        self.m, self.device, self.proj, self.lam = m, device, proj, lam
        self.verbose = verbose
        self.rt = None
        self._dim = None

    def _ensure(self) -> None:
        if self.rt is not None:
            return
        from tower_runtime import TowerRuntime

        t0 = time.time()
        self.rt = TowerRuntime(m=self.m, device=self.device,
                               write_mode="closed", key_mode="entity",
                               lam=self.lam, verbose=self.verbose,
                               cache_dir=TOWER_CACHE).build()
        self._dim = self.rt.dim
        if self.verbose:
            print(f"[mem] 塔就绪 {time.time() - t0:.1f}s · m={self.m} dim={self._dim} "
                  f"proj={'w_k/w_v' if self.proj else 'raw'}", flush=True)

    # 键/值：过塔的投影（默认）或裸嵌入
    def key(self, text: str):
        self._ensure()
        import torch

        k = self.rt.key(text)  # (dim,)
        if not self.proj:
            return k
        with torch.no_grad():
            return self.rt.mem.w_k(k.unsqueeze(0)).squeeze(0)

    def value(self, text: str):
        self._ensure()
        import torch

        v = self.rt.value_of(text)
        if not self.proj:
            return v
        with torch.no_grad():
            return self.rt.mem.w_v(v.unsqueeze(0)).squeeze(0)


def _bigrams(text: str) -> set:
    s = "".join(ch for ch in str(text) if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    return {s[i:i + 2] for i in range(len(s) - 1)} or ({s} if s else set())


def lexical_score(query: str, text: str) -> float:
    """字符 bigram 重合（含 idf 的味道：除以查询长度），纯词面兜底。"""
    a, b = _bigrams(query), _bigrams(text)
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


# ----------------------------------------------------------------------------
# 存储
# ----------------------------------------------------------------------------


class TowerMemory:
    def __init__(self, path: str = DEFAULT_DB, m: int = 256, device: str = "cpu",
                 proj: bool = True, lam: float = 1e-2, verbose: bool = True,
                 enc: Optional[Encoder] = None) -> None:
        self.path = path
        self.enc = enc or Encoder(m=m, device=device, proj=proj, lam=lam, verbose=verbose)
        self.verbose = verbose
        self.facts: List[Dict[str, Any]] = []
        self._K = None   # (N_keys, m)
        self._Z = None   # (N_facts, m)
        self._key_owner: List[int] = []   # 每个键属于哪条事实
        self._W = None   # 闭式解
        self._dirty = True
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.load()

    # ---- 持久化（只存文本，向量重算） ----
    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.facts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self._dirty = True

    def _append(self, rec: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _rewrite(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in self.facts:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    # ---- 写入 ----
    def add(self, text: str, key: Optional[str] = None,
            questions: Optional[Sequence[str]] = None,
            kind: str = "fact", source: str = "") -> Dict[str, Any]:
        text = str(text or "").strip()
        if not text:
            raise ValueError("text 不能为空")
        keys = [str(key or text).strip()]
        for q in (questions or []):
            q = str(q).strip()
            if q and q not in keys:
                keys.append(q)
        with self._lock:
            rec = {
                "id": "m_" + uuid.uuid4().hex[:12],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "text": text, "keys": keys, "kind": kind, "source": source,
                "alive": True,
            }
            self.facts.append(rec)
            self._append(rec)
            self._dirty = True
            if self.verbose:
                print(f"[mem] + {rec['id']} {text[:40]}（{len(keys)} 个键）", flush=True)
            return {k: v for k, v in rec.items() if k != "keys"} | {"n_keys": len(keys)}

    def forget(self, mid: str) -> bool:
        with self._lock:
            for rec in self.facts:
                if rec["id"] == mid:
                    rec["alive"] = False
                    self._rewrite()
                    self._dirty = True
                    return True
        return False

    # ---- 索引 ----
    def _ensure_index(self) -> None:
        if not self._dirty and self._W is not None:
            return
        import torch

        K, Z, owner = [], [], []
        for rec in self.facts:
            if not rec.get("alive", True):
                continue
            # owner 必须指向"存活列表"的行号（= Z 的行号），不是 self.facts 的行号：
            # 一旦 forget 过记录，两者就会错位，Zk 越界。
            row = len(Z)
            Z.append(self.enc.value(rec["text"]))
            for kt in rec.get("keys") or [rec["text"]]:
                K.append(self.enc.key(kt))
                owner.append(row)
        if not K:
            self._K = self._Z = self._W = None
            self._key_owner = []
            self._dirty = False
            return
        self._K = torch.stack(K, 0).float()
        self._Z = torch.stack(Z, 0).float()
        self._key_owner = owner
        # 一条事实可能有多个键：脊回归的目标矩阵要按"键"重复值（K 与 Z 同行数）
        Zk = self._Z[torch.tensor(owner, dtype=torch.long)]
        m = self._K.shape[1]
        G = self._K.t() @ self._K + self.enc.lam * torch.eye(m)
        self._W = torch.linalg.solve(G, self._K.t() @ Zk)
        self._dirty = False
        if self.verbose:
            print(f"[mem] 索引重建：{len(self._Z)} 条事实 / {len(self._K)} 个键", flush=True)

    # ---- 读出 ----
    def recall(self, query: str, k: int = 3, fusion: str = "gated") -> List[Dict[str, Any]]:
        """fusion:
        - "gated"（默认）：词面有信号就按词面排，词面全零才用塔（实测塔在零重合上也无用，
          但它至少不会在词面明明有信号时把答案挤下去）
        - "rrf"：两路排名倒数相加（尺度无关）
        - "tower"：只看塔
        """
        import torch

        query = str(query or "").strip()
        if not query:
            return []
        # JSON-RPC 的入参不可信：JSON 允许 "\ud800" 这种孤立代理项，直接喂给
        # tokenizer 会抛 TextEncodeInput TypeError。先剔掉不可编码的码位。
        if any(0xD800 <= ord(ch) <= 0xDFFF for ch in query):
            query = query.encode("utf-8", "ignore").decode("utf-8", "ignore")
            if not query:
                return []
        with self._lock:
            self._ensure_index()
            if self._W is None:
                return []
            with torch.no_grad():
                q = self.enc.key(query).float().unsqueeze(0)
                r = (q @ self._W).squeeze(0)
                r = torch.nn.functional.normalize(r, dim=-1)
                z = torch.nn.functional.normalize(self._Z, dim=-1)
                tower = (z @ r).tolist()
            alive = [rec for rec in self.facts if rec.get("alive", True)]
            lex = []
            for rec in alive:
                s = lexical_score(query, rec["text"])
                for kt in rec.get("keys") or []:
                    s = max(s, lexical_score(query, kt))
                lex.append(s)
            order_t = sorted(range(len(alive)), key=lambda i: -tower[i])
            order_l = sorted(range(len(alive)), key=lambda i: -lex[i])
            rank_t = {i: r for r, i in enumerate(order_t)}
            rank_l = {i: r for r, i in enumerate(order_l)}
            has_lex = max(lex) > 0 if lex else False
            out = []
            for i, rec in enumerate(alive):
                if fusion == "tower":
                    score = -rank_t[i]
                elif fusion == "rrf":
                    score = 1.0 / (60 + rank_t[i] + 1) + 1.0 / (60 + rank_l[i] + 1)
                else:  # gated：词面有信号就按词面排，否则切塔
                    r_use = rank_l[i] if has_lex else rank_t[i]
                    score = 1.0 / (r_use + 1)
                out.append({"id": rec["id"], "text": rec["text"], "kind": rec.get("kind"),
                            "score": round(score, 6),
                            "tower": round(float(tower[i]), 4),
                            "tower_rank": rank_t[i] + 1,
                            "lexical": round(lex[i], 4), "lex_rank": rank_l[i] + 1})
            out.sort(key=lambda x: x["score"], reverse=True)
            return out[:max(1, int(k))]

    # ---- 管理 ----
    def stats(self) -> Dict[str, Any]:
        alive = [r for r in self.facts if r.get("alive", True)]
        return {"db": os.path.abspath(self.path), "n": len(alive),
                "n_dead": len(self.facts) - len(alive),
                "m": self.enc.m, "proj": self.enc.proj,
                "indexed": self._W is not None}

    def listing(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [{"id": r["id"], "ts": r["ts"], "text": r["text"], "kind": r.get("kind")}
                for r in self.facts if r.get("alive", True)][-limit:]


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------


def serve(mem: TowerMemory, host: str = "0.0.0.0", port: int = 8801) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _json(self, obj: Any, code: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/stats"):
                self._json(mem.stats())
            elif self.path.startswith("/list"):
                self._json({"facts": mem.listing()})
            elif self.path.startswith(("/health", "/ping")):
                self._json({"ok": True, "ts": time.time(), **mem.stats()})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            try:
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                payload = {}
            try:
                if self.path.startswith("/remember"):
                    self._json(mem.add(payload.get("text", ""), key=payload.get("key"),
                                       questions=payload.get("questions"),
                                       kind=str(payload.get("kind") or "fact"),
                                       source=str(payload.get("source") or "api")))
                elif self.path.startswith("/recall"):
                    self._json({"hits": mem.recall(payload.get("query", ""),
                                                   k=int(payload.get("k") or 3))})
                elif self.path.startswith("/forget"):
                    self._json({"ok": mem.forget(str(payload.get("id") or ""))})
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as exc:
                self._json({"error": str(exc)}, 500)

    srv = ThreadingHTTPServer((host, port), H)
    print(f"[mem] HTTP 就绪 http://{host}:{port}  db={os.path.abspath(mem.path)}", flush=True)
    srv.serve_forever()


# ----------------------------------------------------------------------------
# MCP（stdio，JSON-RPC 2.0 换行分隔）
# ----------------------------------------------------------------------------

TOOLS = [
    {"name": "mem_remember",
     "description": "把一条事实写进记忆塔（长期记忆）。可给 key 与多个 questions 作为检索键。",
     "inputSchema": {"type": "object", "properties": {
         "text": {"type": "string", "description": "要记住的事实，一句第一人称/客观陈述"},
         "key": {"type": "string", "description": "主键（默认用 text 本身）"},
         "questions": {"type": "array", "items": {"type": "string"},
                       "description": "以后可能怎么问它（多给几条命中率更高）"},
         "kind": {"type": "string", "description": "fact/preference/plan/persona/address"}},
         "required": ["text"]}},
    {"name": "mem_recall",
     "description": "检索记忆塔里和 query 最相关的条目（塔读出 + 词面兜底）。",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"}, "k": {"type": "integer", "default": 3}},
         "required": ["query"]}},
    {"name": "mem_list", "description": "列出当前存活记忆。",
     "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}}},
    {"name": "mem_forget", "description": "按 id 忘掉一条记忆。",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}},
                     "required": ["id"]}},
    {"name": "mem_stats", "description": "库统计。",
     "inputSchema": {"type": "object", "properties": {}}},
]


def _tool_result(obj: Any) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(obj, ensure_ascii=False)}]}


def run_mcp(mem: TowerMemory) -> None:
    def handle(req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rid = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}

        def ok(result: Any) -> Dict[str, Any]:
            return {"jsonrpc": "2.0", "id": rid, "result": result}

        if method == "initialize":
            return ok({"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                       "serverInfo": {"name": "tower-memory", "version": "0.1.0"}})
        if method in ("notifications/initialized", "initialized"):
            return None
        if method == "ping":
            return ok({})
        if method == "tools/list":
            return ok({"tools": TOOLS})
        if method == "tools/call":
            name = params.get("name")
            a = params.get("arguments") or {}
            try:
                if name == "mem_remember":
                    return ok(_tool_result(mem.add(a.get("text", ""), key=a.get("key"),
                                                  questions=a.get("questions"),
                                                  kind=str(a.get("kind") or "fact"),
                                                  source="mcp")))
                if name == "mem_recall":
                    return ok(_tool_result({"hits": mem.recall(a.get("query", ""),
                                                             k=int(a.get("k") or 3))}))
                if name == "mem_list":
                    return ok(_tool_result({"facts": mem.listing(int(a.get("limit") or 50))}))
                if name == "mem_forget":
                    return ok(_tool_result({"ok": mem.forget(str(a.get("id") or ""))}))
                if name == "mem_stats":
                    return ok(_tool_result(mem.stats()))
                return {"jsonrpc": "2.0", "id": rid,
                        "error": {"code": -32601, "message": f"未知工具 {name}"}}
            except Exception as exc:  # noqa: BLE001
                return {"jsonrpc": "2.0", "id": rid,
                        "error": {"code": -32000, "message": str(exc)}}
        return {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": f"未知方法 {method}"}}

    print(f"[mem] MCP stdio 就绪 db={os.path.abspath(mem.path)}", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(req if isinstance(req, dict) else {})
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    # stdio 三件套一律钉成 UTF-8。Windows 上 stdout/stdin 默认走 cp936 +
    # surrogateescape（实测 stdin.encoding='gbk' errors='surrogateescape'），
    # 于是 MCP 通道上的中文 query 会被 GBK 拆成乱码，末尾落单的字节变成
    # \udcXX 代理项，tokenizer 直接抛
    # TypeError: TextEncodeInput must be Union[TextInputSequence, ...]。
    # CLI 的 query 走 argv 不经 stdin，把 stdin 也钉成 UTF-8 不影响命令行用法。
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    p = argparse.ArgumentParser(description="记忆塔 = 代理自己的长期记忆（CLI / HTTP / MCP）")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--m", type=int, default=256, help="塔宽度")
    p.add_argument("--device", default="cpu", help="cpu（默认，省显存）/ cuda")
    p.add_argument("--proj", action="store_true",
                   help="用塔的 w_k/w_v 随机投影（默认关：实测投影会掉点，见 reports/TOWER_MEM_PLUGIN.md）")
    p.add_argument("--lam", type=float, default=1e-2)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8801)
    p.add_argument("--mcp", action="store_true")
    sub = p.add_subparsers(dest="cmd")
    a1 = sub.add_parser("remember")
    a1.add_argument("text")
    a1.add_argument("--key", default=None)
    a1.add_argument("--q", action="append", default=[], help="问法，可重复")
    a1.add_argument("--kind", default="fact")
    a2 = sub.add_parser("recall")
    a2.add_argument("query")
    a2.add_argument("-k", type=int, default=3)
    a3 = sub.add_parser("list")
    a3.add_argument("--limit", type=int, default=50)
    a4 = sub.add_parser("forget")
    a4.add_argument("id")
    a4.add_argument("--yes", action="store_true", help="确认删除（不加只打印）")
    sub.add_parser("stats")
    # 让 --quiet 也能写在子命令后面（SUPPRESS 避免子解析器把父级设置覆盖成默认值）
    for sp in (a1, a2, a3, a4):
        sp.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS)
    args = p.parse_args(argv)

    mem = TowerMemory(args.db, m=args.m, device=args.device, proj=args.proj,
                      lam=args.lam, verbose=not args.quiet)
    if args.mcp:
        run_mcp(mem)
        return 0
    if args.serve:
        serve(mem, args.host, args.port)
        return 0
    if args.cmd == "remember":
        print(json.dumps(mem.add(args.text, key=args.key, questions=args.q,
                                 kind=args.kind, source="cli"), ensure_ascii=False, indent=2))
    elif args.cmd == "recall":
        print(json.dumps({"hits": mem.recall(args.query, k=args.k)}, ensure_ascii=False, indent=2))
    elif args.cmd == "list":
        print(json.dumps({"facts": mem.listing(args.limit)}, ensure_ascii=False, indent=2))
    elif args.cmd == "forget":
        if not args.yes:
            print(json.dumps({"would_forget": args.id, "hint": "加 --yes 真删"}, ensure_ascii=False))
        else:
            print(json.dumps({"ok": mem.forget(args.id)}, ensure_ascii=False))
    else:
        print(json.dumps(mem.stats(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

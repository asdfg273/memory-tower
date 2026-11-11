"""把「我们自己的对话」当记忆库：真实长流上的检索对照（H11）。

语料：DSH 会话导出的 2055 条消息（`_ab_tmp/session_corpus.jsonl`，约 110 万字符）。
问题：从语料里抽 24 条"金句"，让 7B 模型**不改措辞地**把它反过来问成一个问题
（"这段话回答了什么"），再让四个检索臂去找回它。

臂：
- `idf_msg`  ：idf 加权 1+2 元，整条消息为一个单元（= 产品 `text_score_idf` 的口径）
- `idf_sent` ：同一打分器，句子级单元，取该消息内最高分（= §5.17 的 `G_sent`）
- `vec`      ：nomic 嵌入，整条消息余弦
- `tower`    ：记忆塔（m=256、闭式写入、`fusion="gated"`），每条消息写一条事实

指标：hit@1 / hit@3 / hit@5 / MRR，以及每臂单次查询耗时。

用法：
    venv\\Scripts\\python.exe tools_session_recall.py --arms idf_msg,idf_sent,vec
    venv\\Scripts\\python.exe tools_session_recall.py --arms tower --out _ab_tmp/session_recall.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_service import _char_grams, _cosine_idf, _term_freq  # noqa: E402

CORPUS = Path("_ab_tmp/session_corpus.jsonl")
LM_BASE = "http://127.0.0.1:1234/v1"
Q_MODEL = "qwen2.5-vl-7b"
EMB_MODEL = "text-embedding-nomic-embed-text-v1.5"

SENT_SPLIT = re.compile(r"[。！？!?\n]+")
Q_PROMPT = (
    "下面是一段技术对话里的一条记录。请写出它回答的那个问题。\n"
    "要求：\n"
    "- 一句话，20 字以内；\n"
    "- 不要照抄原文的措辞，用别人自然会怎么问的方式；\n"
    "- 只输出这个问题本身，不要解释、不要引号。\n\n"
    "记录：\n{text}"
)


# ---------------------------------------------------------------- 语料与问题
def load_corpus(path: Path = CORPUS) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec["text"]
            if len(text) < 80 or len(text) > 4000:
                continue
            items.append(rec)
    return items


def pick_gold(items: List[Dict[str, Any]], n: int, seed: int = 0) -> List[Dict[str, Any]]:
    """挑"有信息量"的消息：带数字、带结论词，且不是纯寒暄。"""
    rng = random.Random(seed)
    cand = []
    for it in items:
        t = it["text"]
        has_num = bool(re.search(r"\d", t))
        has_kw = any(w in t for w in ("结论", "结果", "判据", "因为", "改成", "实测", "掉到", "抬到"))
        if has_num and has_kw and 120 <= len(t) <= 900:
            cand.append(it)
    rng.shuffle(cand)
    # 每个会话最多 6 条，避免全挤在一个 session 里
    picked, per_sess = [], {}
    for it in cand:
        sid = it["ref"].split(":")[0]
        if per_sess.get(sid, 0) >= max(2, n // 4):
            continue
        per_sess[sid] = per_sess.get(sid, 0) + 1
        picked.append(it)
        if len(picked) >= n:
            break
    return picked


def _post(path: str, payload: dict, timeout: int = 180) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        LM_BASE + path, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def make_queries(golds: List[Dict[str, Any]], cache: Path) -> Dict[str, str]:
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if len(data) == len(golds):
            print(f"[query] reuse {cache} ({len(data)} 条)")
            return data
    out: Dict[str, str] = {}
    for i, g in enumerate(golds, 1):
        try:
            r = _post(
                "/chat/completions",
                {
                    "model": Q_MODEL,
                    "messages": [
                        {"role": "user", "content": Q_PROMPT.format(text=g["text"][:1200])}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 60,
                },
            )
            q = r["choices"][0]["message"]["content"].strip().splitlines()[0].strip(' "“”')
        except Exception as exc:  # noqa: BLE001
            print(f"[query] {i} 失败：{type(exc).__name__}")
            q = ""
        out[g["ref"]] = q
        print(f"[query] {i}/{len(golds)} {q[:40]}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------- 打分器
class IdfIndex:
    def __init__(self, texts: List[str]):
        import math

        self.n = max(1, len(texts))
        df1: Dict[str, int] = {}
        df2: Dict[str, int] = {}
        for t in texts:
            for g in _char_grams(t, 1):
                df1[g] = df1.get(g, 0) + 1
            for g in _char_grams(t, 2):
                df2[g] = df2.get(g, 0) + 1
        self.idf1 = {g: math.log((self.n + 1) / (c + 0.5)) for g, c in df1.items()}
        self.idf2 = {g: math.log((self.n + 1) / (c + 0.5)) for g, c in df2.items()}

    def score(self, query: str, text: str, w_uni: float = 0.5) -> float:
        if not query or not text:
            return 0.0
        q1: Dict[str, int] = {}
        for g in _char_grams(query, 1):
            q1[g] = q1.get(g, 0) + 1
        q2: Dict[str, int] = {}
        for g in _char_grams(query, 2):
            q2[g] = q2.get(g, 0) + 1
        s1 = _cosine_idf(q1, _term_freq(text, 1), self.idf1)
        s2 = _cosine_idf(q2, _term_freq(text, 2), self.idf2)
        return w_uni * s1 + (1.0 - w_uni) * s2


def embed(texts: List[str], cache: Path, batch: int = 8) -> List[List[float]]:
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if len(data) == len(texts):
            print(f"[vec] reuse {cache}")
            return data
    out: List[List[float]] = []
    t0 = time.time()
    for i in range(0, len(texts), batch):
        chunk = [t[:2000] for t in texts[i : i + batch]]
        r = _post("/embeddings", {"model": EMB_MODEL, "input": chunk}, timeout=300)
        out.extend(d["embedding"] for d in r["data"])
        if (i // batch) % 20 == 0:
            print(f"[vec] {i + len(chunk)}/{len(texts)}  {time.time() - t0:.0f}s")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def cosine(a: List[float], b: List[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def rank_of(order: List[str], gold: str) -> int:
    for i, ref in enumerate(order[:50], 1):
        if ref == gold:
            return i
    return 0


TOKEN_RE = re.compile(r"[0-9][0-9.,]*[0-9]|[A-Za-z_][A-Za-z0-9_.]{3,}")


def build_token_df(texts: List[str]) -> Dict[str, int]:
    df: Dict[str, int] = {}
    for t in texts:
        for tok in set(TOKEN_RE.findall(t)):
            df[tok] = df.get(tok, 0) + 1
    return df


def distinctive_tokens(text: str, df: Dict[str, int], k: int = 3) -> List[str]:
    """挑几个"在这条消息里出现、但在全语料里很少见"的 token 当答案指纹。"""
    toks = [t for t in set(TOKEN_RE.findall(text)) if 3 <= len(t) <= 40]
    toks.sort(key=lambda t: (df.get(t, 10**9), -len(t)))
    return [t for t in toks if df.get(t, 10**9) <= 3][:k]


def rank_in_set(order: List[str], gold_set: set, limit: int = 50) -> int:
    for i, ref in enumerate(order[:limit], 1):
        if ref in gold_set:
            return i
    return 0


def metrics(ranks: List[int]) -> Dict[str, float]:
    n = max(1, len(ranks))
    return {
        "hit@1": round(sum(1 for r in ranks if r == 1) / n, 4),
        "hit@3": round(sum(1 for r in ranks if 1 <= r <= 3) / n, 4),
        "hit@5": round(sum(1 for r in ranks if 1 <= r <= 5) / n, 4),
        "mrr": round(sum((1.0 / r) for r in ranks if r > 0) / n, 4),
        "miss": sum(1 for r in ranks if r == 0),
    }


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="idf_msg,idf_sent,vec")
    ap.add_argument("--n-gold", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default="_ab_tmp/session_recall.json")
    args = ap.parse_args()

    items = load_corpus()
    print(f"[corpus] messages={len(items)} chars={sum(len(i['text']) for i in items):,}")
    golds = pick_gold(items, args.n_gold, args.seed)
    print(f"[corpus] 金句 {len(golds)} 条，覆盖会话 {len({g['ref'].split(':')[0] for g in golds})} 个")

    queries = make_queries(golds, Path("_ab_tmp/session_queries.json"))
    texts = [it["text"] for it in items]
    refs = [it["ref"] for it in items]

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    report: Dict[str, Any] = {
        "corpus": {"messages": len(items), "chars": sum(len(t) for t in texts)},
        "gold": [g["ref"] for g in golds],
        "queries": queries,
        "arms": {},
    }

    if "idf_msg" in arms or "idf_sent" in arms:
        idx = IdfIndex(texts)
        print("[idf] df 建好")

    tok_df = build_token_df(texts)
    gold_sets: Dict[str, set] = {}
    for g in golds:
        toks = distinctive_tokens(g["text"], tok_df)
        hits = {r for t, r in zip(texts, refs) if any(tok in t for tok in toks)}
        gold_sets[g["ref"]] = hits | {g["ref"]}
    sizes = [len(v) for v in gold_sets.values()]
    print(f"[gold] 答案级同义集合 n={len(sizes)} 中位 {statistics.median(sizes):.0f}")
    report["gold_sets"] = {k: sorted(v) for k, v in gold_sets.items()}

    def finish(name: str, orders: List[List[str]], ms: float, extra: Dict[str, Any] | None = None) -> None:
        strict = [rank_of(o, g["ref"]) for o, g in zip(orders, golds)]
        answer = [rank_in_set(o, gold_sets[g["ref"]]) for o, g in zip(orders, golds)]
        info: Dict[str, Any] = {
            "metrics": metrics(strict),
            "answer_level": metrics(answer),
            "ms_per_query": ms,
            "hits": [o[:10] for o in orders],
        }
        if extra:
            info.update(extra)
        report["arms"][name] = info
        print(f"[{name}] strict={info['metrics']} answer={info['answer_level']}")

    if "bigram" in arms or "bigram_sent" in arms:
        from tools_tower_memory import lexical_score

    if "bigram" in arms:
        t0 = time.time()
        orders = []
        for g in golds:
            q = queries[g["ref"]]
            scores = sorted(((lexical_score(q, t), r) for t, r in zip(texts, refs)), reverse=True)
            orders.append([r for _, r in scores][:50])
        finish("bigram_msg", orders, round((time.time() - t0) * 1000 / len(golds), 1))

    if "bigram_sent" in arms:
        sent_texts, sent_refs = [], []
        for t, r in zip(texts, refs):
            for s in SENT_SPLIT.split(t):
                s = s.strip()
                if len(s) >= 12:
                    sent_texts.append(s)
                    sent_refs.append(r)
        t0 = time.time()
        orders = []
        for g in golds:
            q = queries[g["ref"]]
            best: Dict[str, float] = {}
            for s, r in zip(sent_texts, sent_refs):
                sc = lexical_score(q, s)
                if sc > best.get(r, 0.0):
                    best[r] = sc
            orders.append([r for r, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)][:50])
        finish("bigram_sent", orders, round((time.time() - t0) * 1000 / len(golds), 1))

    if "idf_msg" in arms:
        t0 = time.time()
        orders = []
        for g in golds:
            q = queries[g["ref"]]
            scores = sorted(((idx.score(q, t), r) for t, r in zip(texts, refs)), reverse=True)
            orders.append([r for _, r in scores][:50])
        finish("idf_msg", orders, round((time.time() - t0) * 1000 / len(golds), 1))

    if "idf_sent" in arms:
        sent_texts, sent_refs = [], []
        for t, r in zip(texts, refs):
            for s in SENT_SPLIT.split(t):
                s = s.strip()
                if len(s) >= 12:
                    sent_texts.append(s)
                    sent_refs.append(r)
        sidx = IdfIndex(sent_texts)
        print(f"[idf_sent] 句子单元 {len(sent_texts)}")
        t0 = time.time()
        orders = []
        for g in golds:
            q = queries[g["ref"]]
            best: Dict[str, float] = {}
            for s, r in zip(sent_texts, sent_refs):
                sc = sidx.score(q, s)
                if sc > best.get(r, 0.0):
                    best[r] = sc
            orders.append([r for r, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)][:50])
        finish("idf_sent", orders, round((time.time() - t0) * 1000 / len(golds), 1))

    if "vec" in arms:
        vecs = embed(texts, Path("_ab_tmp/session_vec.json"))
        t0 = time.time()
        orders = []
        for g in golds:
            q = queries[g["ref"]]
            qv = _post("/embeddings", {"model": EMB_MODEL, "input": [q]}, timeout=120)["data"][0][
                "embedding"
            ]
            scores = sorted(((cosine(qv, v), r) for v, r in zip(vecs, refs)), reverse=True)
            orders.append([r for _, r in scores][:50])
        finish("vec", orders, round((time.time() - t0) * 1000 / len(golds), 1))

    if "tower" in arms:
        from tools_tower_memory import TowerMemory

        db = "_ab_tmp/session_tower.jsonl"
        if os.path.exists(db):
            os.remove(db)
        t0 = time.time()
        mem = TowerMemory(db, m=256, proj=True, verbose=False)
        ref_of: Dict[str, str] = {}
        for it in items:
            # key 必须省略：传 text[:24] 这种短键会把词面兜底刷高（分母只有 20 多个 bigram），
            # 让 gated 臂比同打分器的 bigram 臂低一截，纯粹是测量伪影。
            rec = mem.add(it["text"], kind="session", source="corpus")
            ref_of[rec["id"]] = it["ref"]
        build_s = time.time() - t0
        print(f"[tower] 写入 {len(ref_of)} 条，{build_s:.0f}s")
        for fusion in ("gated", "tower"):
            t0 = time.time()
            orders = []
            for g in golds:
                q = queries[g["ref"]]
                hits = mem.recall(q, k=max(args.topk, 20), fusion=fusion)
                orders.append([ref_of.get(h["id"], "?") for h in hits][:50])
            finish(
                f"tower_{fusion}",
                orders,
                round((time.time() - t0) * 1000 / len(golds), 1),
                {"build_s": round(build_s, 1), "fusion": fusion},
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

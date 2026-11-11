"""批量版金句提问生成器（DeepSeek R1 distill 7B，每批 6 段话一次请求）。

为什么批量：单条请求 26.6s，其中大部分是 reasoning 开销；把 6 段话放进一次请求，
同样的推理预算能换来 6 个问题（实测 4 段话 30s、四行严格对齐）。

协议（写进 reports/SESSION_RECALL.md §12.7 的扩展）：
- 候选消息：100-800 字、含 df<=5 的稀有二字组、按 ref 去重；
- 分批：每批 6 条，模型必须按输入顺序输出同样行数，行首允许 `[i]`；
- 校验：长度 4-40、不含"这段/上面/如下"、与原文的最长公共子串 < 12 字；
- 去重：按问题文本精确去重，再按 ref 去重；
- 冻结：输出 JSON 内含 model / ts / counts / sha256 由调用方另行计算。

用法：
    venv\\Scripts\\python.exe tools_session_queries_batch.py --n 240 --batch 6 \
        --model deepseek-r1-distill-qwen-7b --out _ab_tmp/gold_final.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CORPUS = Path("_ab_tmp/session_corpus.jsonl")
API = "http://127.0.0.1:1234/v1/chat/completions"
SYSTEM = (
    "你是数据集标注员。用户会给若干段话，你为每段话写一个「用户当时可能问的问题」。"
    "要求：20 字以内；必须带具体名词或数字；不要照抄原句；每行一个问题，按输入顺序排列；"
    "只输出问题本身，不要编号、不要解释。"
)
BAD = ("这段", "上面", "如下", "以下", "原文")


def load_corpus(path: Path = CORPUS) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        t = (d.get("text") or "").strip()
        if 100 <= len(t) <= 800 and d.get("ref"):
            items.append({"ref": d["ref"], "text": t})
    return items


def char_grams(s: str, n: int) -> set[str]:
    s = " " + s.replace("\n", " ") + " "
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def candidates(items: list[dict], n: int, seed: int = 0) -> list[dict]:
    df: dict[str, int] = {}
    for it in items:
        for g in set(char_grams(it["text"], 2)):
            df[g] = df.get(g, 0) + 1
    seen_refs: set[str] = set()
    out: list[dict] = []
    for it in sorted(items, key=lambda x: x["ref"]):
        if it["ref"] in seen_refs:
            continue
        rare = [g for g in set(char_grams(it["text"], 2)) if df.get(g, 99) <= 5]
        if len(rare) < 2:
            continue
        seen_refs.add(it["ref"])
        out.append(it)
    random.Random(seed).shuffle(out)
    return out[:n]


def post(model: str, texts: list[str], max_tokens: int = 900) -> str:
    user = "\n".join(f"[{i + 1}] {t[:400]}" for i, t in enumerate(texts))
    body = json.dumps({"model": model,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": user}],
                       "max_tokens": max_tokens, "temperature": 0.5}).encode("utf-8")
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or ""


def clean(line: str) -> str:
    s = line.strip()
    if s.startswith("["):
        s = s.split("]", 1)[-1].strip()
    s = s.lstrip("0123456789.、) ").strip()
    return s.strip("\"'“”「」* ")


def lcs_len(a: str, b: str) -> int:
    """最长公共子串长度（>12 视为照抄）。"""
    if not a or not b:
        return 0
    best = 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def run_batch(model: str, batch: list[dict], args) -> list[dict]:
    for _ in range(3):
        try:
            raw = post(model, [b["text"] for b in batch], args.max_tokens)
        except Exception as exc:  # noqa: BLE001
            print(f"[retry] {exc}", flush=True)
            time.sleep(3)
            continue
        lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
        if len(lines) < len(batch):
            continue
        got = []
        for it, ln in zip(batch, lines):
            q = clean(ln)
            if not (4 <= len(q) <= 40) or any(b in q for b in BAD):
                continue
            if lcs_len(q, it["text"]) > 12:
                continue
            got.append({"ref": it["ref"], "query": q, "msg_len": len(it["text"])})
        if got:
            return got
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--model", default="deepseek-r1-distill-qwen-7b")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--out", default="_ab_tmp/gold_final.json")
    args = ap.parse_args()

    items = load_corpus()
    cands = candidates(items, args.n)
    print(f"[cand] {len(cands)} 条候选（语料 {len(items)}）", flush=True)
    batches = [cands[i:i + args.batch] for i in range(0, len(cands), args.batch)]
    print(f"[batch] {len(batches)} 批 × {args.batch} 条，workers={args.workers}", flush=True)

    t0 = time.time()
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_batch, args.model, b, args) for b in batches]
        for i, fu in enumerate(futs, 1):
            out.extend(fu.result())
            if i % 5 == 0:
                print(f"[prog] {i}/{len(batches)} 批，已得 {len(out)} 条，{time.time() - t0:.0f}s",
                      flush=True)

    seen: set[str] = set()
    final: list[dict] = []
    refs: set[str] = set()
    for it in out:
        if it["query"] in seen or it["ref"] in refs:
            continue
        seen.add(it["query"])
        refs.add(it["ref"])
        final.append(it)

    payload = {"model": args.model, "n": len(final), "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
               "protocol": {"batch": args.batch, "max_tokens": args.max_tokens,
                            "filters": "4-40 字 / 无指代词 / LCS<=12 / 去重",
                            "candidates": len(cands)},
               "items": final}
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {len(final)} 条 query -> {p}", flush=True)
    print(f"[time] {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""H11 的缩放曲线：语料从 200 条增到全量，各检索臂的命中与耗时怎么变。

复用 `tools_session_recall.py` 的语料、金句、问题生成与打分器，只按前缀截断语料，
在每个规模上重算 hit@1 / hit@3 / MRR（严格 = 命中原消息；答案级 = 命中任一同义消息）
与单次查询耗时。

用法：
    venv\\Scripts\\python.exe tools_session_recall_sweep.py --sizes 200,400,800,1343
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tools_session_recall as base


def gold_sets_for(texts: List[str], refs: List[str], golds: List[Dict[str, Any]]) -> Dict[str, set]:
    tok_df = base.build_token_df(texts)
    out = {}
    for g in golds:
        toks = base.distinctive_tokens(g["text"], tok_df)
        out[g["ref"]] = {r for t, r in zip(texts, refs) if any(tok in t for tok in toks)} | {g["ref"]}
    return out


def sentences(texts: List[str], refs: List[str]) -> Tuple[List[str], List[str]]:
    units, urefs = [], []
    for t, r in zip(texts, refs):
        for s in base.SENT_SPLIT.split(t):
            s = s.strip()
            if len(s) >= 12:
                units.append(s)
                urefs.append(r)
    return units, urefs


def rank_idf_message(idx: base.IdfIndex, q: str, texts: List[str], refs: List[str]) -> List[str]:
    scores = sorted(((idx.score(q, t), r) for t, r in zip(texts, refs)), reverse=True)
    return [r for _, r in scores][:50]


def rank_idf_sentence(idx: base.IdfIndex, q: str, units: List[str], urefs: List[str]) -> List[str]:
    best: Dict[str, float] = {}
    for s, r in zip(units, urefs):
        sc = idx.score(q, s)
        if sc > best.get(r, 0.0):
            best[r] = sc
    return [r for r, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)][:50]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="200,400,800,1343")
    ap.add_argument("--n-gold", type=int, default=24)
    ap.add_argument("--out", default="_ab_tmp/session_recall_sweep.json")
    args = ap.parse_args()

    items_all = base.load_corpus()
    golds_all = base.pick_gold(items_all, args.n_gold, 0)
    queries = base.make_queries(golds_all, Path("_ab_tmp/session_queries.json"))

    # 向量只嵌一次，按文本切片复用
    all_texts = [it["text"] for it in items_all]
    all_vecs = base.embed(all_texts, Path("_ab_tmp/session_vec.json"))
    vec_of = dict(zip(all_texts, all_vecs))

    report: Dict[str, Any] = {"n_gold_total": len(golds_all), "sizes": {}}

    for size in [int(s) for s in args.sizes.split(",") if s.strip()]:
        items = items_all[:size]
        texts = [it["text"] for it in items]
        refs = [it["ref"] for it in items]
        keep = set(refs)
        golds = [g for g in golds_all if g["ref"] in keep]
        if not golds:
            continue
        gsets = gold_sets_for(texts, refs, golds)
        units, urefs = sentences(texts, refs)
        idx_msg = base.IdfIndex(texts)
        idx_sent = base.IdfIndex(units)

        row: Dict[str, Any] = {
            "messages": len(items),
            "golds": len(golds),
            "chars": sum(len(t) for t in texts),
        }

        for name, fn in (
            ("idf_msg", lambda q: rank_idf_message(idx_msg, q, texts, refs)),
            ("idf_sent", lambda q: rank_idf_sentence(idx_sent, q, units, urefs)),
        ):
            t0 = time.time()
            strict, answer = [], []
            for g in golds:
                order = fn(queries[g["ref"]])
                strict.append(base.rank_of(order, g["ref"]))
                answer.append(base.rank_in_set(order, gsets[g["ref"]]))
            row[name] = {
                "strict": base.metrics(strict),
                "answer_level": base.metrics(answer),
                "units": len(texts) if name == "idf_msg" else len(units),
                "ms_per_query": round((time.time() - t0) * 1000 / len(golds), 1),
            }
            print(f"[{size}][{name}] strict={row[name]['strict']['hit@1']} "
                  f"answer={row[name]['answer_level']['hit@1']} {row[name]['ms_per_query']}ms")

        vecs = [vec_of[t] for t in texts]
        t0 = time.time()
        strict, answer = [], []
        for g in golds:
            qv = base._post("/embeddings", {"model": base.EMB_MODEL, "input": [queries[g["ref"]]]}, timeout=120)[
                "data"
            ][0]["embedding"]
            scores = sorted(((base.cosine(qv, v), r) for v, r in zip(vecs, refs)), reverse=True)
            order = [r for _, r in scores][:50]
            strict.append(base.rank_of(order, g["ref"]))
            answer.append(base.rank_in_set(order, gsets[g["ref"]]))
        row["vec"] = {
            "strict": base.metrics(strict),
            "answer_level": base.metrics(answer),
            "ms_per_query": round((time.time() - t0) * 1000 / len(golds), 1),
        }
        print(f"[{size}][vec] strict={row['vec']['strict']['hit@1']} "
              f"answer={row['vec']['answer_level']['hit@1']} {row['vec']['ms_per_query']}ms")

        report["sizes"][str(size)] = row

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

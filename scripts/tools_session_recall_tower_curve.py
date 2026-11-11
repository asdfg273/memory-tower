"""塔的纯读出在真实语料上的容量曲线：固定 m=256，把语料压成 N 条时命中怎么掉。

对每个 N（100/200/400/800/1343）：池子 = 全部 38 条金句 + 随机补到 N 条的其他消息，
写入塔（closed 写入、m=256），用同一批问题做纯塔读出（`fusion="tower"`），
报 hit@1 / hit@3 / MRR 与 h=N/m。

用法：
    venv\\Scripts\\python.exe tools_session_recall_tower_curve.py --sizes 100,200,400,800,1343
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import tools_session_recall as base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100,200,400,800,1343")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n-gold", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="_ab_tmp/session_tower_curve.json")
    args = ap.parse_args()

    items = base.load_corpus()
    golds = base.pick_gold(items, args.n_gold, 0)
    queries = base.make_queries(golds, Path("_ab_tmp/session_queries.json"))
    gold_refs = {g["ref"] for g in golds}
    others = [it for it in items if it["ref"] not in gold_refs]
    rng = random.Random(args.seed)

    from tools_tower_memory import TowerMemory

    report: Dict[str, Any] = {"m": args.m, "n_gold": len(golds), "sizes": {}}
    for size in [int(s) for s in args.sizes.split(",") if s.strip()]:
        pool = list(golds) + rng.sample(others, max(0, min(size, len(items)) - len(golds)))
        pool = pool[:size]
        db = f"_ab_tmp/session_tower_{size}.jsonl"
        if Path(db).exists():
            Path(db).unlink()
        t0 = time.time()
        mem = TowerMemory(db, m=args.m, proj=True, verbose=False)
        ref_of: Dict[str, str] = {}
        for it in pool:
            rec = mem.add(it["text"], kind="session", source="corpus")
            ref_of[rec["id"]] = it["ref"]
        # 首次 recall 把索引建起来（懒加载），把时间记进 build
        mem.recall("预热", k=1, fusion="tower")
        build_s = time.time() - t0

        t0 = time.time()
        ranks: List[int] = []
        for g in golds:
            hits = mem.recall(queries[g["ref"]], k=20, fusion="tower")
            order = [ref_of.get(h["id"], "?") for h in hits]
            ranks.append(base.rank_of(order, g["ref"]))
        ms = round((time.time() - t0) * 1000 / len(golds), 1)
        row = {
            "N": len(pool),
            "h": round(len(pool) / args.m, 2),
            "metrics": base.metrics(ranks),
            "ms_per_query": ms,
            "build_s": round(build_s, 1),
        }
        report["sizes"][str(size)] = row
        print(f"[{size}] h={row['h']} hit@1={row['metrics']['hit@1']} hit@3={row['metrics']['hit@3']} "
              f"mrr={row['metrics']['mrr']} build={row['build_s']}s q={ms}ms")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

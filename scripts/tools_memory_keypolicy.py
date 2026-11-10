"""问法键的用法对照（不改产品代码，只算排序）。

同一批 38 条 query、同一份 data/memory.db 副本，比较几种"键怎么参与打分"的策略：
  raw        只用事实正文（= 改动前的 text_score_idf 口径）
  max        正文与各问法取最大（= 当前 record_similarity）
  disc0.85   问法打八五折后取最大
  disc0.7    问法打七折
  gate0.15   正文分 < 0.15 时才看问法，否则只看正文
  gate0.30   阈值 0.30
输出 _ab_tmp/key_policy.json 与终端表。
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from typing import Dict, List

sys.path.insert(0, os.getcwd())
import tools_memory_ab as ab  # noqa: E402
from memory_service import MemoryService  # noqa: E402

TMP = r"_ab_tmp\key_policy.db"
OUT = r"_ab_tmp\key_policy.json"


def main() -> int:
    if os.path.exists(TMP):
        os.remove(TMP)
    shutil.copyfile(os.path.join("data", "memory.db"), TMP)
    svc = MemoryService(TMP)
    rows = svc._conn.execute(
        "SELECT memory_id, raw_source, keys FROM memories WHERE deleted_at IS NULL").fetchall()
    recs: Dict[str, Dict[str, object]] = {}
    for r in rows:
        try:
            keys = json.loads(r["keys"] or "[]")
        except Exception:  # noqa: BLE001
            keys = []
        recs[r["memory_id"]] = {"raw": r["raw_source"], "keys": [k for k in keys if isinstance(k, str)]}
    orig = ab.load_orig(os.path.join("data", "memory_orig.json"))
    mems = [{"memory_id": mid, "text": v["raw"], "raw_source": v["raw"],
             "orig": orig.get(mid, "")} for mid, v in recs.items()]
    print(f"[db] {len(recs)} 条, 带键 {sum(1 for v in recs.values() if v['keys'])} 条", flush=True)

    def sim(policy: str, query: str, mid: str) -> float:
        v = recs[mid]
        raw = svc.text_similarity(query, v["raw"])
        if not v["keys"]:
            return raw
        ks = [svc.text_similarity(query, k) for k in v["keys"]]
        best = max(ks)
        if policy == "raw":
            return raw
        if policy == "max":
            return max(raw, best)
        if policy.startswith("disc"):
            w = float(policy[4:])
            return max(raw, w * best)
        if policy.startswith("gate"):
            th = float(policy[4:])
            return max(raw, best) if raw < th else raw
        raise ValueError(policy)

    POLICIES = ["raw", "max", "disc0.85", "disc0.7", "gate0.15", "gate0.30"]
    out: Dict[str, Dict[str, float]] = {}
    for policy in POLICIES:
        per_group: Dict[str, List[int]] = {}
        for group, queries in ab.GROUPS:
            for query, snippets in queries:
                targets = ab.resolve(mems, snippets)
                ranked = sorted(recs, key=lambda m: -sim(policy, query, m))[:5]
                per_group.setdefault(group, []).append(ab.rank_of(targets, ranked))
        pooled = [r for g in per_group.values() for r in g]
        weak = [r for g in per_group.values() for r in g]   # 与 bench 的 weak 子集不同，见下
        out[policy] = {
            "pooled": ab.summarize(pooled),
            "keep": ab.summarize(per_group.get("keep", [])),
            "para": ab.summarize(per_group.get("para", [])),
        }
        s = out[policy]
        print(f"[{policy:9s}] pooled {s['pooled']['hit@1']:.4f}/{s['pooled']['hit@3']:.4f} "
              f"keep {s['keep']['hit@1']:.4f} para {s['para']['hit@1']:.4f}", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("[out]", OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

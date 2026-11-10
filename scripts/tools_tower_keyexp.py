"""把改写问句当"键"写进塔，看能不能提升**未见改写**的召回（决定塔是否有独立价值）。

与 A′ 的区别：A′ 是把多条键塞进同一条记录的读出（闭式解会把读出摊薄）；
这里用 **max-over-keys**：同一记录的多个键各算一个相似度，取最大，不参与读出平均。
值仍然只存一条（记录 id / 答案），所以不存在稀释。

四臂（检索 top-1 记录准确率）：
  T1  塔，每条记录 1 个键（事实原句）
  T3  塔，每条记录 3 个键（事实原句 + 2 句改写）
  L1  词面 bigram，只比事实原句
  L3  词面 bigram，事实原句 + 2 句改写（query expansion 式）

评测查询是**第 4 种改写**，四种臂都没见过。

用法：venv\\Scripts\\python.exe tools_tower_keyexp.py --out _ab_tmp/keyexp
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List

import torch


SURNAME = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
GIVEN = ["伟", "芳", "娜", "敏", "静", "强", "磊", "洋", "艳", "杰", "娟", "涛", "明", "超"]
DEPT = ["研发部", "市场部", "财务部", "人事部", "法务部", "供应链部", "运营部"]


def bigram_cos(a: str, b: str) -> float:
    def grams(t: str) -> Dict[str, int]:
        t = "".join(ch for ch in t if not ch.isspace())
        return {t[i:i + 2]: t.count(t[i:i + 2]) for i in range(len(t) - 1)}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    dot = sum(v * gb.get(k, 0) for k, v in ga.items())
    na = sum(v * v for v in ga.values()) ** 0.5
    nb = sum(v * v for v in gb.values()) ** 0.5
    return dot / (na * nb)


def build(n_ent: int, seed: int = 0) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    names = set()
    while len(names) < n_ent:
        names.add(rng.choice(SURNAME) + rng.choice(GIVEN) + (rng.choice(GIVEN) if rng.random() < 0.3 else ""))
    rows, used = [], set()
    for nm in sorted(names):
        code = None
        while code is None or code in used:
            code = str(rng.randint(10000, 99999))
        used.add(code)
        for attr, ans in (("工号", code), ("部门", rng.choice(DEPT))):
            fact = f"「{nm}」的{attr}是 {ans}。"
            rows.append({
                "fact": fact, "answer": ans,
                "keys": [fact, f"{nm}的{attr}是什么？", f"请问 {nm} 的{attr}？"],
                "test": f"帮我查一下 {nm} 的{attr}",
            })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"_ab_tmp\keyexp")
    ap.add_argument("--ents", default="60,120")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    from tools_tower_memory import TowerMemory

    tm = TowerMemory(r"data\tower_mem\agent.jsonl", verbose=False)
    enc = tm.enc
    out = {}

    for n_ent in [int(x) for x in args.ents.split(",")]:
        rows = build(n_ent)
        tests = [r["test"] for r in rows]
        Kq = torch.stack([torch.nn.functional.normalize(enc.key(t).float(), dim=0) for t in tests])

        # T1 / T3：记录的每个键一个向量；T3 用 3 倍 key 数
        K1 = torch.stack([torch.nn.functional.normalize(enc.key(r["keys"][0]).float(), dim=0) for r in rows])
        flat = [(i, k) for i, r in enumerate(rows) for k in r["keys"]]
        K3 = torch.stack([torch.nn.functional.normalize(enc.key(k).float(), dim=0) for _, k in flat])
        owner = torch.tensor([i for i, _ in flat])

        def max_over(scores: torch.Tensor) -> torch.Tensor:
            """scores: (Q, n_keys) → 每条记录取最大相似度后 argmax。"""
            best = torch.full((scores.shape[0], len(rows)), -2.0)
            best.scatter_reduce_(1, owner.expand(scores.shape[0], -1), scores, reduce="amax")
            return best.argmax(1)

        top_t1 = (Kq @ K1.t()).argmax(1)
        top_t3 = max_over(Kq @ K3.t())

        facts = [r["fact"] for r in rows]
        sims_l1 = torch.tensor([[bigram_cos(q, f) for f in facts] for q in tests])
        top_l1 = sims_l1.argmax(1)
        allkeys = [k for r in rows for k in r["keys"]]
        sims_l3 = torch.tensor([[max(bigram_cos(q, rows[i]["keys"][j]) for j in range(3))
                                 for i in range(len(rows))] for q in tests])
        top_l3 = sims_l3.argmax(1)

        gold = torch.arange(len(rows))
        acc = {name: round(float((t == gold).float().mean()), 4) for name, t in
               (("T1", top_t1), ("T3", top_t3), ("L1", top_l1), ("L3", top_l3))}
        out[str(len(rows))] = {"n_records": len(rows), "acc": acc}
        print(f"[n={len(rows)}] " + json.dumps(acc), flush=True)

    out["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "keyexp.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("[summary_json] " + json.dumps(out, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

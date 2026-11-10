"""边界测试：塔的多键检索在"口语化问法"上还剩多少，以及没有实体名时会怎样。

四类查询（都未出现在写入的键里）：
  Q1 名词化问法      「王芳的工号是什么？」
  Q2 同义属性词      「王芳的员工编号是多少？」/「王芳的所在部门？」
  Q3 口语缩写        「王芳工号多少」/「王芳部门是啥」
  Q4 无实体名        「工号是多少？」——理论上不可解，用来看它退化到哪

用法：venv\\Scripts\\python.exe tools_tower_alias.py --out _ab_tmp/keyexp2
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
SYNONYM = {"工号": ["员工编号", "工牌号"], "部门": ["所在部门", "归属部门"]}


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
                "name": nm, "attr": attr, "fact": fact, "answer": ans,
                # 写入的键：只有"名词化问法"和事实原句，不含下面 Q2/Q3/Q4 的形态
                "keys": [fact, f"{nm}的{attr}是什么？", f"请问 {nm} 的{attr}？"],
                "q1": f"{nm}的{attr}是什么？",
                "q2": f"{nm}的{SYNONYM[attr][0]}是多少？",
                "q3": f"{nm}{attr}多少",
                "q4": f"{attr}是多少？",
            })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"_ab_tmp\keyexp2")
    ap.add_argument("--ents", default="60,120")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    from tools_tower_memory import TowerMemory

    enc = TowerMemory(r"data\tower_mem\agent.jsonl", verbose=False).enc
    out = {}
    for n_ent in [int(x) for x in args.ents.split(",")]:
        rows = build(n_ent)
        flat = [(i, k) for i, r in enumerate(rows) for k in r["keys"]]
        K3 = torch.stack([torch.nn.functional.normalize(enc.key(k).float(), dim=0) for _, k in flat])
        owner = torch.tensor([i for i, _ in flat])
        gold = torch.arange(len(rows))

        acc = {}
        for cond in ("q1", "q2", "q3", "q4"):
            qs = [r[cond] for r in rows]
            Kq = torch.stack([torch.nn.functional.normalize(enc.key(t).float(), dim=0) for t in qs])
            best = torch.full((len(qs), len(rows)), -2.0)
            best.scatter_reduce_(1, owner.expand(len(qs), -1), Kq @ K3.t(), reduce="amax")
            tower = float((best.argmax(1) == gold).float().mean())
            lex = float((torch.tensor([[bigram_cos(q, f) for f in [r["fact"] for r in rows]] for q in qs])
                         .argmax(1) == gold).float().mean())
            acc[cond] = {"tower3": round(tower, 4), "lexical": round(lex, 4)}
        out[str(len(rows))] = acc
        print(f"[n={len(rows)}] " + json.dumps(acc, ensure_ascii=False), flush=True)

    out["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "alias.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("[summary_json] " + json.dumps(out, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

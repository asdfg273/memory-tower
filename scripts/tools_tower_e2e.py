"""E2E：把"写记忆 → 检索 → 读出 → 下游作答"整条链一次跑通，确认可行性边界。

与 H 阶段的分别：H 只测了读取头在固定 120 条事实上的消费能力；这里测**整链**，
并同时问三件事：
  1. 端到端准确率是多少（而不是各段分段指标）；
  2. 换成朴素词面检索（产品里那套 bigram）会差多少 —— 塔到底有没有增量；
  3. 事实规模 120 / 240 / 480 时在哪里崩（m=256 的交付默认值）。

用法：venv\\Scripts\\python.exe tools_tower_e2e.py --out _ab_tmp/e2e
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
GIVEN = ["伟", "芳", "娜", "敏", "静", "强", "磊", "洋", "艳", "杰", "娟", "涛", "明", "超", "秀", "霞"]
DEPT = ["研发部", "市场部", "财务部", "人事部", "法务部", "供应链部", "运营部", "设计部"]


def build(n_ent: int, seed: int = 0) -> List[Dict[str, str]]:
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
            rows.append({"name": nm, "attr": attr, "answer": ans,
                         "fact": f"「{nm}」的{attr}是 {ans}。",
                         "query": f"{nm}的{attr}是什么？",
                         "query2": f"帮我查一下 {nm} 的{attr}"})
    return rows


def bigram_cos(a: str, b: str) -> float:
    def grams(t: str) -> Dict[str, int]:
        t = "".join(ch for ch in t if not ch.isspace())
        g: Dict[str, int] = {}
        for i in range(len(t) - 1):
            g[t[i:i + 2]] = g.get(t[i:i + 2], 0) + 1
        return g
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    dot = sum(v * gb.get(k, 0) for k, v in ga.items())
    na = sum(v * v for v in ga.values()) ** 0.5
    nb = sum(v * v for v in gb.values()) ** 0.5
    return dot / (na * nb)


class Head(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc = torch.nn.Linear(d_in, d_out)
        self.scale = torch.nn.Parameter(torch.tensor(4.0))

    def forward(self, x, cand):
        return self.scale * (torch.nn.functional.normalize(self.fc(x), dim=-1) @ cand.t())


def fit_head(X, Y, cand, steps=300, seed=0) -> Head:
    torch.manual_seed(seed)
    head = Head(X.shape[1], cand.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(steps):
        idx = torch.randint(0, X.shape[0], (32,))
        loss = torch.nn.functional.cross_entropy(head(X[idx], cand), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return head


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"_ab_tmp\e2e")
    ap.add_argument("--scales", default="120,240,480")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    from tools_tower_memory import TowerMemory

    tm = TowerMemory(r"data\tower_mem\agent.jsonl", verbose=False)
    enc = tm.enc
    out = {"scales": {}, "seconds": 0.0}

    for n_facts in [int(x) for x in args.scales.split(",")]:
        rows = build(n_facts // 2)
        facts = [r["fact"] for r in rows]
        qs = [r["query"] for r in rows]
        Kf = torch.stack([torch.nn.functional.normalize(enc.key(t).float(), dim=0) for t in facts])
        Vf = torch.stack([enc.value(t).float() for t in facts])
        # 结构化写入：值只放答案本身（而不是整句话）——检验"值被稀释"是不是失败主因
        Va = torch.stack([enc.value(r["answer"]).float() for r in rows])
        Kq = torch.stack([torch.nn.functional.normalize(enc.key(t).float(), dim=0) for t in qs])
        W = torch.linalg.solve(Kf.t() @ Kf + 1e-2 * torch.eye(Kf.shape[1]), Kf.t() @ Vf)
        Wa = torch.linalg.solve(Kf.t() @ Kf + 1e-2 * torch.eye(Kf.shape[1]), Kf.t() @ Va)

        mem_soft = torch.nn.functional.normalize(Kq @ W, dim=-1)
        mem_soft = torch.nn.functional.normalize(mem_soft - mem_soft.mean(0, keepdim=True), dim=-1)
        top = (Kq @ Kf.t()).argmax(1)
        mem_hard = torch.nn.functional.normalize(Vf[top], dim=-1)
        mem_ans = torch.nn.functional.normalize(Kq @ Wa, dim=-1)          # 值=答案的结构化读出
        mem_ans = torch.nn.functional.normalize(mem_ans - mem_ans.mean(0, keepdim=True), dim=-1)
        mem_ans_hard = torch.nn.functional.normalize(Va[top], dim=-1)

        # 词面基线：用 bigram 相似度做同样的软读出（模拟产品里的检索层）
        sims = torch.tensor([[bigram_cos(q, f) for f in facts] for q in qs])
        w = sims / sims.sum(1, keepdim=True).clamp_min(1e-6)
        mem_lex = torch.nn.functional.normalize(w @ Vf, dim=-1)
        lex_top = sims.argmax(1)
        mem_lexans = torch.nn.functional.normalize(Va[lex_top], dim=-1)   # 词面检索 top-1 + 结构化值

        # 塔多键检索：每条记录 3 个键（事实原句 + 2 种问法），max-over-keys 取记录
        flat = [(i, k) for i, r in enumerate(rows)
                for k in (r["fact"], r["query"], f"请问 {r['name']} 的{r['attr']}？")]
        K3 = torch.stack([torch.nn.functional.normalize(enc.key(k).float(), dim=0) for _, k in flat])
        owner = torch.tensor([i for i, _ in flat])
        best = torch.full((Kq.shape[0], len(rows)), -2.0)
        best.scatter_reduce_(1, owner.expand(Kq.shape[0], -1), Kq @ K3.t(), reduce="amax")
        top3 = best.argmax(1)
        mem_t3ans = torch.nn.functional.normalize(Va[top3], dim=-1)

        answers = sorted({r["answer"] for r in rows})
        cand = torch.stack([torch.nn.functional.normalize(enc.key(a).float(), dim=0) for a in answers])
        amap = {a: i for i, a in enumerate(answers)}
        Y = torch.tensor([amap[r["answer"]] for r in rows])

        names = sorted({r["name"] for r in rows})
        n_tr = len(names) * 2 // 3
        tr_mask = torch.tensor([r["name"] in set(names[:n_tr]) for r in rows])
        tr = tr_mask.nonzero().flatten()
        te = (~tr_mask).nonzero().flatten()

        arms = {
            "none": Kq,
            "tower_soft_c": torch.cat([Kq, mem_soft], 1),
            "tower_hard": torch.cat([Kq, mem_hard], 1),
            "lex_soft": torch.cat([Kq, mem_lex], 1),
            "ans_soft": torch.cat([Kq, mem_ans], 1),
            "ans_hard": torch.cat([Kq, mem_ans_hard], 1),
            "lex_ans": torch.cat([Kq, mem_lexans], 1),
            "tower3_ans": torch.cat([Kq, mem_t3ans], 1),
            "oracle": torch.cat([Kq, torch.nn.functional.normalize(Vf, dim=-1)], 1),
        }
        acc = {}
        for name, X in arms.items():
            head = fit_head(X[tr], Y[tr], cand, steps=args.steps)
            with torch.no_grad():
                acc[name] = round(float((head(X[te], cand).argmax(1) == Y[te]).float().mean()), 4)
        retr = {
            "tower_top1": round(float((top == torch.arange(len(top)))[te].float().mean()), 4),
            "lex_top1": round(float((lex_top == torch.arange(len(lex_top)))[te].float().mean()), 4),
        }
        out["scales"][str(n_facts)] = {"n_queries": int(len(te)), "acc": acc, "retrieval": retr}
        print(f"[n={n_facts}] acc={json.dumps(acc)} retr={json.dumps(retr)}", flush=True)

    out["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "e2e.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("[summary_json] " + json.dumps(out, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""H 阶段（纯文本）：验证"下游能不能学会消费塔的读出"。

假设链：V 阶段证明塔的值可复现、可判别；注入没增益的旧解释是"向量不行"，新解释是"下游没被训练"。
本脚本直接测后者：**冻结编码器 + 冻结塔**，只训一个线性读取头。

任务：60 个实体 × 2 个属性（工号/部门），每条事实写成一句话存进塔。
查询写成问句（不是原文），用塔读出 r = q·W，再接线性头预测答案（候选集里选一个）。

四臂（同一训练量、同一候选集）：
  q        —— 只有查询向量（无记忆，应当学不会）
  qr       —— 查询 + 塔读出
  qrc      —— 查询 + 中心化的塔读出
  oracle   —— 查询 + 真实值向量（上界）

预注册判据：
  H1 qr 在留出实体上比 q 高 ≥0.10 → 读出可被下游消费。
  H2 qr 与 oracle 差 ≤0.10 → 读出基本等价于真实值。
  H3 qrc 与 qr 差 ≥ −0.02 → 中心化无害（有助于说明"该给下游什么形状的向量"）。

用法：venv\\Scripts\\python.exe tools_tower_head_probe.py --out _ab_tmp/headprobe
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List, Tuple

import torch


SURNAME = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
GIVEN = ["伟", "芳", "娜", "敏", "静", "强", "磊", "洋", "艳", "杰", "娟", "涛", "明", "超", "秀", "霞"]
DEPT = ["研发部", "市场部", "财务部", "人事部", "法务部", "供应链部"]
ATTRS = ["工号", "部门"]


def build_data(n_ent: int, seed: int = 0) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    names = set()
    while len(names) < n_ent:
        names.add(rng.choice(SURNAME) + rng.choice(GIVEN) + (rng.choice(GIVEN) if rng.random() < 0.3 else ""))
    rows = []
    used = set()
    for nm in sorted(names):
        code = None
        while code is None or code in used:
            code = str(rng.randint(1000, 9999))
        used.add(code)
        dept = rng.choice(DEPT)
        for attr, ans in (("工号", code), ("部门", dept)):
            fact = f"「{nm}」的{attr}是 {ans}。"
            query = f"{nm}的{attr}是什么？"
            rows.append({"name": nm, "attr": attr, "answer": ans, "fact": fact, "query": query})
    return rows


def tower_fit(K: torch.Tensor, Z: torch.Tensor, lam: float = 1e-2) -> torch.Tensor:
    return torch.linalg.solve(K.t() @ K + lam * torch.eye(K.shape[1]), K.t() @ Z)


class Head(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc = torch.nn.Linear(d_in, d_out)
        self.scale = torch.nn.Parameter(torch.tensor(4.0))

    def forward(self, x: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.normalize(self.fc(x), dim=-1)
        return self.scale * (h @ cand.t())


def run_arm(arm: str, Kq, Kr, Rr, Y, cand, tr, te, steps=400, seed=0) -> Dict[str, float]:
    torch.manual_seed(seed)
    d_in = Kq.shape[1] + (Rr.shape[1] if arm != "q" else 0)
    head = Head(d_in, cand.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    n = len(tr)
    best = 0.0
    for step in range(steps):
        idx = tr[torch.randint(0, n, (32,))]
        x = Kq[idx] if arm == "q" else torch.cat([Kq[idx], Rr[idx]], dim=1)
        logits = head(x, cand)
        loss = torch.nn.functional.cross_entropy(logits, Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 49:
            with torch.no_grad():
                xt = Kq[te] if arm == "q" else torch.cat([Kq[te], Rr[te]], dim=1)
                acc = float((head(xt, cand).argmax(1) == Y[te]).float().mean())
                best = max(best, acc)
    return {"heldout_top1": round(best, 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"_ab_tmp\headprobe")
    ap.add_argument("--ents", type=int, default=60)
    ap.add_argument("--train-frac", type=float, default=2 / 3)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    from tools_tower_memory import TowerMemory

    tm = TowerMemory(r"data\tower_mem\agent.jsonl", verbose=False)
    rows = build_data(args.ents)
    facts = [r["fact"] for r in rows]
    queries = [r["query"] for r in rows]
    print(f"[data] {len(rows)} 条事实 / {len(set(r['name'] for r in rows))} 个实体", flush=True)

    Kf = torch.stack([torch.nn.functional.normalize(tm.enc.key(t).float(), dim=0) for t in facts])
    Vf = torch.stack([tm.enc.value(t).float() for t in facts])
    Kq = torch.stack([torch.nn.functional.normalize(tm.enc.key(t).float(), dim=0) for t in queries])
    W = tower_fit(Kf, Vf)

    def read(q: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(q @ W, dim=-1)

    R = read(Kq)
    Rcent = R - R.mean(0, keepdim=True)
    Rcent = torch.nn.functional.normalize(Rcent, dim=-1)
    Roracle = torch.nn.functional.normalize(Vf, dim=-1)
    # hard：先用塔当索引取 top-1 条目，再拿那条的**原始值**（检索+读取，而不是软投影）
    Kf_n = torch.nn.functional.normalize(Kf, dim=-1)
    top = (Kq @ Kf_n.t()).argmax(1)
    Rhard = torch.nn.functional.normalize(Vf[top], dim=-1)
    retr_acc = float((top == torch.arange(len(top))).float().mean())
    print(f"[retrieval] 查询→事实 top-1 = {retr_acc:.4f}（{len(top)} 条）", flush=True)

    # 候选答案 = 全部不同答案文本的键
    answers = sorted({r["answer"] for r in rows})
    cand = torch.stack([torch.nn.functional.normalize(tm.enc.key(a).float(), dim=0) for a in answers])
    amap = {a: i for i, a in enumerate(answers)}
    Y = torch.tensor([amap[r["answer"]] for r in rows])

    names = sorted({r["name"] for r in rows})
    n_tr = int(len(names) * args.train_frac)
    tr_names, te_names = set(names[:n_tr]), set(names[n_tr:])
    tr = torch.tensor([i for i, r in enumerate(rows) if r["name"] in tr_names])
    te = torch.tensor([i for i, r in enumerate(rows) if r["name"] in te_names])
    print(f"[split] train {len(tr)} / heldout {len(te)} queries, candidates {len(answers)}", flush=True)

    arms = {
        "q": Kq,
        "qr": None,
        "qrc": None,
        "oracle": None,
    }
    reads = {"q": Kq, "qr": R, "qrc": Rcent, "hard": Rhard, "oracle": Roracle}
    out = {}
    for arm, rr in reads.items():
        out[arm] = run_arm(arm, Kq, None, rr, Y, cand, tr, te, steps=args.steps)
        print(f"[{arm}] " + json.dumps(out[arm], ensure_ascii=False), flush=True)

    res = {"n_facts": len(rows), "n_entities": len(names), "candidates": len(answers),
           "arms": out, "seconds": round(time.time() - t0, 1)}
    path = os.path.join(args.out, "headprobe.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("[summary_json] " + json.dumps(res, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

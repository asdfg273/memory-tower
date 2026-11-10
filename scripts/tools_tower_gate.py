"""G 阶段：把《新建 文本文档.txt》里"让记忆互相看见 / 用 MLP 选接口 / 两跳组合"做成可证伪实验。

关键设计约束（第一版全 0 的原因，已修）：
  闭式岭回归 K→Z 的输出只能落在 span(K) 里。人工造的"随机值向量"在 80 维里几乎必然
  落在 span(键) 外（键只有 6 维），于是读出上限 cos≈0.75，实验还没开始就塌了。
  真实的塔里键和值都由同一个编码器产出、共享低维流形，所以能用。
  → 这里让**值就是键**（密码值 = 该事实自己的键；朋友值 = 朋友的密码键），
    保证可精确插值，才能干净地问"门控/选择器/两跳"到底带来什么。

预注册判据：
  G1 门控：softmax 混合在单跳上允许 −0.02；若单跳掉 >0.02 且无收益 → 门控不划算。
  G2 MLP 选择器：留出查询上比 max-cos 高 ≥0.05 才算"值得训"。
  G3 两跳：独立槽若已 ≥0.95 → "让记忆互相看见"没有存在理由。

用法：venv\\Scripts\\python.exe tools_tower_gate.py --out _ab_tmp\\gate
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import torch

D = 80                     # 键/值维度
D_ENT = 64                 # 实体子空间（正交化，避免键空间病态）
LAM = 1e-3


def make_facts(n_ent: int, seed: int = 0) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(D_ENT, D_ENT, generator=g))   # 正交键
    ent = q[:n_ent].contiguous()
    friend = torch.randperm(n_ent, generator=g)
    return {"ent": ent, "friend": friend}


def make_key(ent_vec: torch.Tensor, attr: int) -> torch.Tensor:
    k = torch.zeros(D)
    k[:D_ENT] = ent_vec
    k[D_ENT + attr] = 1.0
    return k


def build_memory(facts: Dict[str, torch.Tensor], n_ent: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """两条事实/实体：key(i,pw) → key(i,pw)；key(i,friend) → key(friend(i),pw)。"""
    ents, fr = facts["ent"], facts["friend"]
    keys, vals, owner = [], [], []
    for i in range(n_ent):
        kp = make_key(ents[i], 0)
        keys.append(kp); vals.append(kp); owner.append(i)             # 密码：自映射
        kf = make_key(ents[i], 1)
        keys.append(kf); vals.append(make_key(ents[int(fr[i])], 0)); owner.append(i)
    return torch.stack(keys), torch.stack(vals), owner


def fit(K: torch.Tensor, Z: torch.Tensor, lam: float = LAM) -> torch.Tensor:
    return torch.linalg.solve(K.t() @ K + lam * torch.eye(D), K.t() @ Z)


def fit_slots(K: torch.Tensor, Z: torch.Tensor, owner: List[int], n_slots: int) -> List[torch.Tensor]:
    own = torch.tensor(owner)
    out = []
    for s in range(n_slots):
        mask = (own % n_slots) == s
        out.append(fit(K[mask], Z[mask]) if mask.any() else torch.zeros(D, D))
    return out


def slot_scores(q: torch.Tensor, K: torch.Tensor, owner: List[int], n_slots: int) -> torch.Tensor:
    own = torch.tensor(owner)
    s = torch.full((n_slots,), -1e9)
    for i in range(n_slots):
        mask = (own % n_slots) == i
        if mask.any():
            s[i] = (K[mask] @ q).max()
    return s


def read(arm: str, q: torch.Tensor, ws: List[torch.Tensor], K: torch.Tensor, owner: List[int],
         n_slots: int, gate=None, tau: float = 0.05) -> torch.Tensor:
    scores = slot_scores(q, K, owner, n_slots)
    reads = torch.stack([ws[i].t() @ q for i in range(n_slots)])
    if arm == "indep":
        return reads[int(scores.argmax())]
    if arm == "softmax":
        w = torch.softmax(scores / tau, dim=0)
        return (w[:, None] * reads).sum(0)
    if arm == "mlp":
        w = torch.softmax(gate(q), dim=0)
        return (w[:, None] * reads).sum(0)
    raise ValueError(arm)


def hit(preds: torch.Tensor, targets: torch.Tensor, thr: float = 0.9) -> float:
    cos = torch.nn.functional.cosine_similarity(preds, targets, dim=1)
    return round(float((cos > thr).float().mean()), 4)


def run(n_ent: int, n_slots: int, seed: int = 0) -> Dict[str, object]:
    torch.manual_seed(seed)
    facts = make_facts(n_ent, seed)
    K, Z, owner = build_memory(facts, n_ent)
    ws = fit_slots(K, Z, owner, n_slots)

    n_test = min(8, n_ent)
    test = list(range(n_ent - n_test, n_ent))
    train = list(range(n_ent - n_test))

    def q(i: int, attr: int) -> torch.Tensor:
        return make_key(facts["ent"][i], attr)

    tgt_pw = torch.stack([make_key(facts["ent"][i], 0) for i in test])
    tgt_fr = torch.stack([make_key(facts["ent"][int(facts["friend"][i])], 0) for i in test])

    rows: Dict[str, Dict[str, float]] = {}
    for arm in ("indep", "softmax"):
        pw = hit(torch.stack([read(arm, q(i, 0), ws, K, owner, n_slots) for i in test]), tgt_pw)
        fr = hit(torch.stack([read(arm, q(i, 1), ws, K, owner, n_slots) for i in test]), tgt_fr)
        two = []
        for i in test:                      # 两跳：朋友键读出的正是朋友的"密码键"，直接比
            r1 = read(arm, q(i, 1), ws, K, owner, n_slots)
            q2 = torch.zeros(D); q2[:D_ENT] = r1[:D_ENT]; q2[D_ENT] = 1.0
            two.append(read(arm, q2, ws, K, owner, n_slots))
        hop2 = hit(torch.stack(two), tgt_fr)
        rows[arm] = {"pw": pw, "friend": fr, "2hop": hop2}

    if train:
        gate = torch.nn.Sequential(torch.nn.Linear(D, 64), torch.nn.GELU(), torch.nn.Linear(64, n_slots))
        opt = torch.optim.Adam(gate.parameters(), lr=1e-2)
        ce = torch.nn.CrossEntropyLoss()
        pool = [(q(i, a), i % n_slots) for i in train for a in (0, 1)]
        for _ in range(80):
            for j in torch.randperm(len(pool)):
                qq, tgt = pool[int(j)]
                opt.zero_grad(); ce(gate(qq).unsqueeze(0), torch.tensor([tgt])).backward(); opt.step()
        with torch.no_grad():
            rows["mlp"] = {"pw": hit(torch.stack([read("mlp", q(i, 0), ws, K, owner, n_slots, gate) for i in test]), tgt_pw)}

    n_in_slot = int(((torch.tensor(owner) % n_slots) == 0).sum())

    # ---- 全局单塔（不分槽）：塔本来就是一个全连接混合器，没有"互相看不见"的问题
    Wg = fit(K, Z)
    pw_g = hit(torch.stack([Wg.t() @ q(i, 0) for i in test]), tgt_pw)
    fr_g = hit(torch.stack([Wg.t() @ q(i, 1) for i in test]), tgt_fr)
    two_g = []
    for i in test:
        r1 = Wg.t() @ q(i, 1)
        q2 = torch.zeros(D); q2[:D_ENT] = r1[:D_ENT]; q2[D_ENT] = 1.0
        two_g.append(Wg.t() @ q2)
    hop2_g = hit(torch.stack(two_g), tgt_fr)

    # ---- 消融：值不在 span(K) 里时（第一版全 0 的根因）重建余弦的天花板
    zr = torch.nn.functional.normalize(torch.randn(*Z.shape, generator=torch.Generator().manual_seed(1)), dim=-1)
    Wr = fit(K, zr)
    ceil_cos = float(torch.nn.functional.cosine_similarity(K @ Wr, zr, dim=1).mean())

    res = {"n_ent": n_ent, "n_slots": n_slots, "facts_per_slot": n_in_slot,
           "rank_K": int(torch.linalg.matrix_rank(K)),
           "train_entities": len(train),
           "global": {"pw": pw_g, "friend": fr_g, "2hop": hop2_g},
           "values_outside_span_ceiling_cos": round(ceil_cos, 3),
           "rows": rows}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"_ab_tmp\gate")
    ap.add_argument("--ns", default="16,48")
    ap.add_argument("--slots", default="2,8")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    results = [run(n, s) for n in [int(x) for x in args.ns.split(",")]
               for s in [int(x) for x in args.slots.split(",")]]
    out = {"config": {"d": D, "d_ent": D_ENT, "lam": LAM}, "results": results,
           "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out, "gate.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("[summary_json] " + json.dumps(out, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

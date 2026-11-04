r"""tools_tower_slotdepth_train.py —— T132：训练投影版槽位塔（L1/L2/L4 × S）×（verbatim / 改写）

与 T131 的区别
--------------
T131 用的是"截断代理"：把 256 维已训练投影截短，模拟小 m。
本脚本把投影**真的训练出来**：P（键）与 Qv（值）都是 2048→d 的线性层，用双层目标训练——
外层是"读出后能认出正确答案"（交叉熵），内层是闭式岭回归 α=(KKᵀ+λI)⁻¹Z（可微）。

训练/测试分离
-------------
实体池 496 个单字、名称池 549 个单 token 字，切成互不相交的两份：
    训练 400 条（实体池前 400，名称池前 400）
    测试  96 条（实体池后 96，名称池后 149 → 取前 96）
测试集上的投影**从未见过**，只是拿来写塔、读出、判命中。

两个查询口径
------------
    verbatim  箱子{q}的密码是什么？只回答密码本身。        （与写入键同文）
    para      我想知道{q}那个箱子里存的密码，直接告诉我密码就行。（改写过，实体仍在）

臂
--
    (L,S) ∈ {(1,0), (1,8), (2,8), (4,8)}，d ∈ {48, 96}
    S=0 是交付默认形态（单矩阵）。

跑法
----
    $env:HF_HUB_OFFLINE="1"
    venv\Scripts\python.exe tools_tower_slotdepth_train.py --dims 48,96 --steps 400
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from tower_runtime import _E_POOL, _E_POOL_EXT2, _N_POOL, _N_POOL_EXT, _N_POOL_EXT2

Q_VERB = "箱子{q}的密码是什么？只回答密码本身。"
Q_PARA2 = "{q}的箱子密码是什么？只回答密码本身。"          # 轻度改写：语序略动
Q_PARA = "我想知道{q}那个箱子里存的密码，直接告诉我密码就行。"  # 重度改写：整句换掉


def pools(n_train: int, n_test: int) -> Tuple[List[str], List[str], List[str], List[str]]:
    """返回 (训练实体, 训练名称, 测试实体, 测试名称)，两两不相交。"""
    ents = list(dict.fromkeys(list(_E_POOL) + list(_E_POOL_EXT2)))
    names = list(dict.fromkeys(list(_N_POOL) + list(_N_POOL_EXT2) + list(_N_POOL_EXT)))
    names = [x for x in names if x not in set(ents)]
    tr_e, te_e = ents[:n_train], ents[n_train:n_train + n_test]
    tr_n, te_n = names[:n_train], names[n_train:n_train + n_test]
    assert len(te_e) == n_test and len(te_n) == n_test, (len(te_e), len(te_n))
    assert not (set(tr_e) & set(te_e)) and not (set(tr_n) & set(te_n))
    return tr_e, tr_n, te_e, te_n


def encode(texts: List[str], kind: str, device: str, tag: str) -> torch.Tensor:
    """kind: 'key' | 'value'；缓存到 _ab_tmp。"""
    path = f"_ab_tmp/slotdepth_train_{kind}_{tag}.pt"
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    from tools_tower_memory import Encoder

    enc = Encoder(proj=False, device=device, verbose=False)
    t0 = time.time()
    out = []
    for i, t in enumerate(texts):
        out.append(enc.key(t) if kind == "key" else enc.value(t))
        if (i + 1) % 200 == 0:
            print(f"[enc:{kind}] {i + 1}/{len(texts)} {time.time() - t0:.1f}s", flush=True)
    M = torch.stack(out).float().cpu()
    os.makedirs("_ab_tmp", exist_ok=True)
    torch.save(M, path)
    print(f"[enc:{kind}] {len(texts)} 条 {time.time() - t0:.1f}s dim={M.shape[1]} -> {path}", flush=True)
    return M


# ----------------------------------------------------------------------------
# 塔（可微）
# ----------------------------------------------------------------------------
def build_keys(contents: torch.Tensor, L: int, S: int) -> Tuple[torch.Tensor, List[List[int]], List[int]]:
    """把内容键拼上槽位 one-hot。返回 (K, 分组行号, 每条事实的槽号)。"""
    n = contents.shape[0]
    if S <= 0:
        return F.normalize(contents, dim=-1), [list(range(n))], [0] * n
    slots = [i % S for i in range(n)]
    layers = [(i // S) % L for i in range(n)]
    K = torch.zeros(n, S + contents.shape[1], device=contents.device)
    for i in range(n):
        K[i, slots[i]] = 1.0
        K[i, S:] = contents[i]
    K = F.normalize(K, dim=-1)
    groups = [[i for i in range(n) if layers[i] == l and slots[i] == s]
              for l in range(L) for s in range(S)]
    groups = [g for g in groups if g]
    return K, groups, slots


def slot_query(contents: torch.Tensor, slots: List[int], S: int) -> torch.Tensor:
    if S <= 0:
        return F.normalize(contents, dim=-1)
    Q = torch.zeros(contents.shape[0], S + contents.shape[1], device=contents.device)
    for i, s in enumerate(slots):
        Q[i, s] = 1.0
        Q[i, S:] = contents[i]
    return F.normalize(Q, dim=-1)


def readout(Kq: torch.Tensor, K: torch.Tensor, Z: torch.Tensor, groups: List[List[int]],
            lam: float) -> torch.Tensor:
    """返回 (nq, n) 余弦矩阵：对所有组取最大。"""
    Zl = F.normalize(Z, dim=-1)
    n = K.shape[0]
    sim = torch.full((Kq.shape[0], n), -2.0, device=Kq.device)
    for rows in groups:
        kk = K[rows]
        al = torch.linalg.solve(kk @ kk.t() + lam * torch.eye(len(rows), device=K.device), Z[rows])
        r = (Kq @ kk.t()) @ al
        sim = torch.maximum(sim, F.normalize(r, dim=-1) @ Zl.t())
    return sim


def hit_at_1(sim: torch.Tensor) -> float:
    nq = sim.shape[0]
    return float((sim.argmax(dim=1) == torch.arange(nq, device=sim.device)).sum()) / nq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="48,96")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--align-weight", type=float, default=0.0,
                    help=">0 时额外要求 P(改写键) 与 P(原文键) 对齐（不增加存储键）")
    ap.add_argument("--disc", action="store_true",
                    help="判别式联合训练：原文键与改写键一起进交叉熵（同一答案两个入口）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="_ab_tmp/tower_slotdepth_train.json")
    args = ap.parse_args()

    dev = torch.device(args.device)
    tr_e, tr_n, te_e, te_n = pools(args.n_train, args.n_test)
    # 训练集：verbatim 键；测试集：verbatim 键 + 改写键 + 值
    Kv_tr = encode([Q_VERB.format(q=e) for e in tr_e], "key", args.device, "train_verb").to(dev)
    V_tr = encode(tr_n, "value", args.device, "train_val").to(dev)
    Kv_te = encode([Q_VERB.format(q=e) for e in te_e], "key", args.device, "test_verb").to(dev)
    Kp2_te = encode([Q_PARA2.format(q=e) for e in te_e], "key", args.device, "test_para2").to(dev)
    Kp_te = encode([Q_PARA.format(q=e) for e in te_e], "key", args.device, "test_para").to(dev)
    V_te = encode(te_n, "value", args.device, "test_val").to(dev)
    if args.align_weight > 0 or args.disc:
        Kp_tr = encode([Q_PARA.format(q=e) for e in tr_e], "key", args.device, "train_para").to(dev)
    else:
        Kp_tr = None
    print(f"[data] train={len(tr_e)} test={len(te_e)} dev={args.device}", flush=True)

    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    arms = [(1, 0), (1, 8), (2, 8), (4, 8)]
    result: Dict[str, Dict[str, Dict[str, float]]] = {}

    for d in dims:
        for L, S in arms:
            name = f"L{L}_S{S}"
            torch.manual_seed(0)
            P = torch.nn.Linear(Kv_tr.shape[1], d).to(dev)      # 键投影
            Qv = torch.nn.Linear(V_tr.shape[1], d).to(dev)      # 值投影
            opt = torch.optim.Adam(list(P.parameters()) + list(Qv.parameters()), lr=args.lr)
            t0 = time.time()
            for step in range(args.steps):
                opt.zero_grad()
                Kc, groups, slots = build_keys(P(Kv_tr), L, S)
                Z = F.normalize(Qv(V_tr), dim=-1)
                Kq = slot_query(P(Kv_tr), slots, S)
                n_tr = Kq.shape[0]
                if args.disc and Kp_tr is not None:
                    Kq = torch.cat([Kq, slot_query(P(Kp_tr), slots, S)], dim=0)
                sim = readout(Kq, Kc, Z, groups, args.lam) / 0.05
                labels = torch.arange(n_tr, device=dev).repeat(sim.shape[0] // n_tr)
                loss = F.cross_entropy(sim, labels)
                if Kp_tr is not None:
                    pv = F.normalize(P(Kv_tr), dim=-1)
                    pp = F.normalize(P(Kp_tr), dim=-1)
                    loss = loss + args.align_weight * (1.0 - (pv * pp).sum(dim=-1)).mean()
                loss.backward()
                opt.step()
                if (step + 1) % 100 == 0:
                    print(f"[train] d={d} {name} step={step + 1} loss={loss.item():.4f}"
                          f"  {time.time() - t0:.1f}s", flush=True)

            with torch.no_grad():
                Kc, groups, slots = build_keys(P(Kv_te), L, S)
                Z = F.normalize(Qv(V_te), dim=-1)
                sim_v = readout(slot_query(P(Kv_te), slots, S), Kc, Z, groups, args.lam)
                sim_l = readout(slot_query(P(Kp2_te), slots, S), Kc, Z, groups, args.lam)
                sim_h = readout(slot_query(P(Kp_te), slots, S), Kc, Z, groups, args.lam)
                hit_v, hit_p2, hit_p = hit_at_1(sim_v), hit_at_1(sim_l), hit_at_1(sim_h)
                hit_x = hit_at_1((sim_v + sim_l + sim_h) / 3.0)   # 查询侧扩展（多模板打分融合）
            result.setdefault(str(d), {})[name] = {
                "verbatim": round(hit_v, 4),
                "para_light": round(hit_p2, 4),
                "para": round(hit_p, 4),
                "expand": round(hit_x, 4),
                "train_loss": round(float(loss.item()), 4),
            }
            print(f"[eval] d={d} {name} verbatim={hit_v:.3f} para_light={hit_p2:.3f} "
                  f"para={hit_p:.3f} loss={loss.item():.3f}", flush=True)

    payload = {
        "note": "T132 训练投影版槽位塔；训练/测试实体与名称互不相交",
        "protocol": {"n_train": len(tr_e), "n_test": len(te_e), "lam": args.lam,
                     "steps": args.steps, "lr": args.lr,
                     "query_verbatim": Q_VERB, "query_para": Q_PARA},
        "results": result,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

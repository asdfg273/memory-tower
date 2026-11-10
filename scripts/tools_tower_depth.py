r"""
tools_tower_depth.py —— T130：深度塔（层叠快速权重）—— **深度买到的是容量，还是泛化？**
================================================================================

背景（PAPER_TITANS_TOWER.md §6.2）
----------------------------------
单层塔的失效点已经被钉死：注入是无条件输出指令、多事实叠加缺"可分性维度"、
连接器是逐事实记忆不泛化。但**塔自身层叠成深度网络**这件事没测过：
`tower_runtime.py` 从头到尾只有一对 `(W, Z)`，40 槽是同一状态里的地址，不是层。

假设（先写死，免得事后解释）
----------------------------
H1 线性深度是**平凡**的：L 层纯线性读-写串联 ≡ 一个线性映射，容量断崖只由 m 决定
   → `lin_L2`/`lin_L4` 与 `lin_L1` 的曲线逐点相同（允许数值误差），`lin_L1_m512` 才把断崖右移。
H2 非线性递推（每层读出后归一化 + 残差）**不能**突破 m：它只改误差形状，不改秩。
H3 深度唯一能买到的容量是**地址数**：L 层 × S 槽址的容量 ≈ L·S 个地址，
   → `slot_L4_S40`（160 地址）应接近 `slot_L1_S160`（160 地址），而不是接近 `slot_L1_S40`。

测量协议（纯线性代数，不涉及语言模型）
------------------------------------
- 事实 i：key `k_i ∈ R^m`（高斯，归一化），value `z_i ∈ R^dv`（高斯，归一化）。
- 写入：闭式岭回归 `W = (KᵀK + λI)⁻¹ Kᵀ Z`（与运行时 `write_mode="closed"` 同式，λ=1e-2）。
- 读出：`r = q W`，`pred = argmax_i cos(r, z_i)`，命中即 top-1。
- 查询：训练用的同一批 key（考的是**存储与分辨**，不是语义编码）。
- 槽位臂：key 前 S 维放槽 one-hot，其余放内容；事实 i 分到槽 `i % S`。

臂
--
    lin_L1        1 层，m=256（基线）
    lin_L2        2 层线性串联（H1）
    lin_L4        4 层线性串联（H1）
    lin_L1_m512   1 层，m=512（**参数匹配的宽度对照**：和 2 层 256 同参数量）
    norm_L4       4 层，每层 normalize + 残差（H2）
    slot_L1_S40   1 层，40 地址
    slot_L1_S160  1 层，160 地址（**地址数对照**）
    slot_L4_S40   4 层 × 40 地址 = 160 地址（H3：深度 = 地址空间？）

用法
    python tools_tower_depth.py                 # 全套，CPU，约 1 分钟
    python tools_tower_depth.py --ns 500 --k 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

LAM = 1e-2          # 与 tools_tower_memory.py 的 Encoder(lam=1e-2) 对齐
DV = None           # 值空间维：运行时取 m（读写同维，线性串联才可定义）
SEEDS = (0, 1, 2)


# ---------------------------------------------------------------- 基础设施

def unit(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-8)


def make_facts(n: int, m: int, seed: int):
    """key 与 value 都在 m 维（读写同维），都归一化。"""
    g = torch.Generator().manual_seed(seed)
    K = unit(torch.randn(n, m, generator=g))
    Z = unit(torch.randn(n, m, generator=g))
    return K, Z


def ridge_write(K: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """闭式写入：W = (KᵀK + λI)⁻¹ Kᵀ Z，W ∈ R^{m×DV}。"""
    m = K.shape[1]
    A = K.t() @ K + LAM * torch.eye(m)
    return torch.linalg.solve(A, K.t() @ Z)


def top1(r: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """r [B,DV] 对 Z [N,DV] 做余弦，返回每个 query 的 top-1 下标。"""
    return (unit(r) @ unit(Z).t()).argmax(dim=-1)


def score(r: torch.Tensor, Z: torch.Tensor, tgt: torch.Tensor) -> float:
    return float((top1(r, Z) == tgt).float().mean())


# ---------------------------------------------------------------- 各臂

def arm_lin(K, Z, layers: int):
    """L 层线性串联：q_{l+1} = q_l + q_l W_l（每层都写入全部事实）。"""
    Ws = [ridge_write(K, Z) for _ in range(layers)]
    def read(Q):
        q = Q
        for W in Ws:
            q = q + q @ W
        return q
    return read


def arm_norm(K, Z, layers: int):
    """L 层非线性递推：r_l = normalize(q_l W_l)，q_{l+1} = q_l + r_l。"""
    Ws = [ridge_write(K, Z) for _ in range(layers)]
    def read(Q):
        q = Q
        for W in Ws:
            q = q + unit(q @ W)
        return q
    return read


def arm_lin_width(K, Z, m2: int):
    """宽度对照：1 层但 m=512。key 用前 m2 维补齐（内容与 m=256 臂同一批 key 的扩展）。"""
    g = torch.Generator().manual_seed(999)
    pad = unit(torch.randn(K.shape[0], m2 - K.shape[1], generator=g))
    K2 = unit(torch.cat([K, pad], dim=1))
    W = ridge_write(K2, Z)
    def read(Q):
        return Q @ W
    return read, K2


def slot_key(content: torch.Tensor, slots: int, m: int) -> torch.Tensor:
    """把内容压到 m-slots 维，前面拼 slot one-hot。"""
    return torch.cat([torch.zeros(content.shape[0], slots), content], dim=1)


def arm_slot(K, Z, slots: int, layers: int = 1):
    """地址化：事实 i 写进第 i%slots 个槽；多层时按 (i//slots) 分层。

    查询只带内容 + 槽号（槽号由调用方给出：考的是"同一地址内的分辨与总地址数"，
    不是路由学习）。返回 read(Q) -> [(r_l, Z_l, idx_l), ...]，逐层各自打分后取最大。
    """
    n, m = K.shape
    if slots > m:
        raise SystemExit("slots 不能大于 m")
    content_dim = m - slots
    pad = unit(K[:, :content_dim])
    stores = []
    for L in range(layers):
        idx = torch.tensor([i for i in range(n) if (i // slots) % layers == L])
        if idx.numel() == 0:
            continue
        kk = torch.cat([torch.nn.functional.one_hot(
            torch.tensor([int(i) % slots for i in idx]), slots).float(),
            pad[idx]], dim=1)
        W = ridge_write(unit(kk), Z[idx])
        stores.append((W, Z[idx], idx))

    def read(Q):
        qpad = pad[: Q.shape[0]]
        qslot = torch.nn.functional.one_hot(
            torch.arange(Q.shape[0]) % slots, slots).float()
        kq = unit(torch.cat([qslot, qpad], dim=1))
        return [(kq @ W, Zl, idx) for W, Zl, idx in stores]

    return read, stores


# ---------------------------------------------------------------- 实验

def run_capacity(ns, seeds, m: int, arms):
    """主曲线：每 N 下各臂 top-1。"""
    rows = []
    for n in ns:
        acc = {a: [] for a in arms}
        for seed in seeds:
            K, Z = make_facts(n, m, seed)
            nq = min(n, 256)
            Q = K[:nq]
            tgt = torch.arange(nq) % n
            for a in arms:
                hit = None
                if a == "lin_L1":
                    r = arm_lin(K, Z, 1)(Q)
                elif a == "lin_L2":
                    r = arm_lin(K, Z, 2)(Q)
                elif a == "lin_L4":
                    r = arm_lin(K, Z, 4)(Q)
                elif a == "norm_L4":
                    r = arm_norm(K, Z, 4)(Q)
                elif a == "lin_L1_m512":
                    fn, K2 = arm_lin_width(K, Z, 512)
                    r = fn(K2[:nq])
                elif a.startswith("slot_"):
                    S = int(a.split("S")[1].split("_")[0])
                    L = int(a.split("_")[1][1:])          # slot_L4_S40 -> 4
                    fn, stores = arm_slot(K, Z, S, L)
                    best = torch.full((nq,), -2.0)
                    bestidx = torch.zeros(nq, dtype=torch.long)
                    for r, Zl, idx in fn(Q):
                        sc = unit(r) @ unit(Zl).t()
                        mx, am = sc.max(dim=-1)
                        upd = mx > best
                        best[upd] = mx[upd]
                        bestidx[upd] = idx[am[upd]]
                    hit = (bestidx == tgt).float().mean()
                if hit is None:
                    hit = (top1(r, Z) == tgt).float().mean()
                acc[a].append(float(hit))
        rows.append({"n": n, "n_over_m": round(n / m, 2),
                     **{a: round(sum(v) / len(v), 4) for a, v in acc.items()}})
    return rows


def run_combo(ks, seeds, m: int, n: int = 200):
    """k 事实线性组合：q = normalize(Σk_i)，目标 t = Σz_i，报 cos(r,t) 与"组合首名"命中。"""
    out = []
    for k in ks:
        rec = {"k": k, "lin_L1": [], "norm_L4": []}
        for seed in seeds:
            K, Z = make_facts(n, m, seed)
            g = torch.Generator().manual_seed(seed + 100)
            W1 = ridge_write(K, Z)
            Wn = [ridge_write(K, Z) for _ in range(4)]
            for _ in range(16):
                idx = torch.randint(0, n, (k,), generator=g)
                q = unit(K[idx].sum(0, keepdim=True))
                t = unit(Z[idx].sum(0, keepdim=True))
                r1 = q @ W1
                rec["lin_L1"].append(float(torch.cosine_similarity(r1, t, dim=-1)))
                qn = q
                for W in Wn:
                    qn = qn + unit(qn @ W)
                rec["norm_L4"].append(float(torch.cosine_similarity(qn, t, dim=-1)))
        out.append({"k": k, **{a: round(sum(v) / len(v), 4)
                               for a, v in rec.items() if a != "k"}})
    return out


def run_condition(ns, seeds, m: int):
    """诊断：A = KᵀK + λI 的条件数与 ||W||₂ —— 解释"深度在 N≈m 处爆炸"。"""
    out = []
    for n in ns:
        conds, wn = [], []
        for seed in seeds:
            K, Z = make_facts(n, m, seed)
            A = K.t() @ K + LAM * torch.eye(m)
            conds.append(float(torch.linalg.cond(A)))
            W = torch.linalg.solve(A, K.t() @ Z)
            wn.append(float(torch.linalg.matrix_norm(W, ord=2)))
        out.append({"n": n, "cond_A": round(sum(conds) / len(conds), 1),
                    "w_norm2": round(sum(wn) / len(wn), 3)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[64, 128, 256, 512, 1024])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--out", default=os.path.join("_ab_tmp", "tower_depth.json"))
    args = ap.parse_args()

    global LAM
    LAM = args.lam
    t0 = time.time()
    arms = ["lin_L1", "lin_L2", "lin_L4", "lin_L1_m512", "norm_L4",
            "slot_L1_S40", "slot_L1_S160", "slot_L4_S40"]
    cap = run_capacity(args.ns, args.seeds, args.m, arms)
    cond = run_condition(args.ns, args.seeds, args.m)
    combos = run_combo(args.ks, args.seeds, args.m)

    hdr = ["n", "N/m"] + arms
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in cap:
        cells = [str(r["n"]), f'{r["n_over_m"]:.2f}'] + [f'{r[a]:.3f}' for a in arms]
        print("| " + " | ".join(cells) + " |")

    print("\n| k（事实数） | " + " | ".join(a for a in ("lin_L1", "norm_L4")) + " |")
    print("|---|" + "---|" * 2)
    for r in combos:
        print(f'| {r["k"]} | {r["lin_L1"]:.4f} | {r["norm_L4"]:.4f} |')

    print("\n| n | N/m | cond(KᵀK+λI) | ‖W‖₂ |")
    print("|---|---|---|---|")
    for r in cond:
        print(f'| {r["n"]} | {r["n"] / args.m:.2f} | {r["cond_A"]:,.0f} | {r["w_norm2"]:.3f} |')

    payload = {"m": args.m, "lam": LAM, "dv": args.m, "seeds": args.seeds,
               "capacity": cap, "conditioning": cond, "combo": combos,
               "elapsed_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[out] {args.out}  ({payload['elapsed_s']}s)")


if __name__ == "__main__":
    main()


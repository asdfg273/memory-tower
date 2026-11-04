r"""tools_tower_slotdepth.py —— T131：槽位塔端到端（真模型）——容量墙在哪、地址乘法成不成立

问题
----
T129（tools_tower_depth.py）在纯线性代数上发现：深度的唯一稳收益是"地址乘法"——
L4×S40（每层内容维 m-S）在 N=2048 时 hit@1 = 1.000，而把同样 160 个地址塞进一层的
S160（内容维 m-S）只有 0.779；容量本身只由 m 决定。

本脚本把这条搬到真模型上：键/值来自 7B 的 hidden（2048 维），
    --proj 0   裸嵌入（维度 2048，m 很大 → 墙应该很远）
    --proj 1   过 w_k/w_v 投影（m=256，交付默认形态 → 墙应该更近）
用闭式岭回归写、余弦读出。检验：
    1) 容量的墙在哪里（哪个 N 开始 hit@1 掉）
    2) 同一份"地址预算"下，2 层×S 槽是不是比 1 层×2S 槽更能装（地址乘法）

臂
--
    L*_S0   无槽位（交付默认形态）：键 = normalize(内容)
    L*_S20/40  槽位键 = normalize([onehot(S); 内容(m-S)])，事实 i → 槽 i%S、层 (i//S)%L

跑法
----
    $env:HF_HUB_OFFLINE="1"
    venv\Scripts\python.exe tools_tower_slotdepth.py --proj 0 --ns 160,320,640
    venv\Scripts\python.exe tools_tower_slotdepth.py --proj 1 --ns 160,320,640
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from tools_tower_vs_rag import Q_VERB, build_facts


# ----------------------------------------------------------------------------
# 编码（慢，做缓存；proj 决定键/值是否过 w_k/w_v）
# ----------------------------------------------------------------------------
def encode_all(n: int, cache_path: str, device: str, proj: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        if blob.get("n") == n:
            return blob["K"], blob["Z"]

    facts, _names = build_facts(n, "single", "single")
    from tools_tower_memory import Encoder

    enc = Encoder(proj=proj, device=device, verbose=True)
    t0 = time.time()
    K, Z = [], []
    for i, (ent, _a, _b, name) in enumerate(facts):
        K.append(enc.key(Q_VERB.format(q=ent)))
        Z.append(enc.value(name))
        if (i + 1) % 100 == 0:
            print(f"[enc] {i + 1}/{n}  {time.time() - t0:.1f}s", flush=True)
    K = torch.stack(K).float().cpu()
    Z = torch.stack(Z).float().cpu()
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save({"n": n, "K": K, "Z": Z}, cache_path)
    print(f"[enc] n={n} dim={K.shape[1]} 编码完成 {time.time() - t0:.1f}s -> {cache_path}", flush=True)
    return K, Z


# ----------------------------------------------------------------------------
# 槽位键与读出
# ----------------------------------------------------------------------------
def slot_keys(mat: torch.Tensor, S: int, L: int) -> List[Tuple[torch.Tensor, List[int]]]:
    """按 (层, 槽) 切出键矩阵：返回 [(K_ls, 全局行号列表)]。"""
    n, d = mat.shape
    if S <= 0:
        return [(F.normalize(mat, dim=-1), list(range(n)))]
    out = []
    for layer in range(L):
        for slot in range(S):
            rows = [i for i in range(n) if (i // S) % L == layer and i % S == slot]
            if not rows:
                continue
            k = torch.zeros(len(rows), S + d - S)
            k[:, slot] = 1.0
            k[:, S:] = mat[rows, : d - S]
            out.append((F.normalize(k, dim=-1), rows))
    return out


def query_matrix(mat: torch.Tensor, S: int) -> torch.Tensor:
    """(n, d) -> (S, n, d')：每条查询在每个槽下各拼一次（S=0 时只有一个"槽"）。"""
    n, d = mat.shape
    if S <= 0:
        return F.normalize(mat, dim=-1).unsqueeze(0)
    out = torch.zeros(S, n, S + d - S)
    for slot in range(S):
        out[slot, :, slot] = 1.0
        out[slot, :, S:] = mat[:, : d - S]
    return F.normalize(out, dim=-1)


def evaluate(K: torch.Tensor, Z: torch.Tensor, L: int, S: int, lam: float) -> Tuple[float, float]:
    """返回 (hit@1, 正确-最强的平均间隙)。"""
    n, d = K.shape
    Zl = F.normalize(Z, dim=-1)
    Q = query_matrix(K, S)                      # (nq, n, d')
    sim_tot = torch.full((n, n), -2.0)
    for keys, rows in slot_keys(K, S, L):
        if not rows:
            continue
        G = keys @ keys.t() + lam * torch.eye(len(rows))
        alpha = torch.linalg.solve(G, Z[rows])  # (n_l, d)
        slot = int(torch.argmax(keys[0, :S])) if S > 0 else 0
        q = Q[slot]                             # 只用与该槽匹配的那一支查询
        r = torch.einsum("nd,vd->nv", q, keys)  # (n, n_l)
        r = torch.einsum("nv,vd->nd", r, alpha)  # (n, d)
        sim_tot = torch.maximum(sim_tot, torch.einsum("nd,vd->nv", F.normalize(r, dim=-1), Zl))
    pred = sim_tot.argmax(dim=1)
    hit = float((pred == torch.arange(n)).sum()) / n
    correct = sim_tot.diag()
    off = sim_tot.masked_fill(torch.eye(n, dtype=torch.bool), -2.0).max(dim=1).values
    return hit, float((correct - off).mean())


ARMS = [(1, 0), (1, 4), (2, 4), (4, 4), (8, 4), (1, 8), (4, 8), (1, 20), (1, 40), (2, 20), (4, 20)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="160,320,640")
    ap.add_argument("--proj", type=int, default=0)
    ap.add_argument("--dims", default="", help="逗号分隔的 m' 截断列表（空=用原始维度）")
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache-dir", default="_ab_tmp")
    ap.add_argument("--out", default="_ab_tmp/tower_slotdepth.json")
    args = ap.parse_args()

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    dims = [int(x) for x in args.dims.split(",") if x.strip()] or [0]
    tag = f"proj{args.proj}"
    table: Dict[str, Dict[str, float]] = {}
    margins: Dict[str, Dict[str, float]] = {}
    for n in ns:
        cache = os.path.join(args.cache_dir, f"slotdepth_{tag}_kv_{n}.pt")
        K, Z = encode_all(n, cache, args.device, bool(args.proj))
        for mcut in dims:
            Kc = K if mcut <= 0 else K[:, :mcut].contiguous()
            d = Kc.shape[1]
            for L, S in ARMS:
                if S * L > n or S >= d - 8:
                    continue
                name = f"L{L}_S{S}"
                t0 = time.time()
                hit, gap = evaluate(Kc, Z, L, S, args.lam)
                col = f"N{n}_m{d}"
                table.setdefault(name, {})[col] = round(hit, 4)
                margins.setdefault(name, {})[col] = round(gap, 4)
                print(f"[arm] {tag} N={n:<5} m={d:<5} {name:<8} hit@1={hit:.3f}  gap={gap:+.3f}"
                      f"  {time.time() - t0:.1f}s", flush=True)

    payload = {"note": "T131 槽位塔端到端", "proj": bool(args.proj), "lam": args.lam,
               "hit_at_1": table, "gap": margins}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()

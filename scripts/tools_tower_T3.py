r"""
tools_tower_T3.py —— **T3：完整写入—读取路径 + 四分支归因**
==========================================================

T0 的缺口（必须在本步补上）
-------------------------
```
T0 里 v = w_v(keys)      -> **value 复用 key 的表征来源**，不是独立存储
T0 里 K0/V0 直接交给网络 -> **write 是 oracle**，没测写入
```

T3 的设计
--------
```
1. **key 与 value 独立**：value 由独立的随机向量给出（不挂在 key 上）
2. **真实写入**：快速权重 MLP + 梯度下降式更新（复刻 memory_tower 的做法）
      loss_w = MSE( apply_memory(w_k(x), w), "value 的编码" )
      w <- w - lr_w * grad(loss_w, w)          # **只更新 w，不更新 w_k/w_v**
3. **四阶段分别度量**，以归因：
      阶段 1  write 后**存储中**能否取回？（读快速权重、比对 value 编码）
      阶段 2  query/read 选对 slot 吗？（slot top1）
      阶段 3  raw read 的值排名？
      阶段 4  projection 后的值排名？（**同空间比较**：投影后的 target）
```

四分支归因（按规格）
------------------
| 现象 | 归因 |
|---|---|
| oracle write 都读不出 | **read/query 路径** |
| oracle write 能读、完整 write 不能 | **write/update 路径** |
| raw read 能读、projection 后不能 | **`site_proj`** |
| projection 能读、LM 解码不能 | LM 注入或 decoder（本步不测 LM） |

**「oracle write」定义**：跳过写入，直接把 value 编码塞进 slot。
**「identity read」定义**：`site_proj = Identity`，故阶段 3 与阶段 4 相同。

用法
----
    python tools_tower_T3.py --sizes 4,8,16 --modes oracle,fastweight
    python tools_tower_T3.py --report
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import write_json

DB = os.path.join("data", "tower_T3.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS res (
    r_id INTEGER PRIMARY KEY AUTOINCREMENT,
    M INTEGER, mode TEXT, projection TEXT, arm TEXT, ep INTEGER,
    store_top1 REAL, slot_top1 REAL, raw_vtop1 REAL, raw_vrank REAL,
    proj_vtop1 REAL, proj_vrank REAL, proj_norm REAL, target_norm REAL, wnorm REAL
);
"""


class FastWeightMemory(nn.Module):
    """**复刻 memory_tower 的写入方式**：两层 MLP 的权重作为"快速权重"，
    写入时用梯度下降更新它（而非靠反向传播穿过整个网络）。

    这正是原实现的机制 —— 也是本步要检验的环节。
    """
    def __init__(self, d: int, m: int, layers: int = 2):
        super().__init__()
        self.d, self.m, self.layers = d, m, layers
        # **统一到 m 维**：写 key / 读 query / 值编码都在 m 空间，
        # 这样 apply_memory 的输入输出同维，且 raw 与 value 编码同空间（T2 要求）。
        self.w_k = nn.Linear(d, m, bias=False)     # 写路径的 key（兼寻址）
        self.w_v = nn.Linear(d, m, bias=False)     # 写入目标的编码
        self.w_q = nn.Linear(d, m, bias=False)     # 读路径的 query

    # ---------- 快速权重的初始化与读写 ----------
    def init_fast(self, device, dtype=torch.float32) -> Dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(7)
        w0 = torch.randn(self.m, self.m, generator=g) * (1.0 / self.m ** 0.5)
        w1 = torch.randn(self.m, self.m, generator=g) * (1.0 / self.m ** 0.5)
        return {"w0": w0.to(device, dtype), "w1": w1.to(device, dtype)}

    def apply_memory(self, x: torch.Tensor, w: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.gelu(x @ w["w0"]) @ w["w1"]

    def write_step(self, w: Dict[str, torch.Tensor], x: torch.Tensor,
                   target: torch.Tensor, lr_w: float, steps: int = 1):
        """**梯度下降式写入**：只更新快速权重 w，慢权重（w_k/w_v/w_q）不参与。

        ⚠ **必须自己开 `enable_grad()`**：调用方（评测段）可能在 `no_grad()` 里，
        而这里要对快速权重做一次**独立的内层梯度更新** ——
        它不应该被外层的 `no_grad` 关掉。
        """
        with torch.enable_grad():
            xd = x.detach().clone().requires_grad_(False)
            tgt = target.detach().clone()
            w = {k: v.detach().clone().requires_grad_(True) for k, v in w.items()}
            for _ in range(steps):
                pred = self.apply_memory(xd, w)
                loss = F.mse_loss(pred, tgt)
                grads = torch.autograd.grad(loss, list(w.values()),
                                            allow_unused=False)
                w = {k: (v - lr_w * g).detach().requires_grad_(True)
                     for (k, v), g in zip(w.items(), grads)}
            out = {k: v.detach() for k, v in w.items()}
            return out, float(loss.detach())

    def read(self, w: Dict[str, torch.Tensor], q: torch.Tensor) -> torch.Tensor:
        return self.apply_memory(q, w)


class Reader(nn.Module):
    """读路径：把 w_q 输出与各 slot 的 key 做寻址，再投影。"""
    def __init__(self, d: int, m: int, hidden: int, projection: str):
        # 注：`d` 保留以说明来源维度；实际寻址/读出都在 m 维
        super().__init__()
        self.projection = projection
        if projection == "identity":
            self.site_proj = nn.Identity()
        elif projection == "linear":
            self.site_proj = nn.Linear(m, hidden)
        elif projection == "whiten":
            self.site_proj = nn.Sequential(nn.Linear(m, hidden), nn.LayerNorm(hidden))
        else:
            raise ValueError(projection)
        self.read_gate = nn.Linear(m + m, hidden)   # (query_m, raw_m)

    def forward(self, q_vec: torch.Tensor, slot_keys: torch.Tensor,
                slot_raw: torch.Tensor):
        """
        q_vec     : **(m,)** 查询向量（与 key 同空间）
        slot_keys : (M, m) 各 slot 的 key
        slot_raw  : (M, m) 各 slot 的**已存储内容**
        """
        q = q_vec.reshape(-1)                      # (m,)
        sim = slot_keys @ q                        # (M,)  寻址分数
        attn = F.softmax(sim, dim=-1)              # (M,)
        raw = attn @ slot_raw                      # (m,)  与目标同空间
        gate = torch.sigmoid(self.read_gate(torch.cat([q, raw], -1)))   # (hidden,)
        if self.projection == "identity":
            proj = raw * gate[:raw.shape[-1]]
        else:
            proj = self.site_proj(raw) * gate
        return raw, proj, sim


def run_one(M: int, D: int, hidden: int, mdim: int, mode: str, projection: str,
            epochs: int, seed: int, lr: float, lr_w: float, device: str):
    g = torch.Generator().manual_seed(seed)
    # **key 与 value 独立**（value 不挂在 key 上）
    K = torch.randn(M, D, generator=g); K = K / K.norm(dim=-1, keepdim=True)
    VV = torch.randn(M, D, generator=g); VV = VV / VV.norm(dim=-1, keepdim=True)
    K, VV = K.to(device), VV.to(device)

    mem = FastWeightMemory(D, mdim).to(device)
    rd = Reader(D, mdim, hidden, projection).to(device)
    params = (list(mem.w_k.parameters()) + list(mem.w_v.parameters())
              + list(mem.w_q.parameters()) + list(rd.parameters()))
    opt = torch.optim.AdamW(params, lr=lr)
    q_idx = torch.arange(M, device=device)

    def forward_all():
        """一次前向，返回 (store_top1, slot_top1, raw, proj, v_enc, wnorm)。"""
        k_proj = mem.w_k(K)                     # (M, m) 写路径 key
        v_enc = mem.w_v(VV)                     # (M, m) 写入目标的编码
        if mode == "oracle":
            slot_raw = v_enc.detach().clone()   # 跳过写入（oracle）
            wnorm = 0.0
        else:
            w = mem.init_fast(device)
            raws = []
            for i in range(M):
                # **写入**：只更新快速权重 w
                w, _ = mem.write_step(w, k_proj[i], v_enc[i].detach(), lr_w)
                raws.append(mem.read(w, k_proj[i]))     # 写入后立即读回
            slot_raw = torch.stack(raws, 0)             # (M, m)
            wnorm = float(sum(v.norm() for v in w.values()))
        # 存储取回：slot_raw 与 v_enc 比对
        s_store = slot_raw @ F.normalize(v_enc, dim=-1).t()
        store_top1 = float((s_store.argmax(-1) == q_idx).float().mean())

        q = mem.w_q(K)                          # (M, m)
        raw_l, proj_l, sim_l = [], [], []
        for i in range(M):
            r_, p_, s_ = rd(q[i], k_proj, slot_raw)
            raw_l.append(r_); proj_l.append(p_); sim_l.append(s_)
        raw = torch.stack(raw_l, 0)             # (M, m)
        proj = torch.stack(proj_l, 0)           # (M, hidden) 或 (M, m)
        sim = torch.stack(sim_l, 0)             # (M, M)
        slot_top1 = float((sim.argmax(-1) == q_idx).float().mean())
        return store_top1, slot_top1, raw, proj, sim, v_enc, wnorm

    for ep in range(1, epochs + 1):
        store_top1, slot_top1, raw, proj, sim, v_enc, _ = forward_all()
        slot_ce = F.cross_entropy(sim, q_idx)
        cos = 1.0 - F.cosine_similarity(raw, F.normalize(v_enc, dim=-1), dim=-1).mean()
        info = F.cross_entropy(raw @ F.normalize(v_enc, dim=-1).t() * 10.0, q_idx)
        loss = slot_ce + 1.0 * cos + 1.0 * info
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    with torch.no_grad():
        store_top1, slot_top1, raw, proj, sim, v_enc, wnorm = forward_all()

        def vrank(x: torch.Tensor, tgt: torch.Tensor):
            """**同空间比较**：目标也过同一投影，再比余弦。"""
            t2 = tgt if (rd.projection == "identity" or x.shape[-1] == tgt.shape[-1]) \
                else rd.site_proj(tgt)
            s = x @ F.normalize(t2, dim=-1).t()
            order = torch.argsort(s, dim=-1, descending=True)
            rk = (order == q_idx.unsqueeze(1)).float().argmax(1) + 1
            return float((rk == 1).float().mean()), float(rk.float().median())

        raw_t1, raw_rk = vrank(raw, v_enc)
        proj_t1, proj_rk = vrank(proj, v_enc)
        return dict(store_top1=store_top1, slot_top1=slot_top1,
                    raw_vtop1=raw_t1, raw_vrank=raw_rk,
                    proj_vtop1=proj_t1, proj_vrank=proj_rk,
                    proj_norm=float(proj.norm(dim=-1).mean()),
                    target_norm=float(v_enc.norm(dim=-1).mean()),
                    wnorm=wnorm)


def main() -> None:
    ap = argparse.ArgumentParser(description="T3：完整写入—读取 + 四分支归因")
    ap.add_argument("--sizes", type=str, default="4,8,16")
    ap.add_argument("--modes", type=str, default="oracle,fastweight")
    ap.add_argument("--projections", type=str, default="identity,linear")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--mem-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-w", type=float, default=1e-2)
    ap.add_argument("--db", type=str, default=DB)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", type=str,
                    default=os.path.join("reports", "tower_T3.json"))
    args = ap.parse_args()

    if args.report:
        report(args.db, args.out)
        return

    dev = "cpu"     # **无 LM，纯塔，CPU 足够且不占显存**
    con = sqlite3.connect(args.db)
    con.executescript(SCHEMA)
    con.commit()
    sizes = [int(x) for x in args.sizes.split(",")]
    modes = args.modes.split(",")
    projs = args.projections.split(",")

    print("=" * 104)
    print("**T3：完整写入—读取路径 + 四分支归因**")
    print("=" * 104)
    print("  key 与 value **独立**（value 不再挂在 key 上）")
    print("  写入 = **梯度下降式快速权重**（复刻 memory_tower）")
    print(f"  M={sizes}  modes={modes}  projections={projs}  "
          f"mem_dim={args.mem_dim}  lr_w={args.lr_w}")

    for mode in modes:
        for pj in projs:
            print(f"\n  ===== mode={mode}  projection={pj} =====")
            print(f"  {'M':<5}{'存储取回':<10}{'slot top1':<11}{'raw 值top1':<11}"
                  f"{'raw 排名':<10}{'proj 值top1':<12}{'proj 排名':<11}{'|w|'}")
            for M in sizes:
                vals: List[Dict[str, float]] = []
                for ep in range(args.episodes):
                    try:
                        vals.append(run_one(M, args.dim, args.hidden, args.mem_dim,
                                            mode, pj, args.epochs, seed=2000 + ep,
                                            lr=args.lr, lr_w=args.lr_w, device=dev))
                    except Exception as e:
                        print(f"  {M:<5} 失败: {type(e).__name__}: {str(e)[:60]}")
                        vals = []
                        break
                if not vals:
                    continue
                n = len(vals)
                f = lambda k: sum(v[k] for v in vals) / n
                print(f"  {M:<5}{f('store_top1'):<10.3f}{f('slot_top1'):<11.3f}"
                      f"{f('raw_vtop1'):<11.3f}{f('raw_vrank'):<10.2f}"
                      f"{f('proj_vtop1'):<12.3f}{f('proj_vrank'):<11.2f}"
                      f"{f('wnorm'):.1f}")
                for ep, v in enumerate(vals):
                    con.execute(
                        "INSERT INTO res(M,mode,projection,arm,ep,store_top1,"
                        "slot_top1,raw_vtop1,raw_vrank,proj_vtop1,proj_vrank,"
                        "proj_norm,target_norm,wnorm) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (M, mode, pj, "correct", ep, v["store_top1"], v["slot_top1"],
                         v["raw_vtop1"], v["raw_vrank"], v["proj_vtop1"],
                         v["proj_vrank"], v["proj_norm"], v["target_norm"],
                         v["wnorm"]))
                con.commit()
    con.close()
    print("\n跑完。--report 出归因。")


def report(db: str, out_path: str) -> None:
    con = sqlite3.connect(db)
    rows = con.execute("SELECT M,mode,projection,store_top1,slot_top1,raw_vtop1,"
                       "raw_vrank,proj_vtop1,proj_vrank FROM res").fetchall()
    con.close()
    if not rows:
        print("  尚无数据。"); return
    from collections import defaultdict
    agg: Dict[Tuple, List] = defaultdict(list)
    for r in rows:
        agg[(r[0], r[1], r[2])].append(r[3:])
    print("=" * 104)
    print("T3 归因：**哪一段断了**")
    print("=" * 104)
    print(f"  {'M':<5}{'mode':<12}{'proj':<10}{'存储取回':<10}{'slot':<8}"
          f"{'raw 值':<9}{'proj 值':<10}{'归因'}")
    fail_where: Dict[str, int] = defaultdict(int)
    for k in sorted(agg):
        M, mode, pj = k
        v = agg[k]; n = len(v)
        f = lambda i: sum(x[i] for x in v) / n
        store, slot, rt1, rrk, pt1, prk = f(0), f(1), f(2), f(3), f(4), f(5)
        if store < 0.5:
            why = "**write/update 路径**（存进去就取不回）"
        elif slot < 0.5:
            why = "**read/query 路径**（选不中 slot）"
        elif pj != "identity" and pt1 < 0.5 <= rt1:
            why = "**site_proj**（raw 能读、投影后不能）"
        elif min(rt1, pt1) >= 0.9:
            why = "✅ 通"
        else:
            why = "值检索精度不足"
        fail_where[why] += 1
        print(f"  {M:<5}{mode:<12}{pj:<10}{store:<10.3f}{slot:<8.3f}"
              f"{rt1:<9.3f}{pt1:<10.3f}{why}")
    print()
    print("  **四分支归因汇总**：")
    for k, c in sorted(fail_where.items(), key=lambda x: -x[1]):
        print(f"    {c:>2} 格  {k}")
    print()
    # oracle vs fastweight 的直接对照
    o = {k: v for k, v in agg.items() if k[1] == "oracle"}
    fw = {k: v for k, v in agg.items() if k[1] == "fastweight"}
    if o and fw:
        def mean_of(d, i):
            allv = [x for v in d.values() for x in v]
            return sum(x[i] for x in allv) / len(allv)
        print("  **oracle write  vs  fastweight write**")
        print(f"    oracle      存储取回 {mean_of(o,0):.3f}  slot {mean_of(o,1):.3f}  "
              f"raw 值 top1 {mean_of(o,2):.3f}")
        print(f"    fastweight  存储取回 {mean_of(fw,0):.3f}  slot {mean_of(fw,1):.3f}  "
              f"raw 值 top1 {mean_of(fw,2):.3f}")
        if mean_of(o, 2) - mean_of(fw, 2) > 0.3:
            print("    -> ✅ **oracle 能读、完整 write 不能 -> 归因：write/update 路径**")
        elif mean_of(fw, 2) >= 0.9:
            print("    -> ✅ 完整写入路径也能读出 -> **塔的写读闭环成立**")
    write_json(out_path, {"cells": len(agg), "fail_where": dict(fail_where)})
    print(f"\n[T3] 报告 -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="4,8,16")
    ap.add_argument("--modes", type=str, default="oracle,fastweight")
    ap.add_argument("--projections", type=str, default="identity,linear")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--mem-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-w", type=float, default=1e-2)
    ap.add_argument("--db", type=str, default=DB)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", type=str,
                    default=os.path.join("reports", "tower_T3.json"))
    a, _ = ap.parse_known_args()
    if a.report:
        report(a.db, a.out)
    else:
        main()

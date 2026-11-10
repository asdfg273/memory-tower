"""V 阶段（纯文本）：塔的读出到底是"检索器"还是"复现器"？

背景：G 阶段发现闭式读出只能落在 span(K) 里。这条对"把塔向量当记忆喂给下游"是致命的，
但对"塔当检索器"可能完全无害——因为排序只需要方向对，不需要值一致。

本脚本在真实文本记忆上量三件事（不碰图片）：
  S1 选择准确率：用一条记忆自己的键去查，top-1 是否命中它自己（塔当索引的能力）
  S2 复现保真度：读出向量与真实值向量的余弦（把向量喂给下游的前提）
  S3 扩容能否救复现：把键/值维度从 256 抬到 512/1024，保真度是否上升？
     （理论上不能：rank(K) ≤ 事实条数，扩维不增加秩——这里用数字钉死它）

用法：
    venv\\Scripts\\python.exe tools_tower_valuecheck.py --out reports/repro  # 只读，不写塔
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from typing import Dict, List, Tuple

import torch

MEM_DB = r"data\memory.db"
TOWER_DB = r"data\tower_mem\agent.jsonl"


def load_texts() -> List[Tuple[str, str]]:
    """返回 [(来源, 文本)]：塔里的事实 + 平台记忆库里的条目。"""
    out: List[Tuple[str, str]] = []
    if os.path.exists(TOWER_DB):
        with open(TOWER_DB, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("deleted") or row.get("alive") is False:
                    continue
                t = (row.get("text") or "").strip()
                if t:
                    out.append(("tower", t))
    if os.path.exists(MEM_DB):
        try:
            con = sqlite3.connect(f"file:{MEM_DB}?mode=ro&immutable=1", uri=True)
            for (raw,) in con.execute("SELECT raw_source FROM memories WHERE deleted_at IS NULL"):
                t = (raw or "").strip()
                if t:
                    out.append(("db", t))
            con.close()
        except sqlite3.Error as exc:
            print(f"[warn] memory.db 读取失败：{exc}", flush=True)
    # 去重，保序
    seen, uniq = set(), []
    for src, t in out:
        if t not in seen:
            seen.add(t)
            uniq.append((src, t))
    return uniq


def fit_recall(K: torch.Tensor, Z: torch.Tensor, lam: float = 1e-2) -> Tuple[torch.Tensor, torch.Tensor]:
    rec = K @ torch.linalg.solve(K.t() @ K + lam * torch.eye(K.shape[1]), K.t() @ Z)
    return rec, Z


def measure(K: torch.Tensor, Z: torch.Tensor) -> Dict[str, float]:
    rec, _ = fit_recall(K, Z)
    cos = torch.nn.functional.cosine_similarity(rec, Z, dim=1)
    # S1：用键自身去查值矩阵，top-1 是否命中自己
    sims = K @ Z.t()                      # (n,n)
    top1 = (sims.argmax(dim=1) == torch.arange(len(K))).float().mean()
    # 秩与在空间内比例（P 是到 span(K) 的投影）
    n, d = K.shape
    P = K.t() @ torch.linalg.solve(K @ K.t() + 1e-6 * torch.eye(n), K)
    PZ = Z @ P.t()
    in_span = float((PZ * Z).sum() / (Z * Z).sum())
    return {
        "n": n, "d": d,
        "rank_K": int(torch.linalg.matrix_rank(K)),
        "select_top1": round(float(top1), 4),
        "recon_cos_mean": round(float(cos.mean()), 4),
        "recon_cos_min": round(float(cos.min()), 4),
        "value_in_span_frac": round(in_span, 4),
        "value_in_span_ceil_cos": round(in_span ** 0.5, 4),
    }


def project(x: torch.Tensor, d: int, seed: int) -> torch.Tensor:
    """固定随机高斯投影到 d 维，再单位化——模拟 Encoder 的 proj 分支。"""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(x.shape[1], d, generator=g) / (x.shape[1] ** 0.5)
    y = x @ W
    return torch.nn.functional.normalize(y, dim=-1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"reports\repro")
    ap.add_argument("--dims", default="256,512,1024")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    from tools_tower_memory import TowerMemory

    tm = TowerMemory(TOWER_DB, verbose=False)
    rows = load_texts()
    texts = [t for _, t in rows]
    print(f"[data] {len(texts)} 条文本（塔 {sum(1 for s,_ in rows if s=='tower')} / 库 {sum(1 for s,_ in rows if s=='db')}）", flush=True)

    K2048 = torch.stack([tm.enc.key(t).float() for t in texts])       # (n, 2048)
    V2048 = torch.stack([tm.enc.value(t).float() for t in texts])

    results = {}
    for d in [int(x) for x in args.dims.split(",")]:
        K = project(K2048, d, seed=11)
        Z = project(V2048, d, seed=12)
        results[f"d{d}"] = measure(K, Z)
        print(f"[d={d}] " + json.dumps(results[f"d{d}"], ensure_ascii=False), flush=True)

    # 稠密 2048 维（不降维，看纯空间大小的作用）
    results["d2048"] = measure(torch.nn.functional.normalize(K2048, dim=-1),
                               torch.nn.functional.normalize(V2048, dim=-1))
    print("[d=2048] " + json.dumps(results["d2048"], ensure_ascii=False), flush=True)

    out = {"source": "tower agent.jsonl + memory.db", "texts": len(texts),
           "results": results, "seconds": round(time.time() - t0, 1)}
    path = os.path.join(args.out, "tower_valuecheck.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("[summary_json] " + json.dumps(out, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

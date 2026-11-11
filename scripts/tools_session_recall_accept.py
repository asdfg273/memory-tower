"""H11 验收：同一批冻结题集上，把塔的四种读出与三种词面基线放在一张表里。

预注册见 reports/SESSION_RECALL.md §12.7（跑之前写定）：
  1. 题集：_ab_tmp/gold200.json（200 条，sha256 校验后冻结）
  2. 白化：Σ 只用键/值自身的协方差，收缩强度 Ledoit–Wolf 自动估计，不在题上选
  3. 读出两条都报：白化键余弦、白化后岭回归
  4. H11 逐字复跑：同语料、同单元、同 λ，只换读出
  5. 统计：bootstrap 95% CI + McNemar(b/c)
  6. 判据：白化塔 ≥ bigram − 1 SE 则"输给词面"不成立；< − 2 SE 则封板

用法：
    venv\\Scripts\\python.exe tools_session_recall_accept.py --gold _ab_tmp/gold200.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import binomtest
from sklearn.covariance import ledoit_wolf

from memory_service import _char_grams
from tools_session_recall import load_corpus
from tools_session_recall_hier import to_units
from tools_tower_memory import Encoder

CACHE = Path("_ab_tmp/accept")
LAM = 1e-2
M = 256


# ------------------------------------------------------------------ 编码
def encode_units(texts: list[str], tag: str) -> tuple[np.ndarray, np.ndarray]:
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"{tag}.npz"
    if cp.exists():
        d = np.load(cp)
        return d["K"], d["Z"]
    enc = Encoder(m=M, proj=True, verbose=False)
    ks, zs = [], []
    t0 = time.time()
    for i, t in enumerate(texts):
        ks.append(enc.key(t).detach().cpu().numpy())
        zs.append(enc.value(t).detach().cpu().numpy())
        if (i + 1) % 400 == 0:
            print(f"[enc:{tag}] {i + 1}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    K = _norm(np.asarray(ks, dtype=np.float32))
    Z = _norm(np.asarray(zs, dtype=np.float32))
    np.savez(cp, K=K, Z=Z)
    return K, Z


def _norm(A: np.ndarray) -> np.ndarray:
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


# ------------------------------------------------------------------ 白化
def whiten(K: np.ndarray, Q: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Ledoit–Wolf 收缩协方差 + 对称白化 Σ^{-1/2}。"""
    cov, shrink = ledoit_wolf(K, assume_centered=False)
    evals, evecs = np.linalg.eigh(cov.astype(np.float64))
    evals = np.maximum(evals, 1e-8)
    Wm = (evecs / np.sqrt(evals)) @ evecs.T
    print(f"[whiten:{name}] LW shrinkage={shrink:.4f}", flush=True)
    return _norm(K @ Wm), _norm(Q @ Wm), float(shrink)


# ------------------------------------------------------------------ 词面
def lex_matrix(texts: list[str], n: int, weighted: bool):
    """返回 (行归一化后的稀疏矩阵, vocab, idf)。"""
    df: dict[str, int] = {}
    rows, cols, vals = [], [], []
    vocab: dict[str, int] = {}
    for i, t in enumerate(texts):
        cnt: dict[str, int] = {}
        for g in _char_grams(t, n):
            cnt[g] = cnt.get(g, 0) + 1
        for g, c in cnt.items():
            if g not in vocab:
                vocab[g] = len(vocab)
            df[g] = df.get(g, 0) + 1
            rows.append(i)
            cols.append(vocab[g])
            vals.append(float(c))
    N = max(1, len(texts))
    import math

    idf = {g: math.log((N + 1) / (c + 0.5)) for g, c in df.items()}
    inv = {v: g for g, v in vocab.items()}
    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(len(texts), len(vocab)))
    if weighted:
        w = np.array([idf[inv[j]] for j in range(len(vocab))], dtype=np.float64)
        mat = mat @ sparse.diags(w)
    norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1 + 1e-9
    mat = sparse.diags(1.0 / norms) @ mat
    return mat.tocsr(), vocab, idf


def lex_query(texts: list[str], vocab: dict[str, int], idf: dict[str, float], n: int):
    rows, cols, vals = [], [], []
    for i, t in enumerate(texts):
        cnt: dict[str, int] = {}
        for g in _char_grams(t, n):
            if g in vocab:
                cnt[g] = cnt.get(g, 0) + 1
        for g, c in cnt.items():
            rows.append(i)
            cols.append(vocab[g])
            vals.append(float(c) * idf.get(g, 1.0))
    q = sparse.csr_matrix((vals, (rows, cols)), shape=(len(texts), len(vocab)))
    qn = np.sqrt(q.multiply(q).sum(axis=1)).A1 + 1e-9
    return sparse.diags(1.0 / qn) @ q


# ------------------------------------------------------------------ 指标
def max_by_ref(scores: np.ndarray, refs: np.ndarray, gold_refs: list[str]) -> np.ndarray:
    """(n_query, n_unit) → 每条 gold 的消息级 top-1 是否命中，以及名次。"""
    ranks = np.full(len(gold_refs), 999, dtype=np.int32)
    for i, g in enumerate(gold_refs):
        agg: dict[str, float] = {}
        for s, r in zip(scores[i], refs):
            if s > agg.get(r, -9.0):
                agg[r] = float(s)
        order = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        for j, (r2, _v) in enumerate(order):
            if r2 == g:
                ranks[i] = j + 1
                break
    return ranks


def summarize(ranks: np.ndarray) -> dict:
    hits1 = (ranks == 1).astype(float)
    hits3 = (ranks <= 3).astype(float)
    hits5 = (ranks <= 5).astype(float)
    mrr = (1.0 / np.minimum(ranks, 999)).mean()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(hits1), (1000, len(hits1)))
    lo, hi = np.percentile(hits1[idx].mean(1), [2.5, 97.5])
    return {"hit@1": round(hits1.mean(), 4), "hit@3": round(hits3.mean(), 4),
            "hit@5": round(hits5.mean(), 4), "mrr": round(float(mrr), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "se": round(float(hits1.std(ddof=1) / np.sqrt(len(hits1))), 4),
            "n": int(len(hits1)), "hits": int(hits1.sum()),
            "_hits1": hits1}


def mcnemar(a_hits: np.ndarray, b_hits: np.ndarray) -> dict:
    b = int(((a_hits == 0) & (b_hits == 1)).sum())  # a miss, b hit
    c = int(((a_hits == 1) & (b_hits == 0)).sum())  # a hit, b miss
    if b + c == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    p = binomtest(min(b, c), b + c, 0.5).pvalue
    return {"b": b, "c": c, "p": round(float(p), 4)}


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="_ab_tmp/gold200.json")
    ap.add_argument("--out", default="_ab_tmp/accept/accept.json")
    args = ap.parse_args()

    gold_path = Path(args.gold)
    h = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    sha_file = Path(str(gold_path) + ".sha256")
    if sha_file.exists():
        assert h == sha_file.read_text(encoding="utf-8").strip(), "gold 文件被改过，拒绝运行"
    g = json.loads(gold_path.read_text(encoding="utf-8"))
    print(f"[gold] {g['n']} 条 / model={g['model']} / sha256={h[:16]}…", flush=True)

    items = load_corpus()
    by_ref = {it["ref"]: it for it in items}
    golds = [x for x in g["items"] if x["ref"] in by_ref and x["query"]]
    print(f"[gold] 可用 {len(golds)} 条（语料 {len(items)} 条）", flush=True)
    refs = np.array([it["ref"] for it in items])
    gold_refs = [x["ref"] for x in golds]
    queries = [x["query"] for x in golds]

    texts = [it["text"] for it in items]
    K, Z = encode_units(texts, f"msg_{M}")
    # 查询键/值
    qtag = f"q_{M}_{h[:12]}"
    qp = CACHE / f"{qtag}.npz"
    if qp.exists():
        d = np.load(qp)
        Q, Qz = d["Q"], d["Qz"]
    else:
        enc = Encoder(m=M, proj=True, verbose=False)
        Qs, Qzs = [], []
        for q in queries:
            Qs.append(enc.key(q).detach().cpu().numpy())
            Qzs.append(enc.value(q).detach().cpu().numpy())
        Q = _norm(np.asarray(Qs, dtype=np.float32))
        Qz = _norm(np.asarray(Qzs, dtype=np.float32))
        np.savez(qp, Q=Q, Qz=Qz)
    print(f"[keys] K={K.shape} Q={Q.shape}", flush=True)

    results: dict[str, dict] = {}

    # ---- 塔：四种读出
    def ridge_readout(Km: np.ndarray, Zm: np.ndarray, Qm: np.ndarray) -> np.ndarray:
        G = Km.T @ Km + LAM * np.eye(Km.shape[1], dtype=np.float32)
        W = np.linalg.solve(G, Km.T @ Zm)
        R = Qm @ W
        R = _norm(R)
        return R @ Zm.T

    results["tower_raw_keycos"] = summarize(max_by_ref(Q @ K.T, refs, gold_refs))
    results["tower_raw_ridge"] = summarize(max_by_ref(ridge_readout(K, Z, Q), refs, gold_refs))

    Kw, Qw, sk = whiten(K, Q, "key")
    Zw, Qzw, sz = whiten(Z, Qz, "value")
    results["tower_white_keycos"] = summarize(max_by_ref(Qw @ Kw.T, refs, gold_refs))
    results["tower_white_ridge"] = summarize(max_by_ref(ridge_readout(Kw, Zw, Qw), refs, gold_refs))

    # ---- 词面基线（逐字复跑 H11：同一函数、同一单元）
    from tools_session_recall import IdfIndex, SENT_SPLIT
    from tools_tower_memory import lexical_score

    def rank_of_orders(orders: list[str], gold: str) -> int:
        idx = orders.index(gold) if gold in orders else -1
        return idx + 1 if idx >= 0 else 999

    def arm_from_pairs(name: str, unit_texts: list[str], unit_refs: list[str], sc) -> dict:
        t0 = time.time()
        ranks = []
        for q in queries:
            best: dict[str, float] = {}
            for txt, r in zip(unit_texts, unit_refs):
                v = sc(q, txt)
                if v > best.get(r, -1e9):
                    best[r] = v
            order = [r for r, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)]
            ranks.append(rank_of_orders(order, gold_refs[len(ranks)]))
        print(f"[lex] {name} {time.time() - t0:.0f}s", flush=True)
        return summarize(np.array(ranks, dtype=np.int32))

    results["bigram_msg"] = arm_from_pairs("bigram_msg", texts, list(refs), lexical_score)

    idx = IdfIndex(texts)
    print("[lex] idf 索引建好", flush=True)
    results["idf_msg"] = arm_from_pairs("idf_msg", texts, list(refs), idx.score)

    sent_units, sent_refs = [], []
    for t, r in zip(texts, refs):
        for s in SENT_SPLIT.split(t):
            s = s.strip()
            if len(s) >= 12:
                sent_units.append(s)
                sent_refs.append(r)
    sidx = IdfIndex(sent_units)
    print(f"[lex] 句子单元 {len(sent_units)}", flush=True)
    results["idf_sent"] = arm_from_pairs("idf_sent", sent_units, sent_refs, sidx.score)

    # ---- 对照与判据
    base = results["bigram_msg"]
    for name in ("tower_white_keycos", "tower_white_ridge"):
        a = results[name]
        results[name]["mcnemar_vs_bigram"] = mcnemar(a["_hits1"], base["_hits1"])
        diff = a["hit@1"] - base["hit@1"]
        results[name]["diff_vs_bigram"] = round(diff, 4)
        results[name]["verdict"] = ("不成立(≥−1SE)" if diff > -base["se"]
                                    else "封板(<−2SE)" if diff < -2 * base["se"] else "未区分")
        print(f"[judge] {name}: hit@1 {a['hit@1']:.4f} vs bigram {base['hit@1']:.4f} "
              f"(diff {diff:+.4f}, 1SE={base['se']:.4f}) -> {results[name]['verdict']}", flush=True)

    for k, v in results.items():
        v.pop("_hits1", None)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gold": {"file": str(gold_path), "sha256": h, "n": len(golds),
                                        "model": g["model"]},
                               "setup": {"m": M, "lam": LAM, "lw_shrink_key": sk,
                                         "lw_shrink_value": sz,
                                         "whiten": "Ledoit-Wolf, 只用键/值协方差（无标签, transductive）"},
                               "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n| 臂 | hit@1 | hit@3 | hit@5 | MRR | 95%CI |")
    print("| --- | --- | --- | --- | --- | --- |")
    for k, v in results.items():
        print(f"| {k} | {v['hit@1']:.4f} | {v['hit@3']:.4f} | {v['hit@5']:.4f} | "
              f"{v['mrr']:.4f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |")
    print(f"\n[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

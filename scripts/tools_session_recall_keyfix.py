"""H12b：按用户三点核对/修复 —— ① query/key 同空间 ② 中心化/去 PC1 ③ 任务损失训 W。

用户意见（m05631）：
    核对 query/key 同空间；中心化 / 去第一主成分（直接打 .622）；
    任务损失训 W（天花板由 K 决定，预期低但便宜）。

三件事都在**同一个键矩阵**上做，编码只做一次（带磁盘缓存）：

    A. 空间核对
       - 同一个 Encoder 实例、两个独立实例对同一段文本算 key，逐元素差应为 0
       - q·k_gold 与 q·k_other 的分布（若 query 与 unit 不同空间，两者都会被拉到均值附近）
       - 键均值向量的模、以及"减均值后"的余弦分布

    B. 各向异性修复（K 与 Q 用同一个变换）
       raw / center / pc1 / center_pc1 / white(收缩白化)
       每个都报：两两余弦均值（看 .622 压到多少）+ 消息级 hit@1/hit@3/MRR

    C. 任务损失训 W（K 冻结，天花板的直接测量）
       W: 256→256 线性映射，InfoNCE（26 条 gold 做候选 1343 分类），
       全量训练看能否过拟合（拟合能力），再做留一交叉验证（泛化能力）。

用法：
    venv\\Scripts\\python.exe tools_session_recall_keyfix.py --m 256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools_session_recall import load_corpus, make_queries, pick_gold, metrics
from tools_session_recall_hier import to_units
from tools_tower_memory import Encoder

CACHE = Path("_ab_tmp/hier")


def _npz(tag: str) -> Path:
    return CACHE / f"{tag}.npz"


def get_keys(units, tag: str, m: int, enc: Encoder) -> tuple[np.ndarray, np.ndarray]:
    cp = _npz(tag)
    if cp.exists():
        d = np.load(cp)
        return d["K"], d["Z"]
    ks, zs = [], []
    for u in units:
        ks.append(enc.key(u["text"]).detach().cpu().numpy())
        zs.append(enc.value(u["text"]).detach().cpu().numpy())
    K = np.asarray(ks, dtype=np.float32)
    Z = np.asarray(zs, dtype=np.float32)
    K /= np.linalg.norm(K, axis=1, keepdims=True) + 1e-9
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    np.savez(cp, K=K, Z=Z)
    return K, Z


def get_qkeys(refs, queries: dict, tag: str, enc: Encoder) -> np.ndarray:
    cp = _npz(tag)
    if cp.exists():
        return np.load(cp)["Q"]
    qs = []
    for r in refs:
        k = enc.key(queries[r]).detach().cpu().numpy()
        qs.append(k / (np.linalg.norm(k) + 1e-9))
    Q = np.asarray(qs, dtype=np.float32)
    np.savez(cp, Q=Q)
    return Q


# ---------------------------------------------------------------- 变换
def tf_center(K: np.ndarray, Q: np.ndarray):
    mu = K.mean(0, keepdims=True)
    return _norm(K - mu), _norm(Q - mu), mu


def tf_pc1(K: np.ndarray, Q: np.ndarray):
    mu = K.mean(0, keepdims=True)
    X = K - mu
    # 第一主成分（K 的中心化协方差最大特征向量）
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    v1 = vt[0:1]
    return _norm(X - (X @ v1.T) @ v1), _norm(Q - mu - ((Q - mu) @ v1.T) @ v1), v1


def tf_center_pc1(K: np.ndarray, Q: np.ndarray):
    A, B, v1 = tf_pc1(K, Q)
    return A, B, v1


def tf_white(K: np.ndarray, Q: np.ndarray, shrink: float = 0.1):
    mu = K.mean(0, keepdims=True)
    X = K - mu
    Cov = (X.T @ X) / max(1, len(X) - 1)
    Cov = (1 - shrink) * Cov + shrink * np.eye(Cov.shape[0], dtype=np.float32) * np.trace(Cov) / Cov.shape[0]
    evals, evecs = np.linalg.eigh(Cov.astype(np.float64))
    evals = np.maximum(evals, 1e-6)
    Wm = (evecs / np.sqrt(evals)) @ evecs.T  # 对称白化
    return _norm(X @ Wm), _norm((Q - mu) @ Wm), None


def _norm(A: np.ndarray) -> np.ndarray:
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------- 评测
def ranks_from_scores(scores: np.ndarray, refs: np.ndarray, gold_refs: list[str]) -> list[int]:
    """scores: (n_query, n_unit) → 每条消息取 max 后排序，取 gold 的排名。"""
    out = []
    for i, g in enumerate(gold_refs):
        best: dict[str, float] = {}
        for s, r in zip(scores[i], refs):
            if s > best.get(r, -9.0):
                best[r] = float(s)
        order = [k for k, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)]
        out.append(next((j + 1 for j, r2 in enumerate(order) if r2 == g), 999))
    return out


def mean_pair_cos(K: np.ndarray, n: int = 600, seed: int = 0) -> float:
    idx = np.random.default_rng(seed).choice(len(K), min(n, len(K)), replace=False)
    S = K[idx] @ K[idx].T
    iu = np.triu_indices(len(idx), 1)
    return float(S[iu].mean())


# ---------------------------------------------------------------- 任务损失训 W
def train_W(Kn: np.ndarray, Qn: np.ndarray, gold_idx: list[int], steps: int = 400,
            lr: float = 1e-2, temp: float = 0.05, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    Kt = torch.from_numpy(Kn)
    Qt = torch.from_numpy(Qn)
    W = torch.eye(Kn.shape[1], requires_grad=True)
    opt = torch.optim.Adam([W], lr=lr)
    y = torch.tensor(gold_idx, dtype=torch.long)
    for _ in range(steps):
        opt.zero_grad()
        logits = (Qt @ W) @ Kt.T / temp
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    return W.detach().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--units", default="msg,sent")
    ap.add_argument("--out", default="_ab_tmp/hier/keyfix.json")
    args = ap.parse_args()

    items = load_corpus()
    golds = pick_gold(items, 38)
    queries = make_queries(golds, Path("_ab_tmp/session_queries.json"))
    perm = np.random.default_rng(0).permutation(len(items))
    sub = [items[i] for i in perm[:1343]]
    refs_in = {it["ref"] for it in sub}
    gs = [g for g in golds if g["ref"] in refs_in]
    print(f"[setup] 消息 {len(sub)} / 金句 {len(gs)}", flush=True)

    enc = Encoder(m=args.m, proj=True, verbose=False)
    out: dict = {"setup": {"m": args.m, "n_units": len(sub), "n_gold": len(gs)}}

    # ---------- A. 空间核对：同一文本，两个独立实例
    probe = sub[0]["text"][:120]
    e1 = Encoder(m=args.m, proj=True, verbose=False)
    e2 = Encoder(m=args.m, proj=True, verbose=False)
    k1 = e1.key(probe).detach().cpu().numpy()
    k2 = e2.key(probe).detach().cpu().numpy()
    same = float(np.abs(k1 - k2).max())
    out["space"] = {"max_abs_diff_two_instances": same,
                    "cos_same_text": float(k1 @ k2 / (np.linalg.norm(k1) * np.linalg.norm(k2) + 1e-9))}
    print(f"[space] 两实例同文本 max|Δ| = {same:.3e}", flush=True)

    # ---------- 消息级单元
    res_all: dict = {}
    for kind in [k.strip() for k in args.units.split(",") if k.strip()]:
        units, refs = to_units(sub, kind, 50, 6)
        tag_k = f"{kind}_n1343_50_6_m{args.m}"
        K, Z = get_keys(units, tag_k, args.m, enc)
        refs_arr = np.array([u["ref"] for u in units])
        Q = get_qkeys([g["ref"] for g in gs], queries, f"q_n1343_m{args.m}", enc)
        gold_refs = [g["ref"] for g in gs]
        ref2idx = {r: i for i, r in enumerate(refs_arr)}

        # q·k_gold vs q·k_other（原始空间）
        gidx = [ref2idx[r] for r in gold_refs]
        qg = np.array([float(Q[i] @ K[j]) for i, j in enumerate(gidx)])
        qa = np.array([float(Q[i] @ K[j]) for i in range(len(gs))
                       for j in range(len(K)) if j != gidx[i]]).reshape(len(gs), -1)
        out.setdefault("cosine_raw", {})[kind] = {
            "q_gold_mean": float(qg.mean()), "q_other_mean": float(qa.mean()),
            "q_other_p95": float(np.percentile(qa, 95)), "gap": float(qg.mean() - qa.mean())}
        print(f"[cos:{kind}] q·k_gold {qg.mean():.3f} vs q·k_other {qa.mean():.3f} "
              f"(p95 {np.percentile(qa, 95):.3f})", flush=True)

        # ---------- B. 变换族
        variants = {"raw": (K, Q)}
        Kc, Qc, _ = tf_center(K, Q)
        variants["center"] = (Kc, Qc)
        Kp, Qp, _ = tf_pc1(K, Q)
        variants["pc1"] = (Kp, Qp)
        Kw, Qw, _ = tf_white(K, Q)
        variants["white"] = (Kw, Qw)
        Kw2, Qw2, _ = tf_white(Kw, Qw, shrink=0.02)
        variants["white2"] = (Kw2, Qw2)

        res_b = {}
        for name, (Ka, Qa_) in variants.items():
            scores = Qa_ @ Ka.T
            r = metrics(ranks_from_scores(scores, refs_arr, gold_refs))
            r["mean_pair_cos"] = round(mean_pair_cos(Ka), 3)
            res_b[name] = r
            print(f"[{kind}:{name:6s}] hit@1={r['hit@1']:.4f} hit@3={r['hit@3']:.4f} "
                  f"mrr={r['mrr']:.4f} paircos={r['mean_pair_cos']:.3f}", flush=True)
        res_all[kind] = res_b
    out["variants"] = res_all

    # ---------- C. 任务损失训 W（消息级，中心化空间）
    units, refs = to_units(sub, "msg", 50, 6)
    refs_arr = np.array([u["ref"] for u in units])
    K, Z = get_keys(units, f"msg_n1343_50_6_m{args.m}", args.m, enc)
    Q = get_qkeys([g["ref"] for g in gs], queries, f"q_n1343_m{args.m}", enc)
    ref2idx = {r: i for i, r in enumerate(refs_arr)}
    Kn, Qn, _ = tf_center(K, Q)
    order = [g["ref"] for g in gs]
    Kc2 = Kn
    Qc2 = Qn
    gi = [ref2idx[r] for r in order]

    Wfull = train_W(Kc2, Qc2, gi)
    sc = (Qc2 @ Wfull) @ Kc2.T
    r_full = metrics(ranks_from_scores(sc, refs_arr, order))
    train_acc = float(np.mean([1.0 if np.argmax(sc[i]) == gi[i] else 0.0 for i in range(len(gi))]))
    print(f"[W-full] hit@1={r_full['hit@1']:.4f} 训练集 top1 命中={train_acc:.3f}", flush=True)

    # 留一
    ranks_loo = []
    for i in range(len(gi)):
        tr = [j for j in range(len(gi)) if j != i]
        W = train_W(Kc2, Qc2[tr], [gi[j] for j in tr], seed=i)
        s = (Qc2[i:i + 1] @ W) @ Kc2.T
        best: dict[str, float] = {}
        for val, rf in zip(s[0], refs_arr):
            if val > best.get(rf, -9.0):
                best[rf] = float(val)
        rk = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        ranks_loo.append(next((j + 1 for j, (_r, _v) in enumerate(rk) if _r == order[i]), 999))
    r_loo = metrics(ranks_loo)
    print(f"[W-loo ] hit@1={r_loo['hit@1']:.4f} hit@3={r_loo['hit@3']:.4f} mrr={r_loo['mrr']:.4f}", flush=True)

    out["trained_W"] = {"full_fit": r_full, "train_top1": train_acc, "loo": r_loo}
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[out] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

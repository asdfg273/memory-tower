"""C 阶段对照实验：平台自身的 SQLite 检索 vs 记忆塔读出。

同一批 N 条真实记忆、同一套 ground truth，两组 query：
  - keep：保留原词的自然问法（owner 自己会这么问）
  - para：零词面重叠的改写（同样的意思，用词全换），用来检验
          "bigram 检索瞎了的地方，塔能不能捞回来"

被测系统：
  1. platform      —— chat_app 现用路径：MemoryService.get_relevant
                      （0.55*bigram + 0.20*scope + 0.15*confidence + 0.10*recency）
  2. lexical       —— 消融：只按 bigram 排序
  3. tower:gated   —— tools_tower_memory 默认融合（键 = 原文）
  4. tower:tower   —— 纯塔读出（键 = 原文）
  5. tower:rrf     —— 排名倒数融合（键 = 原文）
  6. tq:gated / tq:tower / tq:rrf —— 同上，但塔键 = 原文 + 离线生成的 4 条问句
                      （_ab_tmp/memory_questions.jsonl，由 tools_memory_questions.py 产出）

指标：hit@1 / hit@3 / MRR（可接受集合的首条命中），另报词面信号强度
（query 与目标记忆的 bigram 相似度），以及"词面信号弱"子集上的 hit@1。

用法：
    venv\\Scripts\\python.exe tools_memory_ab.py [--db data/memory.db] [--k 5]
产物：_ab_tmp/memory_ab_platform.{json,md}
（reports/ 目录当前账号无写权限，见塔记忆 "reports 目录不可写（ACL）"）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import urllib.error
import urllib.request
from typing import Dict, List, Sequence, Tuple

# (query, 可接受答案的文本片段)；片段必须能在 raw_source 里唯一定位到一条记忆
QUERIES_KEEP: Sequence[Tuple[str, Sequence[str]]] = (
    ("报告该报什么才不被批", ["不要报单点", "所有能力测试必须报覆盖", "headline 改成"]),
    ("headline 那一行改成什么", ["headline 改成"]),
    ("测试难度是构造者选的，怎么证明不是挑软柿子", ["难度由构造者", "选某一档当主线"]),
    ("合成数据上 oracle 怎么验收", ["退化 oracle", "污染上失败一次"]),
    ("报告里的数字要不要标版本", ["harness 产出的数字带版本"]),
    ("模型要不要能说不知道", ["模型必须有弃答机制", "训练集必须含", "不要造门"]),
    ("⊥ 样本怎么进训练集", ["训练集必须含", "不要造门", "模型必须有弃答机制"]),
    ("评测集和训练集怎么切", ["先冻结评测集", "改写源必须分离", "跨方法留出"]),
    ("改写源能不能和测试共用", ["改写源必须分离", "至少一种改写过程"]),
    ("留出是跨数据还是跨方法", ["跨方法留出"]),
    ("样本量 n 怎么算", ["可比子集", "判据的门槛"]),
    ("预测区间怎么写", ["预测区间"]),
    ("答题率算不算单独指标", ["强制作答率是独立端点"]),
    ("阈值要不要重新标定", ["阈值冻结"]),
    ("语料从哪拿", ["语料我拿不到"]),
    ("能不能先做个能跑的 demo", ["先让一张非空表", "不要为了跑通"]),
    ("主线选哪一档", ["选某一档当主线"]),
    ("规范化那一步能不能直接拿来打标签", ["canonicalizer"]),
    ("无锚点的区域要不要多采样", ["过采样无锚点区"]),
    ("结论的适用范围怎么写", ["T1.5 假设"]),
)

QUERIES_PARA: Sequence[Tuple[str, Sequence[str]]] = (
    ("报成绩的时候只给一个数，肯定被挑毛病", ["不要报单点"]),
    ("摘要里最关键的指标格子该填什么", ["headline 改成"]),
    ("测试的难易怎么定的，才不算自己给自己放水", ["难度由构造者"]),
    ("用现成标签跑一遍，那个上界方法得接近满分", ["退化 oracle"]),
    ("跑出来的结果要不要带上实验环境的标识", ["harness 产出的数字带版本"]),
    ("自动化标注一上线，就得先证明它会踩坑", ["污染上失败一次"]),
    ("系统要能承认不知道，还得给这个不知道专门的位置", ["模型必须有弃答机制"]),
    ("训练数据里要留一些允许说不知道的例子", ["训练集必须含"]),
    ("出变体的渠道和判分的渠道不能共用", ["改写源必须分离"]),
    ("考题要先定死，再准备复习材料", ["先冻结评测集"]),
    ("统计多少个够用的时候，那个分母取什么", ["可比子集"]),
    ("门限是先定死还是事后调，两个情形都得给结论", ["阈值冻结"]),
    ("我打算只走其中一条路，要不要声明这是没验证过的", ["选某一档当主线"]),
    ("这话能管到多大范围", ["T1.5 假设"]),
    ("别为了让流程先转起来就编输入", ["不要为了跑通"]),
    ("原始材料那块得你出手", ["语料我拿不到"]),
    ("第一步先别追效果，先跑出一行真数", ["先让一张非空表"]),
    ("必须回答的比例要单独列出来", ["强制作答率是独立端点"]),
)

GROUPS: Sequence[Tuple[str, Sequence[Tuple[str, Sequence[str]]]]] = (
    ("keep", QUERIES_KEEP),
    ("para", QUERIES_PARA),
)

SYSTEMS = ("platform", "lexical", "emb", "hyb",
           "tower:gated", "tower:tower", "tower:rrf",
           "tq:gated", "tq:tower", "tq:rrf")
WEAK_SIM = 0.10  # 低于此值算"词面信号弱"


def _tmp_path(name: str) -> str:
    """沙箱下系统临时目录可能只读，统一用工作区内的 _ab_tmp。"""
    d = os.path.join(os.getcwd(), "_ab_tmp")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def load_rewrite(path: str) -> Dict[str, str]:
    """memory_id -> 改写后的条目文本（自描述化变体）。空路径返回空表。"""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    return {str(k): str(v) for k, v in obj.items()}


def load_orig(path: str) -> Dict[str, str]:
    """memory_id -> 迁移前的原始条目（ground truth 片段只在原文上定位）。"""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    return {str(k): str(v) for k, v in obj.items()}


def load_memories(db_path: str, rewrite: Dict[str, str] | None = None) -> List[Dict[str, object]]:
    tmp = _tmp_path("memory.db")
    shutil.copyfile(db_path, tmp)
    con = sqlite3.connect(tmp)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "select memory_id,raw_source,scope,confidence,updated_at from memories "
        "where deleted_at is null order by rowid"
    ).fetchall()
    if rewrite:
        # 变体实验：改写后的条目要同时进平台库，否则平台仍在给旧文本排序
        con.executemany("update memories set raw_source=? where memory_id=?",
                        [(txt, mid) for mid, txt in rewrite.items()])
        con.commit()
    con.close()
    out: List[Dict[str, object]] = []
    for r in rows:
        d = dict(r)
        # orig 是 ground truth 定位用的原文；text 才是进场检索的文本
        d["orig"] = str(r["raw_source"])
        d["text"] = str((rewrite or {}).get(str(r["memory_id"]), r["raw_source"]))
        out.append(d)
    return out


def embed_texts(texts: Sequence[str], url: str, model: str,
                timeout: float = 120.0, batch: int = 8) -> List[List[float]]:
    """LM Studio /v1/embeddings，批量拿向量（返回顺序与入参一致）。"""
    out: List[List[float]] = []
    for i in range(0, len(texts), batch):
        chunk = [str(t) for t in texts[i:i + batch]]
        body = {"model": model, "input": chunk}
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        items = sorted(payload["data"], key=lambda it: it.get("index", 0))
        out.extend([list(it["embedding"]) for it in items])
    return out


def embed_cached(texts: Sequence[str], cache_path: str, url: str,
                 model: str) -> List[List[float]]:
    """按文本哈希缓存向量，避免每次重跑都打 embedding 接口。"""
    cache: Dict[str, List[float]] = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    keys = [hashlib.sha1(str(t).encode("utf-8")).hexdigest() for t in texts]
    todo = [(k, str(t)) for k, t in zip(keys, texts) if k not in cache]
    if todo:
        vecs = embed_texts([t for _, t in todo], url, model)
        for (k, _), v in zip(todo, vecs):
            cache[k] = v
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    return [cache[k] for k in keys]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def load_questions(path: str) -> Dict[str, List[str]]:
    """memory_id -> 生成的问句列表（缺文件时返回空，tq 系统退化成 tower）。"""
    if not os.path.exists(path):
        return {}
    out: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["memory_id"])] = [str(q) for q in (rec.get("questions") or [])]
    return out


def resolve(mems: List[Dict[str, object]], snippets: Sequence[str]) -> List[str]:
    ids: List[str] = []
    for snip in snippets:
        hit = [str(m["memory_id"]) for m in mems if snip in str(m.get("orig") or m["raw_source"])]
        if len(hit) != 1:
            raise SystemExit(f"ground truth 片段 {snip!r} 命中 {len(hit)} 条：{hit}")
        ids.append(hit[0])
    return ids


def rank_of(targets: Sequence[str], ranked: Sequence[str]) -> int:
    for pos, mid in enumerate(ranked, start=1):
        if mid in targets:
            return pos
    return 0


def summarize(ranks: Sequence[int]) -> Dict[str, float]:
    n = len(ranks) or 1
    return {
        "hit@1": round(sum(1 for r in ranks if r == 1) / n, 4),
        "hit@3": round(sum(1 for r in ranks if 1 <= r <= 3) / n, 4),
        "mrr": round(sum((1.0 / r) if r else 0.0 for r in ranks) / n, 4),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join("data", "memory.db"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=os.path.join("_ab_tmp", "memory_ab_platform.json"))
    ap.add_argument("--rewrite", default="", help="memory_id -> 改写文本的 JSON（自描述化变体）")
    ap.add_argument("--orig", default=os.path.join("data", "memory_orig.json"),
                    help="memory_id -> 迁移前原文（库里已是自描述文本时用它定位 ground truth）")
    ap.add_argument("--emb-url", default="http://127.0.0.1:1234/v1/embeddings")
    ap.add_argument("--emb-model", default="text-embedding-nomic-embed-text-v1.5")
    ap.add_argument("--emb-cache", default=os.path.join("_ab_tmp", "memory_emb.json"))
    ap.add_argument("--no-emb", action="store_true")
    args = ap.parse_args(argv)

    rewrite = load_rewrite(args.rewrite)
    mems = load_memories(args.db, rewrite)
    orig = load_orig(args.orig)
    if rewrite:
        print(f"[ab] variant=rewrite({len(rewrite)} 条)")
    elif orig:
        n = sum(1 for m in mems if str(m["memory_id"]) in orig)
        print(f"[ab] 库里 {n} 条已被自描述化，ground truth 用 {args.orig} 定位")
    if orig:
        for m in mems:
            mid = str(m["memory_id"])
            if mid in orig:
                m["orig"] = orig[mid]
    mem_vec: Dict[str, List[float]] = {}
    qvec: Dict[str, List[float]] = {}
    if not args.no_emb:
        try:
            vecs = embed_cached([str(m["text"]) for m in mems], args.emb_cache,
                                args.emb_url, args.emb_model)
            mem_vec = {str(m["memory_id"]): v for m, v in zip(mems, vecs)}
            all_q = [q for _, qs in GROUPS for q, _ in qs]
            qvec = dict(zip(all_q, embed_cached(all_q, args.emb_cache,
                                                args.emb_url, args.emb_model)))
            print(f"[ab] embeddings dim={len(next(iter(mem_vec.values())))} "
                  f"mem={len(mem_vec)} query={len(qvec)}")
        except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
            print(f"[ab] embeddings 不可用（{exc!r}），emb/hyb 记 0")

    by_text = {str(m["text"]): str(m["memory_id"]) for m in mems}
    text_of = {str(m["memory_id"]): str(m["text"]) for m in mems}
    total_q = sum(len(qs) for _, qs in GROUPS)
    print(f"[ab] memories={len(mems)} queries={total_q} k={args.k} "
          f"groups={[g for g, _ in GROUPS]}")

    from memory_service import MemoryService, text_score

    # 打开副本：data/memory.db 在本机以读写方式打开会偶发 unable to open database file
    svc = MemoryService(_tmp_path("memory.db"))
    from tools_tower_memory import TowerMemory

    tower_path = _tmp_path("tower.jsonl")
    if os.path.exists(tower_path):
        os.remove(tower_path)
    tower = TowerMemory(tower_path, verbose=False)
    for m in mems:
        tower.add(text=str(m["text"]), key=str(m["text"]))
    print(f"[ab] tower_index={tower.stats()['n']}")

    # 第二座塔：键 = 原文 + 离线生成的问句（共享同一个 Encoder 实例，省一次建模）
    qmap = load_questions(_tmp_path("memory_questions.jsonl"))
    tower_q_path = _tmp_path("tower_q.jsonl")
    if os.path.exists(tower_q_path):
        os.remove(tower_q_path)
    tower_q = TowerMemory(tower_q_path, verbose=False, enc=tower.enc)
    n_with_q = 0
    for m in mems:
        qs = qmap.get(str(m["memory_id"]), [])
        n_with_q += int(bool(qs))
        tower_q.add(text=str(m["text"]), key=str(m["text"]),
                    questions=qs, source="questions")
    n_keys_q = sum(len(r["keys"]) for r in tower_q.facts)
    print(f"[ab] tower_q_index={len(tower_q.facts)} with_questions={n_with_q}/{len(mems)} "
          f"keys={n_keys_q}")

    rows: List[Dict[str, object]] = []
    for group, queries in GROUPS:
        for query, snippets in queries:
            targets = resolve(mems, snippets)
            sim = text_score(query, text_of[targets[0]])

            plat = [r.memory_id for r in svc.get_relevant(
                user_id="owner", task_id="luent_ai", query=query, limit=args.k)]
            lex = [str(m["memory_id"]) for m in sorted(
                mems, key=lambda m: -text_score(query, str(m["text"])))]
            tower_ranks = {f: [by_text[h["text"]] for h in tower.recall(query, k=args.k, fusion=f)]
                           for f in ("gated", "tower", "rrf")}
            tq_ranks = {f: [by_text[h["text"]] for h in tower_q.recall(query, k=args.k, fusion=f)]
                        for f in ("gated", "tower", "rrf")}
            emb_rank: List[str] = []
            hyb_rank: List[str] = []
            if mem_vec:
                qv = qvec[query]
                sc = {mid: cosine(qv, v) for mid, v in mem_vec.items()}
                emb_rank = sorted(sc, key=lambda mid: -sc[mid])[:args.k]
                hs = {mid: 0.5 * sc[mid] + 0.5 * text_score(query, text_of[mid])
                      for mid in sc}
                hyb_rank = sorted(hs, key=lambda mid: -hs[mid])[:args.k]
            ranked = {
                "platform": plat,
                "lexical": lex,
                "emb": emb_rank,
                "hyb": hyb_rank,
                "tower:gated": tower_ranks["gated"],
                "tower:tower": tower_ranks["tower"],
                "tower:rrf": tower_ranks["rrf"],
                "tq:gated": tq_ranks["gated"],
                "tq:tower": tq_ranks["tower"],
                "tq:rrf": tq_ranks["rrf"],
            }
            rows.append({
                "group": group, "query": query, "targets": targets,
                "sim": round(float(sim), 4),
                "ranks": {name: rank_of(targets, seq) for name, seq in ranked.items()},
                "top1": {name: (seq[0] if seq else "") for name, seq in ranked.items()},
            })

    def table_of(sel: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
        return {name: summarize([int(r["ranks"][name]) for r in sel]) for name in SYSTEMS}

    per_group = {g: table_of([r for r in rows if r["group"] == g]) for g, _ in GROUPS}
    pooled = table_of(rows)
    weak = [r for r in rows if float(r["sim"]) < WEAK_SIM]
    weak_tbl = table_of(weak) if weak else {}
    sim_mean = {g: round(sum(float(r["sim"]) for r in rows if r["group"] == g) /
                         max(1, len([1 for r in rows if r["group"] == g])), 4)
                for g, _ in GROUPS}
    top1_same = sum(1 for r in rows if r["top1"]["platform"] == r["top1"]["lexical"])  # type: ignore[dict-item]

    print("\n[ab] ==== pooled ====")
    for name, m in pooled.items():
        print(f"  {name:<12} hit@1={m['hit@1']:.3f} hit@3={m['hit@3']:.3f} mrr={m['mrr']:.3f}")
    for g, _ in GROUPS:
        m = per_group[g]["tower:tower"]
        print(f"[ab] group={g} lexical_sim={sim_mean[g]:.3f} "
              f"tower_hit1={m['hit@1']:.3f} lex_hit1={per_group[g]['lexical']['hit@1']:.3f}")
    if weak:
        print(f"[ab] weak-lexical subset (sim<{WEAK_SIM}): n={len(weak)}")
        for name, m in weak_tbl.items():
            print(f"  {name:<12} hit@1={m['hit@1']:.3f} mrr={m['mrr']:.3f}")
    print(f"[ab] platform==lexical top1: {top1_same}/{len(rows)}")

    # 论文数字的唯一来源：这一行是机器产出，抄进 reports/repro/ 时不要手改
    summary = {
        "n_memories": len(mems), "n_queries": len(rows), "k": args.k,
        "pooled": pooled, "per_group": per_group, "lexical_sim_mean": sim_mean,
        "weak_subset": {"threshold": WEAK_SIM, "n": len(weak), "table": weak_tbl},
        "top1_platform_eq_lexical": top1_same,
    }
    print("[ab] summary_json=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "db": os.path.abspath(args.db), "n_memories": len(mems), "n_queries": len(rows),
        "k": args.k, "pooled": pooled, "per_group": per_group,
        "tower_q": {"with_questions": n_with_q, "n_keys": n_keys_q},
        "lexical_sim_mean": sim_mean, "weak_subset": {"threshold": WEAK_SIM,
                                                      "n": len(weak), "table": weak_tbl},
        "top1_platform_eq_lexical": top1_same, "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    md = ["# 平台记忆检索：SQLite vs 塔（C 阶段对照）", "",
          f"- 记忆 {len(mems)} 条（全部 `scope=task`、`confidence=0.9`、同一时间戳）",
          f"- query {len(rows)} 条（keep {len(QUERIES_KEEP)} / para {len(QUERIES_PARA)}），"
          f"k={args.k}，指标为可接受集合的首条命中", "",
          f"- platform 与纯词面 top-1 一致：**{top1_same}/{len(rows)}**",
          f"- 塔键：原文 = 1 键/条；原文+问句 = {n_keys_q / max(1, len(mems)):.1f} 键/条"
          f"（{n_with_q}/{len(mems)} 条有生成问句）", "",
          "## 汇总（全部 query）", "", "| 系统 | hit@1 | hit@3 | MRR |", "| --- | --- | --- | --- |"]
    for name, m in pooled.items():
        md.append(f"| {name} | {m['hit@1']:.3f} | {m['hit@3']:.3f} | {m['mrr']:.3f} |")
    for g, qs in GROUPS:
        md += ["", f"## 分组：{g}（{len(qs)} 条，目标词面相似度均值 {sim_mean[g]:.3f}）", "",
               "| 系统 | hit@1 | hit@3 | MRR |", "| --- | --- | --- | --- |"]
        for name, m in per_group[g].items():
            md.append(f"| {name} | {m['hit@1']:.3f} | {m['hit@3']:.3f} | {m['mrr']:.3f} |")
    if weak:
        md += ["", f"## 词面信号弱子集（sim<{WEAK_SIM}，n={len(weak)}）", "",
               "| 系统 | hit@1 | MRR |", "| --- | --- | --- |"]
        for name, m in weak_tbl.items():
            md.append(f"| {name} | {m['hit@1']:.3f} | {m['mrr']:.3f} |")
    md += ["", "## 逐条（数字=首条正确记忆的排名，0=前 k 名内没有）", "",
           "| 组 | query | sim | platform | lexical | emb | hyb | t:gated | t:tower | t:rrf |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        rk = r["ranks"]
        md.append(f"| {r['group']} | {r['query']} | {r['sim']:.3f} | {rk['platform']} | "
                  f"{rk['lexical']} | {rk['emb']} | {rk['hyb']} | "
                  f"{rk['tower:gated']} | {rk['tower:tower']} | {rk['tower:rrf']} |")
    md_path = os.path.splitext(args.out)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[ab] wrote {args.out} and {md_path}")

    svc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

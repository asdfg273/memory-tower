r"""
tools_tower_vs_rag.py —— A#3：闭式运行时下的「塔 vs RAG」端到端对照
================================================================================

T98 的对照用的是**梯度写**运行时（每条 400+3x20 步），所以只能做到 n=80/300。
现在写入换成闭式（`--write closed`），把同一套对照推到 n=500，并回答：

  1 换了写入方式，塔的检索命中率会不会掉？（对照：同一格的 grad 写）
  2 与 RAG（同编码器的密集 top-1 检索）比，塔在哪里赢、在哪里输？
  3 "自选键"是否仍然是对 RAG 的唯一结构性优势（改写成描述式问句后）？

四种机制（与 T98 同名）
    dict         精确子串匹配（**平凡解，上界**）：问句里出现完整实体串就命中
    RAG          同编码器密集 top-1 检索：问句嵌入 vs 每条事实的问句嵌入
    tower-grad   塔读出 -> 最近邻存储码 -> 查表（原交付路径）
    tower-closed 塔读出 -> 最近邻存储码 -> 查表（**闭式写**，本次新增）

两种问法
    verbatim  「箱子XY的密码是什么？」            （实体串原样出现 -> dict 可用）
    described 「X和Y的箱子，密码是什么？」        （实体串被打断 -> dict 必失；塔用自选键重建 XY）

跑法
----
    $env:HF_HUB_OFFLINE="1"
    venv\Scripts\python.exe tools_tower_vs_rag.py --ns 80,300,500 --out reports/repro/tower_vs_rag.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from tower_runtime import (_E_POOL, _E_POOL_EXT2, _N_POOL, _N_POOL_EXT, _N_POOL_EXT2,
                           TowerRuntime)

Q_VERB = "箱子{q}的密码是什么？只回答密码本身。"
Q_DESC = "{a}和{b}的箱子，密码是什么？只回答密码本身。"
_E30 = "龘靐齾爩麤龗灪龖厵滟癵驫麣纞虋讟钃鸜麷鞻韽韾顟顠饙饡騳髾"


def build_facts(n: int, name_mode: str = "single",
                entity_mode: str = "pair") -> Tuple[List[Tuple[str, str, str, str]], List[str]]:
    """返回 facts: (实体, a, b, 名称) 与名称池。

    name_mode="single"：名称用**单 token** 字（A#1 表明一个槽只能解出一个 token，
        多字名称会被均值池化压成不可逆投影，塔这一臂会不公平地吃亏）；
    name_mode="multi" ：名称用两字组合，专门用来暴露"长值"瓶颈。
    entity_mode="pair"：实体是两字组合（描述式问句 "X和Y的箱子" 才有意义），
        但键 = 两个字嵌入之和 ⇒ 秩 ≤ 字符表大小（加法结构，见 T132 的绑定分析）；
    entity_mode="single"：实体是单字 ⇒ 键秩 = n，是塔的**公平**工况。
    """
    chars = list(dict.fromkeys(_E30))
    tpl = set("箱子密码是什么只回答本身电话地址，。？")
    if entity_mode == "single":
        epool = list(dict.fromkeys(list(_E_POOL) + list(_E_POOL_EXT2)))
        ents = [c for c in epool if c not in set(chars) and c not in tpl][:n]
        combos = ents
    elif entity_mode == "pair_big":
        # 400 字的两字组合：两把互素步长，避免行优先枚举带来的首字共线。
        # 注意键 = 两字嵌入之和 ⇒ 秩 ≤ 用到的不同字数（≈min(n,400)），这是加法结构的硬上限。
        pool = [c for c in list(_E_POOL_EXT2) + list(_E_POOL) if c not in tpl]
        pool = list(dict.fromkeys(pool))
        na, nb = len(pool), len(pool)
        sa, sb = 37, 101
        while math.gcd(sa, na) != 1:
            sa += 1
        while math.gcd(sb, nb) != 1:
            sb += 1
        combos = [pool[(i * sa) % na] + pool[(i * sb) % nb] for i in range(n)]
    else:
        ents = []
        combos = [x + y for x in chars for y in chars][:n]
    if name_mode == "single":
        pool = [c for c in list(_N_POOL) + list(_N_POOL_EXT2) + list(_N_POOL_EXT)
                if c not in set(chars) and c not in set(combos)]
        names = list(dict.fromkeys(pool))[:n]
    else:
        names = list(dict.fromkeys(_N_POOL)) + [x + y for x in dict.fromkeys(_N_POOL_EXT)
                                                for y in dict.fromkeys(_N_POOL_EXT)]
        names = names[:n]
    assert len(combos) == n and len(names) == n, f"池不够 n={n}"
    assert not (set(names) & set(combos)), "名称与实体字重叠"
    facts = [(c, c[0], c[-1], names[i]) for i, c in enumerate(combos)]
    return facts, names


def key_from_desc(question: str) -> str:
    """自选键：从描述式问句重建实体串。"""
    import re
    mm = re.match(r"(.+?)和(.+?)的箱子", question)
    if mm:
        return mm.group(1) + mm.group(2)
    return question


def run_one(n: int, write: str, m: int, name_mode: str = "single",
            entity_mode: str = "pair") -> Dict[str, object]:
    facts, _ = build_facts(n, name_mode, entity_mode)
    tc0 = time.time()
    rt = TowerRuntime(m=m, write_mode=write, verbose=False).build()
    t_build = time.time() - tc0

    # --- 两种问句
    qv = [(Q_VERB.format(q=e), nm) for e, _, _, nm in facts]
    qd = [(Q_DESC.format(a=a, b=b), nm) for _, a, b, nm in facts]

    res: Dict[str, object] = {"n": n, "write": write, "m": m, "name_mode": name_mode,
                              "entity_mode": entity_mode, "build_s": round(t_build, 1)}

    # --- dict（平凡解）
    def dict_hit(q: str) -> str:
        for e, _, _, nm in facts:
            if e in q:
                return nm
        return ""
    res["dict_verb"] = round(sum(dict_hit(q) == nm for q, nm in qv) / n, 4)
    res["dict_desc"] = round(sum(dict_hit(q) == nm for q, nm in qd) / n, 4)

    # --- RAG：问句嵌入 vs 每条事实的 verbatim 问句嵌入
    with torch.no_grad():
        docv = torch.stack([rt._embed(q) for q, _ in qv], 0)          # (n,d)
        docv = F.normalize(docv, dim=-1)
        for tag, qs in (("rag_verb", qv), ("rag_desc", qd)):
            hit = 0
            for q, nm in qs:
                s = F.normalize(rt._embed(q), dim=-1) @ docv.t()
                hit += int(facts[int(s.argmax())][3] == nm)
            res[tag] = round(hit / n, 4)

    # --- 塔：写入 + 读出
    rt.teach_many(qv)
    t0 = time.time(); rt.commit(); res["commit_s"] = round(time.time() - t0, 2)
    res["commit_ms_per_fact"] = round(res["commit_s"] * 1e3 / n, 2)

    for tag, qs in (("tower_verb", qv), ("tower_desc", qd)):
        hit, addr_ok = 0, 0
        for q, nm in qs:
            p = rt.probe(q, "dict")
            hit += int(p["answer"] == nm)
            addr_ok += int(float(p["addr"]) >= rt.addr_thr)   # type: ignore[arg-type]
        res[tag] = round(hit / n, 4)
        res[tag + "_addr_pass"] = round(addr_ok / n, 4)

    # --- described 形态用自选键再跑一遍（同一段记忆，只换键）
    if entity_mode == "pair":
        rt2 = TowerRuntime(m=m, write_mode=write, verbose=False, key_fn=key_from_desc).build()
        rt2.teach_many(qd)
        t0 = time.time(); rt2.commit(); res["commit_s_desc"] = round(time.time() - t0, 2)
        hit = sum(int(rt2.probe(q, "dict")["answer"] == nm) for q, nm in qd)
        res["tower_desc_selfkey"] = round(hit / n, 4)
        # 用 verbatim 问句去查自选键写成的记忆（跨形态）
        hit = sum(int(rt2.probe(q, "dict")["answer"] == nm) for q, nm in qv)
        res["tower_desc_selfkey_crossverb"] = round(hit / n, 4)

    tail = "" if entity_mode != "pair" else f" · 自选键 {res['tower_desc_selfkey']:.3f}"
    print(f"  n={n:<4} write={write:7s} m={m:<5d} 实体={entity_mode:6s}"
          f" dict {res['dict_verb']:.3f} · RAG {res['rag_verb']:.3f} · "
          f"tower {res['tower_verb']:.3f}{tail} · commit {res['commit_s']:.2f}s", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="80,300,500")
    ap.add_argument("--writes", default="grad,closed")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--name-mode", choices=("single", "multi"), default="single",
                    help="single=单 token 名称（公平对照）；multi=两字名称（暴露长值瓶颈）")
    ap.add_argument("--entity-mode", choices=("single", "pair", "pair_big"), default="single",
                    help="single=单字实体（键满秩，公平）；pair=两字实体（加法键，秩≤字符表）")
    ap.add_argument("--out", default="reports/repro/tower_vs_rag.json")
    args = ap.parse_args()
    rows = []
    t0 = time.time()
    for n in [int(x) for x in args.ns.split(",")]:
        for w in args.writes.split(","):
            rows.append(run_one(n, w, args.m, args.name_mode, args.entity_mode))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"rows": rows, "wall_s": round(time.time() - t0, 1)},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[done] {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()

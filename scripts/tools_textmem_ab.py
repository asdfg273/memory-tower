"""文本侧记忆选择实验：固定 token 预算下，A/B/C/D/E 五臂对照。

问题不是"能不能注入"（文本场景通道免费），而是**在固定预算下该放哪几条进去**。

臂：
  A_full     全历史进上下文（天花板，超预算时从最早处截断）
  B_tower    塔检索 + 注入 k 条（本项目方案）
  C_summary  滚动摘要（LLM 写，压到同等预算）
  D_vec      向量检索原始历史片段（nomic 余弦，工程默认解）
  E_none     无记忆（地板）

判据（先写死）：
  若 B 在同等预算下不超过 max(C, D)，则"塔在文本场景"判定为死，转负结果写作。

评测修正（都在本脚本里）：
  1) 关键实体判据：答案必须包含目标实体（人名/数字/专名），不是"像不像"。
  2) lag 分层：目标事实分别放在问句前 1/10/100/1000 轮。
  3) 干扰记忆：每个目标配 3 条相似干扰（同句式、实体不同），窄化检索命中。

用法：
  venv\\Scripts\\python.exe tools_textmem_ab.py --lags 1,10,100 --budget-chars 600
  venv\\Scripts\\python.exe tools_textmem_ab.py --lags 1000 --budget-chars 600 --out reports\\repro\\textmem_ab_1k.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import urllib.error
import urllib.request
from typing import Dict, List, Sequence, Tuple

TMP = os.path.join("_ab_tmp", "textmem")
BASE = "http://127.0.0.1:1234/v1"
MODEL = "qwen2.5-vl-7b"        # 非思考模型：qwen3-14b 会把预算全花在 reasoning_content 上
EMB_MODEL = "text-embedding-nomic-embed-text-v1.5"

# 目标事实：(提问, 目标实体, 陈述, 干扰陈述列表)
# v2：问题一律改写成不带原句措辞的版本；每条目标配 5 条高词面重叠的干扰
# （同一实体类型、同一话题、值不同），逼检索器靠“归属/更新”而不是靠词面匹配取胜。
TARGETS: List[Tuple[str, str, str, List[str]]] = [
    ("咱家那只橘猫叫什么？", "豆豆",
     "我家的猫叫豆豆，是只橘猫。",
     ["小区里那只流浪橘猫叫大橘。", "我表姐家的布偶猫叫雪球。", "同事小王养的猫叫毛毛。",
      "楼下便利店养了只橘猫。", "我朋友说她以后养的猫也要叫豆豆。"]),
    ("我今天车停哪儿了？", "B2",
     "我的车停在 B2 层 37 号车位。",
     ["老张的车停在 B1 层 12 号。", "上周我停错在 C3 层。", "访客车位在 A 区，只有 40 个。",
      "我同事的车是 B2 层 38 号。", "B2 层的充电桩坏了两个。"]),
    ("我的工号是多少？", "A7391",
     "我的工号是 A7391，入职两年了。",
     ["新来的实习生工号是 A7402。", "财务的工号前缀是 F 开头。", "工号 A7311 那位已经离职。",
      "上家公司我的工号是 A7391 吗？记不清了。", "面签时给过一个临时编号 T5510。"]),
    ("点菜的时候我要避开什么？", "花生",
     "我对花生过敏，点菜要特别说。",
     ["李工对海鲜过敏。", "公司食堂每周三有花生炖猪手。", "我妈对芒果过敏。",
      "花生油也算花生。", "小赵说坚果类他都不能吃。"]),
    ("下周三我去哪儿？", "杭州",
     "我下周三要去杭州出差，待两天。",
     ["上个月去了南京出差。", "老赵下周去成都。", "杭州那边的对接人姓周。",
      "下周二的评审会取消了。", "销售部分拆去了苏州。"]),
    # v3 扩充到 12 条：每条都有 5 条同话题干扰，问题不带原句措辞
    ("我租的房子在哪个小区？", "翠湖",
     "我租的房子在翠湖小区 3 号楼。",
     ["公司附近还有个碧水小区。", "老张住在翠湖对面。", "我看过金桥的房子，太贵。",
      "翠湖小区门口在修路。", "房东说她另一套在锦绣苑。"]),
    ("医保卡放哪了？", "抽屉", "医保卡放在书房第二层抽屉里。",
     ["身份证在钱包里。", "钥匙挂在门口挂钩。", "书房的书架有三层。",
      "第二层抽屉里还有充电线。", "我妈把她的医保卡放在包里。"]),
    ("我用的什么手机？", "iPhone", "我用的手机是 iPhone 15，蓝色的。",
     ["同事小李用华为。", "公司发的测试机是安卓。", "蓝色那台是去年的旧机型。",
      "iPhone 的电池最近不太行。", "我对象用的是 iPhone 13。"]),
    ("我每周几打羽毛球？", "周四", "我每周四晚上七点打羽毛球。",
     ["球馆在体育馆二楼。", "老周每周三打球。", "上周四下雨没去成。",
      "羽毛球拍该换线了。", "公司羽毛球群有 20 多个人。"]),
    ("我的牙医姓什么？", "陈", "我的牙医姓陈，在城西分院。",
     ["城东分院有位姓李的医生。", "陈医生每周只坐诊三天。", "洗牙大概两百块。",
      "我上次约的是周三下午。", "同事的牙医也姓陈。"]),
    ("公司 Wi-Fi 密码是什么？", "qd2024", "公司访客 Wi-Fi 的密码是 qd2024。",
     ["办公网要单独申请账号。", "qd2024 每季度换一次。", "茶水间的 Wi-Fi 信号差。",
      "家里的 Wi-Fi 密码我贴在路由器上。", "行政说访客密码不能外传。"]),
    ("我上次体检是什么时候？", "三月", "我上次体检是今年三月，血脂偏高。",
     ["老赵体检在五月。", "三月那次抽了四管血。", "血脂偏高要少吃油炸。",
      "下次体检约在九月。", "公司体检套餐今年涨价了。"]),
]

FILLER = [
    "今天天气还行，不冷不热的。", "刚吃完饭，有点困。", "这个需求又改了一版。",
    "周会开了一个半小时。", "地铁今天又晚点了。", "代码跑通了，但是慢。",
    "我把那个脚本挪到 tools_ 前缀了。", "中午吃了食堂的面，还行。",
    "打印机又卡纸了。", "空调太冷，我穿了外套。", "下午要跟产品对一下口径。",
    "昨天熬夜改报告，眼睛疼。", "这周的 CI 全绿。", "楼下咖啡店换了豆子。",
]


def http_json(url: str, payload: Dict, timeout: int = 180) -> Dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat(messages: List[Dict[str, str]], model: str = MODEL, temperature: float = 0.0,
         max_tokens: int = 64) -> str:
    body = http_json(f"{BASE}/chat/completions", {
        "model": model, "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens,
    })
    return (body["choices"][0]["message"].get("content") or "").strip()


def embed(texts: Sequence[str]) -> List[List[float]]:
    body = http_json(f"{BASE}/embeddings", {"model": EMB_MODEL, "input": list(texts)})
    return [d["embedding"] for d in body["data"]]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb + 1e-12)


def build_history(lag: int, seed: int = 7) -> List[Dict[str, str]]:
    """构造一条 lag 轮之前的对话流：最新一条目标事实说在「倒数第 lag 轮」，其余更早。

    事实分散在整段历史里（t_i = i*step），不再挤在开头——否则 12 条事实只有 6 个
    可用轮位，后面的会被 set 静默吞掉（v5 就是这么废掉的）。
    """
    rnd = random.Random(seed)
    n = len(TARGETS)
    total = max(lag + 6, n + 4)          # 历史长度（轮），lag 越大历史越长
    span = max(n, total // 2)            # 12 条事实均匀铺在历史的前半段
    slots: List[int] = [min(int(round(span * i / (n - 1))), total - 2) for i in range(n)]
    assert len(set(slots)) == n, f"轮位冲突: {slots}"
    slot_set = set(slots)
    turns: List[Dict[str, str]] = []
    fact_turns: List[Tuple[int, int]] = []      # (turn_index, target_index)

    for t in range(total):
        utt = rnd.choice(FILLER)
        turns.append({"role": "user", "content": utt})
        turns.append({"role": "assistant", "content": "嗯，" + rnd.choice(FILLER)})
        if t in slot_set:
            idx = slots.index(t)
            fact_turns.append((len(turns), idx))
            turns.append({"role": "user", "content": TARGETS[idx][2]})
            for d in TARGETS[idx][3]:
                turns.append({"role": "user", "content": d})
            turns.append({"role": "assistant", "content": "记下了。"})
    return turns


def truncate_text(texts: Sequence[str], budget_chars: int) -> str:
    out, used = [], 0
    for t in texts:
        if used + len(t) > budget_chars:
            break
        out.append(t)
        used += len(t)
    return "\n".join(out)


def arm_full(turns: List[Dict[str, str]], budget_chars: int) -> str:
    """天花板：把整段历史塞进去；超了就从最早处截断（保留最近的部分）。"""
    lines = [f"{t['role']}: {t['content']}" for t in turns]
    cap = budget_chars * 8
    out, used = [], 0
    for t in reversed(lines):
        if used + len(t) > cap:
            break
        out.append(t)
        used += len(t)
    return "\n".join(reversed(out))


def build_tower(facts: List[str]):
    from tools_tower_memory import TowerMemory

    path = os.path.join(TMP, "tower_ab.jsonl")
    if os.path.exists(path):
        os.remove(path)
    tower = TowerMemory(path, verbose=False)
    for f in facts:
        tower.add(text=f, key=f[:24], source="textmem")
    return tower


def arm_tower(tower, question: str, k: int, budget_chars: int) -> str:
    hits = tower.recall(question, k=k)
    return truncate_text([h["text"] for h in hits], budget_chars)


def arm_vec(chunks: List[str], question: str, k: int, budget_chars: int,
            cache: Dict[str, List[float]]) -> str:
    if "_chunks" not in cache:
        cache["_chunks"] = embed(chunks)          # type: ignore[assignment]
    if "_q" not in cache:
        cache["_q"] = embed([question])[0]        # type: ignore[assignment]
    scores = [cosine(cache["_q"], e) for e in cache["_chunks"]]   # type: ignore[arg-type]
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:k]
    return truncate_text([chunks[i] for i in order], budget_chars)


def arm_idf(chunks: List[str], question: str, k: int, budget_chars: int,
            cache: Dict[str, object]) -> str:
    """真正的 RAG 基线：idf 加权 1+2 元（就是产品 memory_service.text_score_idf 的打分器）。"""
    import math

    from memory_service import _term_freq, text_score_idf

    if "_idf" not in cache:
        n = max(1, len(chunks))
        idf1: Dict[str, float] = {}
        idf2: Dict[str, float] = {}
        for c in chunks:
            for g in _term_freq(c, 1):
                idf1[g] = idf1.get(g, 0.0) + 1.0
            for g in _term_freq(c, 2):
                idf2[g] = idf2.get(g, 0.0) + 1.0
        cache["_idf"] = ({g: math.log((n + 1) / (c + 0.5)) for g, c in idf1.items()},
                         {g: math.log((n + 1) / (c + 0.5)) for g, c in idf2.items()})
    idf1, idf2 = cache["_idf"]                     # type: ignore[misc]
    scored = [(text_score_idf(question, c, idf1, idf2), i) for i, c in enumerate(chunks)]
    order = [i for _, i in sorted(scored, key=lambda x: -x[0])[:k]]
    return truncate_text([chunks[i] for i in order], budget_chars)


def arm_summary(turns: List[Dict[str, str]], budget_chars: int,
                cache: Dict[str, str], chunk: int = 60) -> str:
    """滚动摘要：每 chunk 行把'已有摘要 + 新片段'压成新摘要，摘要上限与其它臂同预算。"""
    lines = [f"{t['role']}: {t['content']}" for t in turns]
    if "_sum" not in cache:
        summary = ""
        for i in range(0, len(lines), chunk):
            piece = "\n".join(lines[i:i + chunk])
            summary = chat([
                {"role": "system", "content": f"把下面的对话压缩进不超过 {budget_chars} 字的事实摘要，"
                                              f"保留人名、数字、专名、时间、地点等硬信息，丢掉寒暄。"},
                {"role": "user", "content": f"已有摘要：{summary or '（空）'}\n\n新增片段：\n{piece}"},
            ], max_tokens=budget_chars * 2)
        cache["_sum"] = summary
    return cache["_sum"][:budget_chars]


def ask(question: str, memory_text: str, arm: str) -> str:
    if memory_text:
        sys_prompt = ("你是一个记住用户信息的中文助手。下面是你记得的内容，"
                      "只能依据它回答；如果没有相关信息，就回答'不知道'。\n\n" + memory_text)
    else:
        sys_prompt = "你是一个中文助手。如果不知道，就回答'不知道'。"
    return chat([{"role": "system", "content": sys_prompt},
                 {"role": "user", "content": question}], max_tokens=48)


def score(answer: str, entity: str) -> Dict[str, object]:
    hit = entity in answer
    refuse = ("不知道" in answer) or ("没" in answer and "记住" in answer) or ("无法" in answer)
    return {"hit": bool(hit), "refuse": bool(refuse), "answer": answer}


def run(lags: Sequence[int], budget_chars: int, k: int) -> Dict[str, object]:
    os.makedirs(TMP, exist_ok=True)
    res: Dict[str, object] = {"budget_chars": budget_chars, "k": k, "model": MODEL,
                              "by_lag": {}}
    for lag in lags:
        turns = build_history(lag)
        lines = [f"{t['role']}: {t['content']}" for t in turns]
        chunk_size = 10
        chunks = ["\n".join(lines[i:i + chunk_size]) for i in range(0, len(lines), chunk_size)]
        cache: Dict[str, object] = {}
        # 公平起见：B 存的是与 C/D 相同的输入（整段历史里的用户话），不是只有目标事实
        user_texts = [t["content"] for t in turns if t["role"] == "user"]
        sentences = user_texts                        # 句子级 RAG 用的存储粒度
        tower = build_tower(user_texts)
        per_arm: Dict[str, List[Dict[str, object]]] = {a: [] for a in
                                                       ("A_full", "B_tower", "C_summary", "D_vec", "F_idf", "G_sent", "E_none")}
        for qi, (question, entity, _stmt, _distr) in enumerate(TARGETS):
            mem_a = arm_full(turns, budget_chars)
            mem_b = arm_tower(tower, question, k, budget_chars)
            mem_c = arm_summary(turns, budget_chars, cache)          # type: ignore[arg-type]
            mem_d = arm_vec(chunks, question, k, budget_chars, cache)
            mem_f = arm_idf(chunks, question, k, budget_chars, cache)
            mem_g = arm_idf(sentences, question, k, budget_chars, {})   # 句子级 RAG（不计缓存，口径更公平）
            for arm, mem in (("A_full", mem_a), ("B_tower", mem_b),
                             ("C_summary", mem_c), ("D_vec", mem_d),
                             ("F_idf", mem_f), ("G_sent", mem_g), ("E_none", "")):
                ans = ask(question, mem, arm)
                row = score(ans, entity)
                row.update({"q": question, "target": entity, "ctx_chars": len(mem)})
                per_arm[arm].append(row)
                print(f"[lag {lag}] {arm:10s} hit={row['hit']} ctx={len(mem):4d} "
                      f"q={question} -> {ans[:40]}", flush=True)
        res["by_lag"][str(lag)] = {
            arm: {"hit@1": round(sum(1 for r in rows if r["hit"]) / len(rows), 3),
                  "mean_ctx_chars": round(sum(r["ctx_chars"] for r in rows) / len(rows), 1),
                  "rows": rows}
            for arm, rows in per_arm.items()
        }
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lags", default="1,10,100")
    ap.add_argument("--budget-chars", type=int, default=600)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(TMP, "textmem_ab.json"))
    args = ap.parse_args(argv)
    lags = [int(x) for x in args.lags.split(",")]
    res = run(lags, args.budget_chars, args.k)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print("\n[summary] hit@1 by lag / arm")
    arm_ids = ("A_full", "B_tower", "C_summary", "D_vec", "F_idf", "G_sent", "E_none")
    print(f"{'lag':>6} " + " ".join(f"{a:>10s}" for a in arm_ids))
    for lag in lags:
        d = res["by_lag"][str(lag)]
        print(f"{lag:>6d} " + " ".join(f"{d[a]['hit@1']:>10.3f}" for a in arm_ids))
    for lag in lags:
        d = res["by_lag"][str(lag)]
        b = d["B_tower"]["hit@1"]
        best = max(d["C_summary"]["hit@1"], d["D_vec"]["hit@1"],
                   d["F_idf"]["hit@1"], d["G_sent"]["hit@1"])
        print(f"[H10 lag={lag}] B={b:.3f} best_rag/summary={best:.3f} -> "
              f"{'塔存活' if b > best else '塔判死'}", flush=True)
    print("[done] ->", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

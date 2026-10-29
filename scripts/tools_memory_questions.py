"""A′ 第一步：给每条记忆离线生成 3–5 条"用户会怎么问它"，作为塔的检索键。

背景：C 阶段测得，在零词面重叠的改写（bigram 相似度 ≈ 0.01）上，所有检索通路
都崩到 hit@1 ≈ 0.11；原因是记忆条目是裸句，塔的键就是原文，没有独立语义通道。
塔设计当初的前提是"一条事实挂多个键"，这个脚本负责把那批键造出来。

产物 data/memory_questions.jsonl，一行一条：
    {"memory_id": "...", "text": "...", "questions": ["...", "..."]}

用法：
    venv\\Scripts\\python.exe tools_memory_questions.py [--model qwen2.5-vl-7b] [--n 4]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from typing import List, Sequence

API = "http://127.0.0.1:1234/v1/chat/completions"

PROMPT = """你在给一个长期记忆库打"检索键"。下面是一条项目规则：

{text}

请写出 {n} 个工程师在真的需要用到这条规则时、会脱口而出或敲进搜索框的问句。
要求：
- 问的是"该怎么做 / 要不要做 / 有没有硬规矩"，不是问某个具体数值；
- 口语化，每条 6–18 个字；
- {n} 条彼此用词不同，模拟不同的人、不同的说法；
- 不要照抄原句里的名词组合，换一种说法；
- 只输出一个 JSON 字符串数组，不要解释、不要代码块。

示例（示例里的规则和上面这条无关，只示范风格）：
规则：接口必须带版本号
输出：["新接口怎么命名才规范", "以后字段能直接改吗", "上线后老调用方怎么兼容", "接口改动要不要跟版本走"]
规则：日志里不能出现用户手机号
输出：["打日志要注意什么", "用户隐私字段能进日志吗", "日志脱敏怎么做", "排查问题需要原始号码怎么办"]"""


def ask(model: str, text: str, n: int, timeout: float) -> List[str]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你只输出 JSON 字符串数组。"},
            {"role": "user", "content": PROMPT.format(text=text, n=n)},
        ],
        "temperature": 0.7,
        "max_tokens": 220,
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    return parse(content, n)


def parse(content: str, n: int) -> List[str]:
    text = content.strip()
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                out = [str(x).strip().strip("\"'") for x in arr if str(x).strip()]
                if out:
                    return out[:n]
        except json.JSONDecodeError:
            pass
    # 兜底：按行/顿号切，去掉序号
    parts = re.split(r"[\n、]+", text)
    out = []
    for p in parts:
        p = re.sub(r"^\s*[\-\d\.\)、]+\s*", "", p).strip().strip("\"'[]")
        if p and not p.lower().startswith("json"):
            out.append(p)
    return out[:n]


def load_memories(db_path: str) -> List[dict]:
    uri = "file:" + os.path.abspath(db_path).replace("\\", "/") + "?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "select memory_id,raw_source from memories where deleted_at is null order by rowid"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join("data", "memory.db"))
    ap.add_argument("--out", default=os.path.join("data", "memory_questions.jsonl"))
    ap.add_argument("--model", default="qwen2.5-vl-7b")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0=全部")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    mems = load_memories(args.db)
    if args.limit:
        mems = mems[:args.limit]
    print(f"[q] memories={len(mems)} model={args.model} n={args.n}", flush=True)

    out_path = args.out
    t0 = time.time()
    done = 0
    try:
        fh = open(out_path, "w", encoding="utf-8")
    except PermissionError:
        out_path = os.path.join("_ab_tmp", os.path.basename(args.out))
        os.makedirs("_ab_tmp", exist_ok=True)
        fh = open(out_path, "w", encoding="utf-8")
        print(f"[q] {args.out} 不可写，改写到 {out_path}", flush=True)

    from concurrent.futures import ThreadPoolExecutor

    def run_one(mem: dict) -> tuple:
        mid, text = str(mem["memory_id"]), str(mem["raw_source"])
        try:
            return mid, text, ask(args.model, text, args.n, args.timeout)
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as exc:
            print(f"[q] {mid} 失败：{exc!r}", flush=True)
            return mid, text, []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool, fh:
        for i, (mid, text, qs) in enumerate(pool.map(run_one, mems), 1):
            fh.write(json.dumps({"memory_id": mid, "text": text, "questions": qs},
                                ensure_ascii=False) + "\n")
            fh.flush()
            done += 1
            if qs:
                print(f"[q] {i}/{len(mems)} {mid} -> {qs[0]}", flush=True)
            else:
                print(f"[q] {i}/{len(mems)} {mid} -> (空)", flush=True)

    print(f"[q] done {done}/{len(mems)} in {time.time() - t0:.1f}s -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

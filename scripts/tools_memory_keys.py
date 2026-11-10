"""把"用户可能会怎么问"写进 memories.keys 列（产品改动的数据侧）。

问法来源：tools_memory_questions.py 生成的 jsonl（每条记忆若干问句）。
写库需要 danger-full-access（沙箱对 data/ 只读）。

用法：
    venv\\Scripts\\python.exe tools_memory_keys.py --questions _ab_tmp\\memory_questions_new.jsonl --dry-run
    venv\\Scripts\\python.exe tools_memory_keys.py --questions _ab_tmp\\memory_questions_new.jsonl --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import Dict, List


def load_questions(path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mid = str(row.get("memory_id") or "")
            qs = [str(q).strip() for q in (row.get("questions") or []) if str(q).strip()]
            if mid and qs:
                out[mid] = qs
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join("data", "memory.db"))
    ap.add_argument("--questions", default=os.path.join("_ab_tmp", "memory_questions_new.jsonl"))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.apply and not args.dry_run:
        args.dry_run = True

    qs = load_questions(args.questions)
    print(f"[load] {len(qs)} 条记忆带问法", flush=True)
    conn = sqlite3.connect(args.db, timeout=30)
    conn.row_factory = sqlite3.Row
    ids = {r["memory_id"] for r in conn.execute(
        "SELECT memory_id FROM memories WHERE deleted_at IS NULL")}
    hit = [m for m in qs if m in ids]
    miss = [m for m in qs if m not in ids]
    print(f"[match] 命中 {len(hit)} / 缺失 {len(miss)}", flush=True)
    for mid in hit[:3]:
        print(f"  {mid}: " + " | ".join(qs[mid][:3]), flush=True)
    if args.dry_run:
        print("[dry-run] 未写入", flush=True)
        return 0
    n = 0
    for mid in hit:
        conn.execute("UPDATE memories SET keys=?, updated_at=updated_at WHERE memory_id=?",
                     (json.dumps(qs[mid], ensure_ascii=False), mid))
        n += 1
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM memories WHERE keys != '[]'").fetchone()[0]
    print(f"[apply] 写入 {n} 条，库里带问法的记忆 {total} 条", flush=True)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

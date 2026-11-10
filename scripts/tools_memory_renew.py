"""记忆续期：把即将过期/已过期的记忆往后推，避免"突然什么都不记得"。

背景：memory_service 的 TTL 到期后 `get_relevant` 会静默跳过记录，产品层面就是
记忆凭空消失（2026-11-10 撞过一次：28 条研究纪律全部过期，检索恒空）。

用法：
    venv\\Scripts\\python.exe tools_memory_renew.py --dry-run
    venv\\Scripts\\python.exe tools_memory_renew.py --apply --days 30
    venv\\Scripts\\python.exe tools_memory_renew.py --apply --days 7 --ids m_abc,m_def

写库需要 danger-full-access（沙箱对 data/ 只读）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_service import MemoryService, now_ts  # noqa: E402


def describe(svc: MemoryService) -> List[str]:
    ts = now_ts()
    rows = svc._conn.execute(
        "SELECT memory_id, expires_at, user_id, task_id FROM memories WHERE deleted_at IS NULL"
    ).fetchall()
    out = []
    for r in rows:
        exp = r["expires_at"]
        if exp is None:
            state = "永不过期"
        elif exp <= ts:
            state = f"已过期 {int((ts - exp) / 86400)} 天"
        else:
            state = f"还剩 {int((exp - ts) / 86400)} 天"
        out.append(f"  {r['memory_id']}  {r['user_id']}/{r['task_id']}  {state}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join("data", "memory.db"))
    ap.add_argument("--days", type=float, default=30.0, help="从现在起再保多少天")
    ap.add_argument("--ids", default="", help="只续期这些 memory_id（逗号分隔）")
    ap.add_argument("--user-id", default="", help="只续期这个用户的记忆")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.apply and not args.dry_run:
        args.dry_run = True

    svc = MemoryService(args.db)
    before = svc.stats()
    lines = describe(svc)
    print(f"[db] {args.db}  存活 {before['alive']} 条，其中已过期 {before.get('expired', 0)} 条", flush=True)
    for line in lines[:10]:
        print(line, flush=True)
    if len(lines) > 10:
        print(f"  …… 共 {len(lines)} 条", flush=True)

    if args.dry_run:
        print(f"[dry-run] 会续期 {len(lines)} 条到 {args.days:g} 天后（--apply 才写库）", flush=True)
        svc.close()
        return 0

    ids = [s.strip() for s in args.ids.split(",") if s.strip()] or None
    renewed = svc.renew(days=args.days, memory_ids=ids, user_id=args.user_id or None)
    after = svc.stats()
    print(f"[apply] 续期 {len(renewed)} 条至 {time.strftime('%Y-%m-%d %H:%M')} 后 {args.days:g} 天；"
          f"已过期 {after.get('expired', 0)} 条 -> {after['alive']} 条可召回", flush=True)
    svc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

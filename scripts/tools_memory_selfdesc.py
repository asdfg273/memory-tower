"""把记忆条目改写成自描述句（结论 + 场景 + 为什么）并写回 data/memory.db。

依据 reports/memory_retrieval_ab.md：同一批 38 条 query、同一套 ground truth，
只把条目文本从"裸规则"改成"完整句子"，para 组 hit@1 从 0.111 抬到 0.333
（塔融合 0.389），keep 组持平，弱词面子集 0.182 → 0.263。

用法：
    venv\\Scripts\\python.exe tools_memory_selfdesc.py          # 干跑，打印 diff
    venv\\Scripts\\python.exe tools_memory_selfdesc.py --apply  # 写回（先备份，走 update() 版本化）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from typing import List, Sequence, Tuple

from memory_service import MemoryService


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join("data", "memory.db"))
    ap.add_argument("--map", default=os.path.join("data", "memory_selfdesc.json"))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    with open(args.map, encoding="utf-8") as f:
        mapping = json.load(f)
    if not os.path.exists(args.db):
        raise SystemExit(f"找不到 {args.db}")

    if args.apply:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        try:
            bak = args.db + f".bak-{stamp}"
            shutil.copyfile(args.db, bak)
        except OSError:
            os.makedirs("_ab_tmp", exist_ok=True)
            bak = os.path.join("_ab_tmp", f"memory.db.bak-{stamp}")
            shutil.copyfile(args.db, bak)
        print(f"[sd] backup -> {bak}")

    svc = MemoryService(args.db)
    todo: List[Tuple[str, str, str]] = []
    missing: List[str] = []
    same = 0
    for mid, new in mapping.items():
        rec = svc.get(mid)
        if rec is None:
            missing.append(mid)
            continue
        if rec.raw_source.strip() == str(new).strip():
            same += 1
            continue
        todo.append((mid, rec.raw_source, str(new)))

    print(f"[sd] map={len(mapping)} 待改={len(todo)} 已同={same} 缺失={len(missing)}")
    if missing:
        print(f"[sd] 缺失 id：{missing}")
    if not args.quiet:
        for mid, old, new in todo:
            print(f"[sd] {mid}\n     - {old}\n     + {new}")

    if not args.apply:
        print("[sd] 干跑结束（加 --apply 写回）")
        svc.close()
        return 0

    ok = 0
    for mid, _old, new in todo:
        try:
            ver = svc.update(mid, raw_source=new,
                             note="selfdesc: 改写为自描述句（reports/memory_retrieval_ab.md）")
            ok += 1
            print(f"[sd] {mid} -> v{ver}")
        except Exception as exc:  # noqa: BLE001
            print(f"[sd] {mid} 失败：{exc!r}")
    print(f"[sd] 写回 {ok}/{len(todo)}；stats={json.dumps(svc.stats(), ensure_ascii=False)}")
    svc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

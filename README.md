# 记忆塔：Titans 式快速权重记忆的实测研究（开源包）

一个小模型长时记忆系统的**实测研究 + 可运行实现**：把 Titans（arXiv:2501.00663）的
"神经长期记忆"从公式落成能在消费级单卡（RTX 5060 Laptop 8 GB）上跑的运行时，
然后用**可机检任务**逐一测量它的能力边界——包括它不成立的地方。

## TL;DR（每条数字都能复现，出处见主论文）

| # | 结论 | 关键数字 | 出处 |
|---|---|---|---|
| 1 | 塔作为**联想记忆**成立 | 值级检索 16/16 格 1.000；写入—读取链路四格 1.000 | §5.1–§5.3 |
| 2 | **写入协议**决定上限 | 40 槽匹配率 0.725 → **1.000**（3×20 步精修，代价 +15% 步数） | §5.4 |
| 3 | 键与值用**模型词表嵌入** | 未覆盖名称 0.000 → **1.000**；读出向量名次中位 0 | §5.5 |
| 4 | 交付的运行时可用（限定适用域） | 40 槽 40/40、80 槽 80/80、多 token 名称 40/40；未教实体拒答 | §5.6–§5.7 |
| 5 | 塔作为**推理增益通道**不成立 | 向量注入 M0=M1=M2=M3=**0.0000**，同题文本通道 0.125 / 0.250 | §5.8 |
| 6 | 容量有**闭式**解法 | 460 步梯度写入 → 闭式求解：N=1280 从 0.624 → **0.988**；300 槽 65 s → **0.26 s** | §5.5、§5.10 |
| 7 | 扩容**不产生新能力** | 36 格阶梯被一条两参数曲线压平（RMSE **0.013**） | §5.11 |
| 8 | 塔**不能当检索层**，杠杆在条目形态与打分器 | 零重叠改写 hit@1 塔 .111 = 平台 .111；自描述化 .111→**0.333**；idf 打分 .605→**0.658** | §5.12 |
| 9 | **容量与深度都不是瓶颈**，语义不变性在编码器里就丢了 | 交付 m=256 下 480 条全中（1.000）；堆到 8 层零收益；改写从 .767 训到 .958 但重度改写仍 **.062**，判别式训练最多 .448 | §5.14 |

一句话定位：**它是一个可从文本挂载的记忆运行时，不是一个 RAG 卖点，更不是"更强的记忆结构"。**

## 目录

```
output/
├─ README.md                     ← 本文件
├─ PAPER_TITANS_TOWER.md         ← 主论文（含参考文献 [R1]–[R27]）
├─ PAPER_FULL_LOG.md             ← 原始实验记录（330 KB，未整理，仅存档）
├─ references.bib                ← 27 条参考文献的 BibTeX
├─ reports/                      ← 九份关键报告
│   ├─ memory_retrieval_ab.md    ← 检索层对照实验全过程（C 阶段 / A′ / 自描述化 / embedding / 问法键 / TTL）
│   ├─ TOWER_MECHANISM.md        ← 塔的机制与结构
│   ├─ TOWER_SLOTDEPTH.md        ← 容量与不变性：T131–T133 三次端到端实验
│   ├─ TOWER_GATE.md             ← 门控互见 / MLP 选择器 / 两跳，以及 E2E 与口语边界（§2c–§2f）
│   ├─ TOWER_DEPTH.md            ← 深度塔：深度是多项式滤波，不是容量
│   ├─ ALBUM_TOWER.md            ← SD1.5 相册身份一致性：免训练注入 + 小投影头 + 真人数据
│   ├─ TOWER_VS_RAG.md           ← 与 RAG 的全维度对照
│   ├─ INSTRUMENT_FAILURES.md    ← 仪器失效清单（本项目最可迁移的产出）
│   └─ HANDOFF.md                ← 交接摘要
├─ requirements.txt              ← 依赖（CUDA 12.8 轮子说明见文件头注释）
├─ scripts/                      ← 关键脚本快照（权威版本在仓库根目录）
│   ├─ memory_service.py         ← 产品侧 SQLite 检索层（idf 加权 1+2 元；KEYS_ENABLED 开关）
│   ├─ tools_memory_ab.py        ← 检索层 5 臂对照 + 分组统计
│   ├─ tools_memory_keypolicy.py ← 问法键六种打法的对照（结论：默认关）
│   ├─ tools_memory_keys.py      ← 把问法回填进 memories.keys
│   ├─ tools_memory_renew.py     ← 记忆续期（TTL 到期会让检索静默变空）
│   ├─ tools_memory_questions.py ← 给记忆生成问句键（负结果实验，保留以证伪）
│   ├─ tools_memory_selfdesc.py  ← 记忆条目自描述化迁移器（带备份 + 版本化）
│   ├─ tools_tower_memory.py     ← 塔记忆：CLI / HTTP / MCP 三入口
│   ├─ tools_tower_gate.py       ← G1 门控 / G2 MLP 选择器 / G3 两跳 + 代数事实
│   ├─ tools_tower_head_probe.py ← H 阶段：冻结塔 + 线性读取头
│   ├─ tools_tower_e2e.py        ← 端到端：两个必要条件（检索 top-1、值必须是答案）
│   ├─ tools_tower_keyexp.py     ← 多问法键 + max-over-keys
│   ├─ tools_tower_alias.py      ← 边界：名词化对得上、口语对不上
│   ├─ tools_tower_valuecheck.py ← 真实文本上塔的值复现（更正"值在 span 外"）
│   ├─ tools_tower_depth.py      ← 深度塔：1/2/4 层与槽位地址乘法
│   ├─ tools_tower_slotdepth.py  ← §5.14 T131：槽位塔 + m 截断扫描
│   ├─ tools_tower_slotdepth_train.py ← §5.14 T132/T133：真训练投影 + 判别式训练
│   ├─ tools_tower_vs_rag.py     ← 事实池与问法模板（上面两个脚本的依赖）
│   ├─ tools_album_tower.py      ← SD1.5 相册：免训练注入六臂
│   ├─ tools_album_head.py       ← 小投影头 / 连接器（合成角色）
│   ├─ tools_album_head_real.py  ← 小投影头（CelebA 真人）
│   ├─ tower_runtime.py          ← 塔运行时（验收自检）
│   ├─ tools_tower_T3.py         ← 快速权重核（tower_runtime 的依赖）
│   ├─ longctx.py                ← 分块 prefill 长上下文路径
│   └─ common.py                 ← 共用工具（键/值编码、字面相似度等）
├─ numbers/                      ← 机器产物（论文数字来源）
│   ├─ memory_ab_platform.json   ← 检索层实验
│   ├─ tower_slotdepth.json      ← §5.14 T131（容量 / m 截断）
│   ├─ tower_slotdepth_train.json← §5.14 T132（训练投影）
│   ├─ tower_slotdepth_disc.json ← §5.14 T133（判别式 + 查询侧扩展）
│   ├─ tower_depth.json          ← 深度塔（容量 / 条件数 / 组合 / λ 对照）
│   ├─ tower_e2e.json            ← 端到端五指标 × 三规模
│   ├─ tower_head_probe.json     ← H 阶段五臂
│   ├─ album_tower.json          ← 相册六臂 + 逐图分数
│   ├─ album_head.json           ← 小投影头（合成）
│   └─ album_head_real.json      ← 小投影头（CelebA）
└─ data/
    └─ memory_selfdesc.json      ← 28 条真实记忆 → 自描述句 的映射（实验数据集）
```

## 最短复现路径（约 30 分钟，单卡 8 GB）

```powershell
python -m venv venv; .\venv\Scripts\pip install -r requirements.txt   # torch / transformers 等
$env:HF_HUB_OFFLINE="1"

# 1) 运行时验收：40/80 槽严格名称匹配 + 未教实体拒答（≈30 s/次）
.\venv\Scripts\python.exe tower_runtime.py --selftest --n 40 --out reports\repro\tower_accept40.json

# 2) 检索层对照：5 臂 × 38 query（≈3 min，含塔加载）
.\venv\Scripts\python.exe tools_memory_ab.py

# 3) 记忆条目自描述化（干跑先看 diff，确认后加 --apply）
.\venv\Scripts\python.exe tools_memory_selfdesc.py

# 4) 容量与不变性：T131 截断扫描（≈2 min）/ T132 训练投影（≈6 min，需 GPU）
.\venv\Scripts\python.exe tools_tower_slotdepth.py --ns 480 --dims 256,96,48,24,16
.\venv\Scripts\python.exe tools_tower_slotdepth_train.py --dims 48,96
```

产物写到 `_ab_tmp/` 与 `reports/repro/`；`data/` 需要可写（SQLite）。

## 数据与隐私

- `data/memory_selfdesc.json` 里的 28 条是**作者自己的研究纪律条目**（如"报告要报覆盖–精度对"），
  不含第三方个人数据，可直接作为检索实验的示例数据集。
- 本包**不含**：模型权重、`checkpoints_tower/`、`data/memory.db`、`venv/`、`_ab_tmp/`、任何 API key。

## 已知缺口（如实列出）

1. 论文附录 B 第 1 条要求"数字必须由脚本重算并比对"，但 `tools_gen_doc_numbers.py --check`
   目前跑不通：它依赖的 `tools_perturbation_ladder` 模块已不存在。数字自检链是断的。
2. `PAPER.md`（原始记录）早期章节描述的是另一套检索实现（`nya_memory` + BGE-M3），
   与实际运行的 `memory_service.py` 有出入——以 `PAPER_TITANS_TOWER.md` 为准。
3. 检索实验规模很小：28 条记忆 × 38 条手写 query，`hit@1` 的单条波动就是 ±0.026。
4. §5.14 只试了"冻结编码器 + 线性投影"这一族。**微调编码器**与**查询侧 LLM 扩展**都没跑，
   所以"塔永远读不出改写"不成立，成立的只是"线性投影救不了"；且 T132/T133 只有单 seed。

## 引用

参考文献 [R1]–[R27] 见主论文文末；BibTeX 见 `references.bib`。
若引用本工作，请引用主论文与本仓库：

> asdfg273. (2026). 基于 Titans 架构的记忆塔：小模型长时记忆的实测研究.
> https://github.com/asdfg273/memory-tower

## 许可

| 内容 | 协议 | 文件 |
| --- | --- | --- |
| 代码（`scripts/*.py`） | MIT | [`LICENSE`](LICENSE) |
| 文字（论文、报告、README） | CC BY 4.0 | [`LICENSE-TEXT`](LICENSE-TEXT) |
| 实验数据（`numbers/`、`data/`） | 随代码，MIT | [`LICENSE`](LICENSE) |

引用本仓库建议使用：`asdfg273 (2026). 基于 Titans 架构的记忆塔：小模型长时记忆的实测研究. GitHub: asdfg273/memory-tower`

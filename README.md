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

一句话定位：**它是一个可从文本挂载的记忆运行时，不是一个 RAG 卖点，更不是"更强的记忆结构"。**

## 目录

```
output/
├─ README.md                     ← 本文件
├─ PAPER_TITANS_TOWER.md         ← 主论文（含参考文献 [R1]–[R27]）
├─ PAPER_FULL_LOG.md             ← 原始实验记录（330 KB，未整理，仅存档）
├─ references.bib                ← 27 条参考文献的 BibTeX
├─ reports/                      ← 四份关键报告
│   ├─ memory_retrieval_ab.md    ← 检索层对照实验全过程（C 阶段 / A′ / 自描述化 / embedding）
│   ├─ TOWER_MECHANISM.md        ← 塔的机制与结构
│   ├─ INSTRUMENT_FAILURES.md    ← 仪器失效清单（本项目最可迁移的产出）
│   └─ HANDOFF.md                ← 交接摘要
├─ requirements.txt              ← 依赖（CUDA 12.8 轮子说明见文件头注释）
├─ scripts/                      ← 关键脚本快照（权威版本在仓库根目录）
│   ├─ tools_memory_ab.py        ← 检索层 5 臂对照 + 分组统计
│   ├─ tools_memory_questions.py ← 给记忆生成问句键（负结果实验，保留以证伪）
│   ├─ tools_memory_selfdesc.py  ← 记忆条目自描述化迁移器（带备份 + 版本化）
│   ├─ tools_tower_memory.py     ← 塔记忆：CLI / HTTP / MCP 三入口
│   ├─ tower_runtime.py          ← 塔运行时（验收自检）
│   ├─ tools_tower_T3.py         ← 快速权重核（tower_runtime 的依赖）
│   ├─ longctx.py                ← 分块 prefill 长上下文路径
│   ├─ common.py                 ← 共用工具（键/值编码、字面相似度等）
│   └─ memory_service.py         ← 产品侧 SQLite 检索层（idf 加权 1+2 元）
├─ numbers/
│   └─ memory_ab_platform.json   ← 检索层实验的机器产物（论文数字来源）
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

r"""
tower_runtime.py —— **记忆塔运行时（可用形态）**
================================================================

把 T3–T53 验证过的通路打包成一个可复用组件。

    写入  键   key  = normalize( mean( E[ tokenize(键短语) ] ) )
          值   val  = normalize( mean( E[ tokenize(名称) ] ) )
          -> w_k / w_v -> 码(R^m) -> 快速权重：基础 400 步 + **精修 3 轮 x 20 步**
             （T48 的 R3s20_fwd；顺序 = 写入顺序；精修 = 全局再平衡）
    查询  问题文本 -> 键短语 -> key -> **用 w_k 同一编码读出**（关键，见下）
          -> r(R^m) -> 免训练最近邻解码（T49）-> 名称字符串
          -> 两个拒绝信号：读出边际（dict 模式）+ 键寻址相似度（无记录阻断）

**键与值用同一个原则（T49 + T53）**：都取**模型自己的词嵌入空间**
    -> 两者的**编码构造无需逐项学习映射**（模型自带参数；不含任何按实体/名称训练的参数）
    -> **注意区分两笔成本**：编码构造 = 0 训练成本；**记忆写入** = 每条基础 400 步 + 精修 3x20 步
    -> "未见"一律指**相对于暖机与标定所用集合**未见（暖机只用随机向量，
       标定只用同模板随机容器词，二者都不含被评估的实体或名称）
    -> T53：40 槽 top1 **0.975~1.000**（实体键两两 cos 中位 0.025、最大 0.635；
       对比 T51 的隐状态键：中位 0.44~0.52、最大 0.93 -> 白化后最好也只有 0.913）

两个必须踩对的坑（本项目实测）
    1  读出必须用 **w_k** 编码（与写入同一投影）；
       用 w_q（那是寻址用的）读出 -> cos 中位从 0.62 掉到 0.10、40 槽 top1 0.98 -> 0.125
    2  m = 256（T50：容量是主因）+ 精修 3 轮（T48）

用法
----
    python tower_runtime.py --selftest --n 40          # 端到端验收（未覆盖名称）
    python tower_runtime.py --demo                     # 演示
    python tower_runtime.py --build                    # 预建缓存（暖机）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from tools_tower_T3 import FastWeightMemory

HF = r"E:\ai_models\qianwen\gguf_xiaoxue\merged_hf"
CACHE_DIR = "checkpoints_tower"
QTPL = "箱子{q}的密码是什么？只回答密码本身。"
QTPL_A = "箱子{q}的{a}是什么？只回答{a}本身。"
_ATTRS = ("密码", "电话", "地址")
# 默认键短语抽取：模板 "箱子X的密码" -> "X"；抽不到就用整句
KEY_PATTERNS = [re.compile(r"箱子(\S+?)的密码"), re.compile(r"「(.+?)」"),
                re.compile(r"的密码是(.+?)[。？?]"), re.compile(r"箱子(\S+?)的")]
# 绑定键（bind）：把**属性**一起编进键短语，"箱子甲的电话" -> "甲的电话"
ATTR_PATTERN = re.compile(r"箱子(\S+?)的(.+?)(?:是什么|是|多少|[？?。])")
# 验收用池（与暖机无关：暖机只用**随机向量**，不含任何实体/名称）
_N_POOL = ("锕铽钬齉爨鑫淼焱垚钌锂铍钠镁铝硅磷硫氯氩钾钙钪钛钒铬锰铁钴镍铜锌"
           "镓锗砷硒溴氪铷锶钇锆铌钼锝银镉铟锡锑碲碘铯钡镧铈镨钕钷钐铕钆镝铒铥镱镥"
           "鼐鼒龘靐齾爩麤龗灪龖厵滟癵驫麣纞虋讟钃鸜麷鞻韽韾顟顠饙饡騳髾鬾魑魅魍魉"
           "饕餮夔獬麒麟凤凰鸳鸯鹧鸪鸬鹚鹈鹕")
_E_POOL = ("金木水火土东南西北春夏秋冬红橙黄绿青蓝紫灰白黑一二三四五六七八九十"
           "百千万亿零壹贰叁肆伍陆柒捌玖拾你我他她它们上下左右前后内外中边角圆方"
           "长短高低凡尘世俗情缘梦影虚幻空灵静默宁康泰平安喜乐忧愁思念")
# 大 n 用的扩展池（n > 96 时补两字组合）：与 _E_POOL 不重叠，避免键共线
_E_POOL_EXT = "龘靐齾爩麤龗灪龖厵滟癵驫麣纞虋讟钃鸜麷鞻韽韾顟顠饙饡騳髾鬾魑魅魍魉"
_N_POOL_EXT = "琮璟瑄珂瑾瑜璇瓒瓔璩琚瑭瑨琬瑯璆瓘瓚瓛瑀瑢璠璟瑧瑨瑫瑮璪璬璯璟瑱瑳瑽"
# n > 96 时用下面两个池（tools_tower_poolgen.py 从词表里 max-min 挑出的 400 个单 token 汉字，
# 彼此嵌入几乎正交：两两 cos 均值 0.014、最大 0.094）。旧 EXT 池 30 个罕字同为生僻字，
# 组合后键/值高度共线，n=300 的验收会卡在 0.32 而与写入器无关 —— 那是池的锅。
_E_POOL_EXT2 = (
    "年滉谢竹驻谋尖表威便架這丐氏鲁配真听链呼私甲省众工鉴啊旭予印过括量欲比卡电史佛团办浆烦投还逐取之侧秀快亡双袗味化道约稀负總容盐洞靖些替首良斐冲处非掉被雏城特相底振导编劝徒睡里怎剥院久戎螺第寒脚效王册庄条川错"
    "业游临艾汉满透类心云薮登恒哥祭关丨软明育切述来结仅魄符昀尔具笔股女琳碌微間题犬拖动抗伤补極帆压友花臣紧宝核後変鼎生本向斗期例暨马得据管今轮复洲根研翻试靠韶应莫萧扇野象梗责致索埋开散饭孝干亲解枚操彭豪涨离廣垂炎访污望商陌适阶禧悦付造立政自入"
    "跨见钢抱旬领固滤任吉整打元弊发律血判共申醐准弹懂亭料坦祖止厚禁推涂回努迷偶倍销席机风皮坡拳运厥果宛最蒸财英塘赤形恋您序燕教老鲜翁事梳唇考悉峥令字理塑盖抑但殊荷罢靶次司垩租客虹加助腺伝布版钟凑呈郝算焦央盆洢着挑様芒沃社恩克企体械毒等系驰饹胎"
    "认壁設破帖面易智简含孤装兰会琼歼尊包差蒂茗纬收纪筋楞對指钦哨营灯况周各折品分贫质氧玄粟觉堪悔鲀惯注偭门砂少褒归除介愈用"
)
_N_POOL_EXT2 = (
    "数菥瓷雄拿改曳陈肝受台灌亦景联别乎了谁帕仓善約作初这阴菌主泛足问鄂倒报感学帅保悬温丁极雪络歌像身惇域节姐辟眉牛素硬刻実住屈乱長只典费渡盈制桃锄朝地语忘参份官现进线村敌晓能闭半撞吸娇卢口帮属太夹待通追频翼巨称"
    "环者递占剩重童來役米排展球审假识忑潜图拢色赋圣选模走哦痛局耐油旨決佐科曾滑铅支斛乘毛正径修侵诸旁从使汛祥乾坑调頂無医种敬刷信裁专引视产芯芳消巳并旅顿精定尋再扁林驱弱力希照法庙笑多拥突传冈层宇古彼性每日基严浒骓碧赛寿小床诉疏垫棍流匪录宾亚秘"
    "復乳該培纹患恐崭石甘京综運翰熔断雀落构起丧揣露键党菜控示添夫夸柴內成齐历勉宜缑限交廉议竖移蒙郁海動供革趣拜稍集远梠州施澳许查殖洽伯薛魔堂幕阘罪急偰尾放超摇亮针缩家波習执绩関類礴阮邑庶必源矢囊牌争完民物滴宗時蓬禅纠践吐武熟赖星撰沫點悠番藤宕"
    "服瞬因杀顾警肖达迟郊阂豆利欠偕裂漂玩斯喃卦機演氣穷端卷丛委朗卿轨套押偏尿片计炮杯笆却稷奴甸俩酬组變統钧闲貌泽爱媒防伞寄"
)


class TowerRuntime:
    """记忆塔运行时：文本进、文本出。"""

    def __init__(self, model_dir: str = HF, m: int = 256, dim: int = 2048,
                 seed: int = 0, device: Optional[str] = None,
                 base_steps: int = 400, refine_rounds: int = 3,
                 refine_steps: int = 20, warm_pairs: int = 40,
                 cache_dir: str = CACHE_DIR, verbose: bool = True,
                 key_fn: Optional[Callable[[str], str]] = None,
                 write_mode: str = "grad", key_mode: str = "entity",
                 lam: float = 1e-2) -> None:
        self.m, self.dim, self.seed = m, dim, seed
        self.base_steps, self.refine_rounds, self.refine_steps = (
            base_steps, refine_rounds, refine_steps)
        self.warm_pairs = warm_pairs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        # write_mode: "grad" = R3s20_fwd 梯度写（原交付路径）
        #             "closed" = 岭回归闭式写 W=(KᵀK+λI)⁻¹KᵀZ（+ Sherman-Morrison 增量）
        # key_mode:   "entity" = 键只用实体（属性不进键，同实体多属性会撞）
        #             "bind"   = 键 = 实体+属性（绑定键，但字符串池化是**加法**绑定）
        #             "hrr"    = 键 = 实体嵌入 ⊙ 属性嵌入（乘法绑定，键满秩）
        if write_mode not in ("grad", "closed"):
            raise ValueError(f"write_mode 必须是 grad/closed，得到 {write_mode!r}")
        if key_mode not in ("entity", "bind", "hrr"):
            raise ValueError(f"key_mode 必须是 entity/bind/hrr，得到 {key_mode!r}")
        self.write_mode, self.key_mode, self.lam = write_mode, key_mode, lam
        self.key_fn = key_fn or (self._bind_key_phrase
                                 if key_mode in ("bind", "hrr")
                                 else self._default_key_phrase)

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, dtype=torch.bfloat16, trust_remote_code=True
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.Ein = self.model.get_input_embeddings().weight
        if self.verbose:
            print(f"[runtime] 模型载入 {time.time()-t0:.1f}s · hidden={self.dim}")

        self.mem = FastWeightMemory(self.dim, self.m)
        os.makedirs(cache_dir, exist_ok=True)
        tag = hashlib.sha1(f"{os.path.basename(model_dir)}|{m}|{dim}|{seed}|"
                           f"{base_steps}|{refine_rounds}|{refine_steps}|"
                           f"{warm_pairs}".encode()).hexdigest()[:12]
        self.cache = os.path.join(cache_dir, f"runtime_{tag}.pt")
        self.entries: List[Tuple[str, str]] = []
        self.Z: Optional[torch.Tensor] = None
        self.W: Optional[torch.Tensor] = None
        self.Wq: Optional[torch.Tensor] = None
        self.fast: Optional[Dict[str, torch.Tensor]] = None
        self._dirty = True
        self._Zall: Optional[torch.Tensor] = None
        # 闭式写入的状态：Wc = (KᵀK+λI)⁻¹KᵀZ；Ainv = (KᵀK+λI)⁻¹（增量用）
        self.Wc: Optional[torch.Tensor] = None
        self._Ainv: Optional[torch.Tensor] = None
        # 拒绝阈值：由 --selftest 标定（40 槽实测 键寻址 已知最小 1.000 / 未知最大 0.497）
        self.margin_thr, self.addr_thr = 0.02, 0.75

    # ---------------- 编码（键与值同一原则） ----------------
    @staticmethod
    def _default_key_phrase(question: str) -> str:
        for p in KEY_PATTERNS:
            mm = p.search(question)
            if mm:
                return mm.group(1)
        return question

    @staticmethod
    def _bind_key_phrase(question: str) -> str:
        """绑定键：实体 + 属性。'箱子甲的电话是什么？' -> '甲的电话'。"""
        mm = ATTR_PATTERN.search(question)
        if mm:
            return f"{mm.group(1)}的{mm.group(2).strip()}"
        return TowerRuntime._default_key_phrase(question)

    def _embed(self, text: str) -> torch.Tensor:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"空文本：{text!r}")
        with torch.no_grad():
            E = self.Ein[torch.tensor(ids, device=self.device)].float()
        return F.normalize(E.mean(0), dim=-1).cpu()

    def key(self, question: str) -> torch.Tensor:
        if self.key_mode != "hrr":
            return self._embed(self.key_fn(question))
        # hrr：乘法绑定（μ_实体 ⊙ μ_属性）。字符串拼接是加法绑定，
        # 键矩阵秩只有 n_实体 + n_属性 − 1，装不下 n_实体 × n_属性 条事实。
        mm = ATTR_PATTERN.search(question)
        if not mm:
            return self._embed(self.key_fn(question))
        e, a = mm.group(1).strip(), mm.group(2).strip()
        return F.normalize(self._embed(e) * self._embed(a), dim=-1)

    def value_of(self, name: str) -> torch.Tensor:
        return self._embed(name)

    # ---------------- 暖机（随机键/值；可缓存） ----------------
    def _warm(self, steps: int = 200) -> None:
        g = torch.Generator().manual_seed(909 + self.seed)
        K = torch.randn(self.warm_pairs, self.dim, generator=g)
        K = K / K.norm(dim=-1, keepdim=True)
        V = torch.randn(self.warm_pairs, self.dim, generator=g)
        V = V / V.norm(dim=-1, keepdim=True)
        torch.manual_seed(self.seed)
        mps = (list(self.mem.w_k.parameters()) + list(self.mem.w_v.parameters())
               + list(self.mem.w_q.parameters()))
        opt = torch.optim.AdamW(mps, lr=3e-3)
        for _ in range(steps):
            kp = self.mem.w_k(K); ve = self.mem.w_v(V)
            w = self.mem.init_fast("cpu"); raws = []
            for i in range(self.warm_pairs):
                w, _ = self.mem.write_step(w, kp[i], ve[i].detach(), 0.1)
                raws.append(self.mem.read(w, kp[i]))
            sr = torch.stack(raws, 0)
            loss = (F.cross_entropy(self.mem.w_q(K) @ kp.t(),
                                    torch.arange(self.warm_pairs))
                    + 1.0 - F.cosine_similarity(sr, ve.detach(), dim=-1).mean())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mps, 1.0); opt.step()
        for p in mps:
            p.requires_grad_(False)

    def build(self, force: bool = False) -> "TowerRuntime":
        t0 = time.time()
        if os.path.exists(self.cache) and not force:
            st = torch.load(self.cache, map_location="cpu", weights_only=False)
            with torch.no_grad():
                self.mem.w_k.weight.copy_(st["w_k"])
                self.mem.w_v.weight.copy_(st["w_v"])
                self.mem.w_q.weight.copy_(st["w_q"])
            if self.verbose:
                print(f"[runtime] 缓存命中（{time.time()-t0:.1f}s）")
            return self
        self._warm()
        torch.save({"w_k": self.mem.w_k.weight.detach().clone(),
                    "w_v": self.mem.w_v.weight.detach().clone(),
                    "w_q": self.mem.w_q.weight.detach().clone(),
                    "cfg": dict(m=self.m, dim=self.dim, seed=self.seed)}, self.cache)
        if self.verbose:
            print(f"[runtime] 暖机完成 -> {self.cache}（{time.time()-t0:.1f}s）")
        return self

    # ---------------- 写入 ----------------
    def teach(self, question: str, name: str) -> int:
        self.entries.append((question, name))
        self._dirty = True
        return len(self.entries) - 1

    def teach_many(self, pairs: List[Tuple[str, str]]) -> "TowerRuntime":
        for q, n in pairs:
            self.teach(q, n)
        return self

    def commit(self) -> None:
        if not self.entries:
            return
        N = len(self.entries)
        t0 = time.time()
        with torch.no_grad():
            K = torch.stack([self.key(q) for q, _ in self.entries], 0)
            V = torch.stack([self.value_of(n) for _, n in self.entries], 0)
            self.W = self.mem.w_k(K)                 # **写与读共用**
            self.Z = self.mem.w_v(V)
            self.Wq = self.mem.w_q(K)                # 寻址用（缓存，避免每次查询重算）
        tw = time.time() - t0
        if self.write_mode == "closed":
            t1 = time.time()
            self._fit_closed()
            self._dirty = False
            if self.verbose:
                print(f"[runtime] commit {N} 条 · **闭式** W=(KᵀK+λI)⁻¹KᵀZ"
                      f"（编码 {tw:.2f}s + 求解 {time.time()-t1:.3f}s）")
            return
        w = self.mem.init_fast("cpu")
        for i in range(N):
            for _ in range(self.base_steps):
                w, _ = self.mem.write_step(w, self.W[i], self.Z[i], 2.0)
        for _ in range(self.refine_rounds):
            for i in range(N):
                for _ in range(self.refine_steps):
                    w, _ = self.mem.write_step(w, self.W[i], self.Z[i], 2.0)
        self.fast = w
        self._dirty = False
        if self.verbose:
            print(f"[runtime] commit {N} 条 · {self.base_steps}+"
                  f"{self.refine_rounds}x{self.refine_steps} 步/条 · {time.time()-t0:.1f}s")

    # ---------------- 闭式写入（W = (KᵀK+λI)⁻¹KᵀZ） ----------------
    def _fit_closed(self) -> None:
        K, Z = self.W, self.Z                     # type: ignore[assignment]
        G = K.t() @ K + self.lam * torch.eye(self.m, dtype=K.dtype)
        self.Wc = torch.linalg.solve(G, K.t() @ Z)
        self._Ainv = torch.linalg.inv(G)

    def _closed_add(self, k: torch.Tensor, z: torch.Tensor) -> float:
        """Sherman–Morrison 增量写入（O(m²)，不重解）。返回耗时 ms。"""
        t0 = time.time()
        v = self._Ainv @ k                        # type: ignore[operator]
        denom = 1.0 + float(k @ v)
        self._Ainv = self._Ainv - torch.outer(v, v) / denom
        self.Wc = self.Wc + torch.outer(v, z - k @ self.Wc) / denom
        return (time.time() - t0) * 1e3

    def add(self, question: str, name: str) -> Dict[str, object]:
        """追加一条并**立刻**写入（闭式模式下 O(m²)，不重写全部）。"""
        with torch.no_grad():
            k = self.mem.w_k(self.key(question).unsqueeze(0))[0]
            z = self.mem.w_v(self.value_of(name).unsqueeze(0))[0]
            wq = self.mem.w_q(self.key(question).unsqueeze(0))[0]
        self.entries.append((question, name))
        self._Zall = None
        if self.write_mode == "closed" and self.Wc is not None:
            if self.W is not None:
                self.W = torch.cat([self.W, k[None]], 0)
                self.Z = torch.cat([self.Z, z[None]], 0)
                self.Wq = torch.cat([self.Wq, wq[None]], 0)
            ms = self._closed_add(k, z)
            self._dirty = False
            return {"mode": "closed", "ms": round(ms, 3), "n": len(self.entries)}
        self._dirty = True                         # 梯度模式：需整体重写
        return {"mode": "grad", "ms": None, "n": len(self.entries)}

    def _read(self, x: torch.Tensor) -> torch.Tensor:
        if self.write_mode == "closed" and self.Wc is not None:
            return x @ self.Wc
        return self.mem.read(self.fast, x)         # type: ignore[arg-type]

    # ---------------- 查询 ----------------
    def _vocab_codes(self) -> torch.Tensor:
        if self._Zall is None:
            chunks = []
            with torch.no_grad():
                for s in range(0, self.Ein.shape[0], 16384):
                    blk = F.normalize(self.Ein[s:s + 16384].to(self.device).float(),
                                      dim=-1).cpu()
                    chunks.append(self.mem.w_v(blk))
            self._Zall = torch.cat(chunks, 0)
        return self._Zall

    def probe(self, question: str, mode: str = "dict") -> Dict[str, object]:
        if self._dirty:
            self.commit()
        with torch.no_grad():
            kq = self.key(question)
            x = self.mem.w_k(kq.unsqueeze(0))[0]          # **同一编码**
            r = self._read(x)
            Z = self._vocab_codes() if mode == "vocab" else self.Z
            S = F.normalize(r, dim=-1) @ F.normalize(Z, dim=-1).t()
            top = torch.topk(S, k=min(2, S.numel()))
            margin = float(top.values[0] - top.values[1]) if S.numel() > 1 else 9.0
            idx = int(top.indices[0])
            # 键寻址相似度：这个键在不在记忆里（无记录阻断用）
            aq = F.normalize(self.mem.w_q(kq.unsqueeze(0)), dim=-1)
            addr = float((aq @ F.normalize(self.Wq, dim=-1).t()).max())
        if mode == "vocab":
            return {"answer": self.tok.decode([idx], skip_special_tokens=True),
                    "score": float(top.values[0]), "margin": margin, "addr": addr,
                    "entry": None, "mode": mode}
        return {"answer": self.entries[idx][1], "score": float(top.values[0]),
                "margin": margin, "addr": addr, "entry": idx, "mode": mode}

    def query(self, question: str, mode: str = "dict",
              margin_thr: Optional[float] = None,
              addr_thr: Optional[float] = None) -> Optional[str]:
        """返回名称；**任一拒绝信号不过阈值 -> None（无记录阻断）**。"""
        p = self.probe(question, mode)
        mt = self.margin_thr if margin_thr is None else margin_thr
        at = self.addr_thr if addr_thr is None else addr_thr
        if p["margin"] < mt or p["addr"] < at:   # type: ignore[operator]
            return None
        return p["answer"]  # type: ignore[return-value]

    # ---------------- 多跳组合（接口化读出，§5.6.3） ----------------
    def link(self, src: str, rel: str, dst: str) -> int:
        """教一条**关系边**：`箱子{src}的{rel}` = `{dst}`。

        与 `teach` 的区别只是问句模板固定为 `QTPL_A`，好处是同实体多条关系在
        `key_mode="hrr"`（乘性绑定，满秩）下互不干扰 —— 这是多跳链能建起来的前提。
        """
        return self.teach(QTPL_A.format(q=src, a=rel), dst)

    def hop(self, src: str, rel: str, mode: str = "dict") -> Dict[str, object]:
        """单跳：`{src}` 的 `{rel}` 是谁（返回 probe 的完整字典）。"""
        return self.probe(QTPL_A.format(q=src, a=rel), mode)

    def chain(self, src: str, rels: Sequence[str], route: str = "iface",
              mode: str = "dict", margin_thr: Optional[float] = None,
              addr_thr: Optional[float] = None) -> Dict[str, object]:
        """多跳组合：`src -> rels[0] -> rels[1] -> ...`。

        两条读出路线（与 `tools_tower_compose_deep.py` 的语义一致）：
            route="iface"  每跳**解码成名称**，再用该名称重建下一跳的键（误差吸收）
            route="vec"    把读出向量 r 归一化后直接当下一跳的查询（误差乘法累积）

        返回值：`path`（含起点的实体名序列）、`answers`、`margins`、`addrs`、
        `ok`（每跳是否同时过 margin/addr 阈值，vec 路线的 addr 为 None）、`hit_last`。
        """
        if route not in ("iface", "vec"):
            raise ValueError(f"route 必须是 iface/vec，得到 {route!r}")
        if self._dirty:
            self.commit()
        mt = self.margin_thr if margin_thr is None else margin_thr
        at = self.addr_thr if addr_thr is None else addr_thr
        path: List[str] = [src]
        answers: List[str] = []
        margins: List[float] = []
        addrs: List[Optional[float]] = []
        oks: List[bool] = []
        x_vec: Optional[torch.Tensor] = None
        for k, rel in enumerate(rels):
            with torch.no_grad():
                q = QTPL_A.format(q=path[-1], a=rel)
                if route == "vec" and x_vec is not None:
                    r = self._read(F.normalize(x_vec, dim=-1))
                    addr: Optional[float] = None          # 向量路线没有"键在不在库里"的信号
                else:
                    kq = self.key(q)
                    r = self._read(self.mem.w_k(kq.unsqueeze(0))[0])
                    aq = F.normalize(self.mem.w_q(kq.unsqueeze(0)), dim=-1)
                    addr = float((aq @ F.normalize(self.Wq, dim=-1).t()).max())
                Z = self._vocab_codes() if mode == "vocab" else self.Z
                S = F.normalize(r, dim=-1) @ F.normalize(Z, dim=-1).t()
                top = torch.topk(S, k=min(2, S.numel()))
                margin = float(top.values[0] - top.values[1]) if S.numel() > 1 else 9.0
                idx = int(top.indices[0])
            name = (self.tok.decode([idx], skip_special_tokens=True) if mode == "vocab"
                    else self.entries[idx][1])
            ok = margin >= mt and (addr is None or addr >= at)
            answers.append(name)
            margins.append(margin)
            addrs.append(addr)
            oks.append(bool(ok))
            path.append(name)
            x_vec = r if route == "vec" else None
        return {"path": path, "answers": answers, "margins": margins, "addrs": addrs,
                "ok": oks, "route": route, "hit_last": (path[-1] if path else src)}

    # ---------------- 长上下文（新增，实现见 longctx.py） ----------------
    def answer_long(self, context: str, question: str, **kw) -> Dict[str, object]:
        """长上下文问答：**分块 prefill**（8 GB 卡可跑 32k）+ 可选记忆前置 / 向量注入。

        一次性 prefill 会把整段 logits 物化（32k 时 >20 GiB）而 OOM；分块后 32k 约 8.8 GB，
        且快约 90 倍（见 PAPER §4.5）。参数见 longctx.answer_long：
            use_memory=True  先用塔查表，命中则把答案作为一行短记忆前置（约 30 token）
            inject_vec       调用方若有连接器，可给 (1, hidden) 向量在第 site 层注入
        """
        from longctx import answer_long as _al
        return _al(self, context, question, **kw)
    # ---------------- 持久化 ----------------
    def save(self, path: str) -> None:
        torch.save({"entries": self.entries, "fast": self.fast, "Z": self.Z,
                    "W": self.W, "Wq": self.Wq, "margin_thr": self.margin_thr,
                    "addr_thr": self.addr_thr, "Wc": self.Wc, "Ainv": self._Ainv,
                    "write_mode": self.write_mode, "key_mode": self.key_mode,
                    "lam": self.lam,
                    "cfg": dict(m=self.m, dim=self.dim, seed=self.seed)}, path)

    def load(self, path: str) -> "TowerRuntime":
        st = torch.load(path, map_location="cpu", weights_only=False)
        self.entries = [tuple(x) for x in st["entries"]]
        self.fast, self.Z, self.W = st["fast"], st["Z"], st["W"]
        self.Wq = st.get("Wq")
        self.Wc, self._Ainv = st.get("Wc"), st.get("Ainv")
        self.write_mode = st.get("write_mode", self.write_mode)
        self.key_mode = st.get("key_mode", self.key_mode)
        self.lam = st.get("lam", self.lam)
        self.margin_thr = st.get("margin_thr", 0.0)
        self.addr_thr = st.get("addr_thr", 0.0)
        self._dirty = self.fast is None and self.Wc is None
        return self

    def info(self) -> Dict[str, object]:
        return {"m": self.m, "dim": self.dim, "entries": len(self.entries),
                "committed": not self._dirty,
                "write_mode": self.write_mode, "key_mode": self.key_mode,
                "protocol": ("closed-form ridge" if self.write_mode == "closed"
                             else f"{self.base_steps}+{self.refine_rounds}x"
                                  f"{self.refine_steps}")}


# ==================== 验收 / 演示 / CLI ====================
def _spread_combos(ext: List[str], need: int) -> List[str]:
    """从扩展字池里生成 need 个两字组合，并打散"同首字聚集"。

    行优先枚举 `a+b` 时前 30 个组合共享首字 —— 均值池化的键会高度共线。
    这里让首、尾字各按与 n 互素的步长独立轮转，组合两两不重复，
    前几百个实体的两个字符都分散在整个字池上。
    """
    n = len(ext)
    sa = next(s for s in (37, 53, 61, 71, 83, 97) if n % s)
    sb = next(s for s in (101, 103, 107, 109, 113) if n % s)
    return [ext[(i * sa) % n] + ext[(i * sb) % n] for i in range(need)]


def _pools(n: int) -> Tuple[List[str], List[str]]:
    """验收用池：n ≤ 96 用原池（与旧结果逐位可比），更大时补两字组合。"""
    e = list(dict.fromkeys(_E_POOL))
    if len(e) < n:
        e = e + _spread_combos(list(dict.fromkeys(_E_POOL_EXT2)), n - len(e))
    nm = list(dict.fromkeys(_N_POOL))
    if len(nm) < n:
        nm = nm + _spread_combos(list(dict.fromkeys(_N_POOL_EXT2)), n - len(nm))
    e, nm = e[:n], nm[:n]
    assert len(e) == n and len(nm) == n, f"池不够：E{len(e)} N{len(nm)} < {n}"
    return e, nm


def selftest(n: int = 40, seed: int = 0, out: str = "", nunknown: int = 20,
             multi: bool = False, write_mode: str = "grad",
             key_mode: str = "entity", lam: float = 1e-2,
             quiet: bool = False, m: int = 256) -> Dict[str, object]:
    """端到端验收：教 n 条（未覆盖名称）-> 逐条提问；再问 nunknown 条没教过的。"""
    ents, names = _pools(n)
    all_ents, _ = _pools(n + nunknown)
    if multi:
        names = [nm + "子" for nm in names]          # 多 token 名称
    qs = [QTPL.format(q=q) for q in ents]
    unknown = [QTPL.format(q=q) for q in all_ents[n:n + nunknown]]
    rt = TowerRuntime(seed=seed, m=m, write_mode=write_mode, key_mode=key_mode,
                      lam=lam, verbose=not quiet).build()
    rt.teach_many(list(zip(qs, names)))
    t0 = time.time()
    rt.commit()
    commit_s = time.time() - t0
    hit, mg, ad = 0, [], []
    for q, nm in zip(qs, names):
        p = rt.probe(q, "dict")
        hit += int(p["answer"] == nm)
        mg.append(float(p["margin"])); ad.append(float(p["addr"]))
    umg, uad = [], []
    for q in unknown:
        p = rt.probe(q, "dict")
        umg.append(float(p["margin"])); uad.append(float(p["addr"]))
    # 全词表解码只出一个 token -> 多 token 名称不适用
    vhit = (None if multi else
            sum(int(rt.probe(q, "vocab")["answer"] == nm) for q, nm in zip(qs, names)))
    mg, ad, umg, uad = sorted(mg), sorted(ad), sorted(umg), sorted(uad)
    res = {"n": n, "multi": multi, "acc": hit / n,
           "vocab_acc": (None if vhit is None else vhit / n),
           "margin_known_min": mg[0], "margin_unknown_max": umg[-1],
           "addr_known_min": ad[0], "addr_unknown_max": uad[-1],
           "margin_known_med": mg[len(mg) // 2], "addr_known_med": ad[len(ad) // 2],
           "commit_s": round(commit_s, 3),
           "config": rt.info()}
    print(f"  严格精确（名称）      {hit}/{n} = {res['acc']:.3f}")
    print("  全词表解码（无字典）  " + ("不适用（多 token 名称）" if vhit is None
                                        else f"{vhit}/{n} = {vhit/n:.3f}"))
    print(f"  读出边际  已知最小 {mg[0]:+.3f} · 未知最大 {umg[-1]:+.3f}"
          f" {'✅ 可分' if mg[0] > umg[-1] else '⚠ 重叠'}")
    print(f"  键寻址    已知最小 {ad[0]:+.3f} · 未知最大 {uad[-1]:+.3f}"
          f" {'✅ 可分' if ad[0] > uad[-1] else '⚠ 重叠'}")
    if out:
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  -> {out}")
    return res


def demo() -> None:
    rt = TowerRuntime().build()
    facts = [(QTPL.format(q="甲"), "锇"), (QTPL.format(q="乙"), "铱"),
             (QTPL.format(q="丙"), "镅")]
    rt.teach_many(facts).commit()
    for q, nm in facts:
        p = rt.probe(q)
        print(f"  问：{q}\n  答：{p['answer']}（教的是 {nm}）"
              f" 边际 {p['margin']:+.3f} 寻址 {p['addr']:+.3f}")
    q = QTPL.format(q="丁")
    p = rt.probe(q)
    print(f"  未教过的箱子丁 -> 最像「{p['answer']}」但 边际 {p['margin']:+.3f} / "
          f"寻址 {p['addr']:+.3f}（阈值 边际 {rt.margin_thr:.2f} / 寻址 {rt.addr_thr:.2f}）")


def persist_check(write_mode: str = "grad") -> bool:
    """持久化自检：教 3 条 -> save -> 新实例 load -> 查询 + 未教过的仍拒绝。"""
    import tempfile
    rt = TowerRuntime(verbose=False, write_mode=write_mode).build()
    facts = [(QTPL.format(q="甲"), "锇"), (QTPL.format(q="乙"), "铱"),
             (QTPL.format(q="丙"), "镅")]
    rt.teach_many(facts).commit()
    path = os.path.join(tempfile.gettempdir(), f"tower_persist_{write_mode}.pt")
    rt.save(path)
    rt2 = TowerRuntime(verbose=False, write_mode=write_mode).build().load(path)
    ok = all(rt2.query(q) == nm for q, nm in facts)
    ok2 = rt2.query(QTPL.format(q="丁")) is None
    print(f"  持久化[{write_mode}]：save/load 后 3/3 正确 = {ok} · 未教实体仍拒绝 = {ok2}")
    return bool(ok and ok2)


def main() -> None:
    ap = argparse.ArgumentParser(description="记忆塔运行时")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--persist", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nunknown", type=int, default=20)
    ap.add_argument("--multi", action="store_true", help="多 token 名称")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--write", choices=("grad", "closed"), default="grad",
                    help="写入方式：grad=R3s20_fwd 梯度写；closed=岭回归闭式写")
    ap.add_argument("--key-mode", choices=("entity", "bind", "hrr"), default="entity",
                    help="键构造：entity=只用实体；bind=实体+属性")
    ap.add_argument("--lam", type=float, default=1e-2, help="闭式写的岭参数")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    if args.selftest:
        print("=" * 96)
        print(f"**记忆塔运行时 · 端到端验收**（未覆盖名称 · 槽 {args.n}"
              f"{' · 多 token 名称' if args.multi else ''}"
              f" · 写入 {args.write} · 键 {args.key_mode}）")
        print("=" * 96)
        selftest(args.n, args.seed, args.out, args.nunknown, args.multi,
                 args.write, args.key_mode, args.lam)
        return
    if args.demo:
        demo(); return
    if args.build:
        TowerRuntime(m=args.m).build(force=True); return
    if args.persist:
        persist_check(args.write); return
    ap.print_help()


if __name__ == "__main__":
    main()

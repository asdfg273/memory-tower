"""相册一致性 A 阶段：把参考图的中层 token 经"塔"或"裸拼接"送进 SD1.5，比较身份稳定性。

预注册判据见 reports/ALBUM_TOWER.md（H1/H2/H3，跑数字前写死）。本脚本不训练任何东西。

用法（256×256、6 张、5 臂，单卡约 2-4 min）：
    venv\\Scripts\\python.exe tools_album_tower.py --quick
    venv\\Scripts\\python.exe tools_album_tower.py --size 256 --n 6 --out _ab_tmp\\album

指标：CLIP 图像余弦（身份）、两两像素距离（多样性）、CLIP 文-图余弦（指令遵循）、耗时。
自检：copy 臂（img2img strength=0.15）身份应接近 1.0；无关角色交叉分应显著更低。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Sequence

import torch
from PIL import Image

SD15 = r"_ab_tmp\sd15"
CLIP_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "models--openai--clip-vit-base-patch32", "snapshots",
)

SCENES = [
    "a portrait of the character, studio light, neutral background",
    "the character walking in a rainy city street, full body",
    "the character sitting in a cafe by the window, side light",
    "the character seen from behind on a rooftop at sunset",
    "the character smiling, close-up, outdoors, daylight",
    "the character running, motion blur, side view",
]
CHAR = "sks person, short black hair, blue jacket"   # 伪"角色 token"，SD1.5 不认识，靠图像条件
CHAR_B = "sks man, long white beard, red coat"       # 无关角色，用来做指标自检

REF_PROMPT = f"portrait photo of {CHAR}, looking at camera, neutral background"
REF_PROMPT_B = f"portrait photo of {CHAR_B}, looking at camera, neutral background"

MID_TS = 300            # 抽参考 token 用的噪声时刻
LAM = 1e-2              # 塔的岭回归正则
TOWER_W = 0.5           # 塔残差强度


# ---------------------------------------------------------------- 模型

def load_pipe(device: str = "cuda"):
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        SD15, dtype=torch.float16, variant="fp16",
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_clip(device: str = "cuda"):
    from transformers import CLIPImageProcessor, CLIPModel

    snaps = sorted(os.listdir(CLIP_DIR)) if os.path.isdir(CLIP_DIR) else []
    path = os.path.join(CLIP_DIR, snaps[-1]) if snaps else "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(path, local_files_only=True).to(device).eval()
    # 本地缓存里没有 preprocessor_config.json；默认参数与官方一致（224 crop / CLIP 均值方差）
    proc = CLIPImageProcessor()
    return model, proc


@torch.no_grad()
def clip_image_emb(model, proc, imgs: Sequence[Image.Image], device="cuda") -> torch.Tensor:
    px = proc(images=list(imgs), return_tensors="pt")["pixel_values"].to(device)
    # transformers 5.x 的 get_image_features 返回容器对象，这里走底层等价路径
    out = model.vision_model(pixel_values=px)
    feat = model.visual_projection(out.pooler_output)
    return feat / feat.norm(dim=-1, keepdim=True)


@torch.no_grad()
def clip_text_emb(model, tokenizer, prompts: Sequence[str], device="cuda") -> torch.Tensor:
    batch = tokenizer(list(prompts), return_tensors="pt", padding="max_length",
                      truncation=True, max_length=77).to(device)
    out = model.text_model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"])
    feat = model.text_projection(out.pooler_output)
    return feat / feat.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------- 抽参考 token

class TokenTap:
    """挂一个 forward_pre_hook 抓 mid-block 自注意力的输入 token。"""

    def __init__(self, pipe):
        self.attn = pipe.unet.mid_block.attentions[0].transformer_blocks[0].attn1
        self.captured: torch.Tensor | None = None
        self._h = self.attn.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        if hs is not None and hs.dim() == 3:
            self.captured = hs.detach()
        return None

    def close(self):
        self._h.remove()


@torch.no_grad()
def ref_tokens(pipe, img: Image.Image, prompt: str, size: int, device="cuda") -> torch.Tensor:
    """把参考图在噪声 t=MID_TS 处过一遍 UNet，返回 mid-block 输入 token（N,D），单位化。"""
    px = pipe.image_processor.preprocess(img, height=size, width=size)
    vd = pipe.vae.dtype                      # VAE 可能被 diffusers 钉在 float32
    lat = pipe.vae.encode(px.to(device, vd)).latent_dist.mode() * 0.18215
    g = torch.Generator(device=device).manual_seed(0)
    noise = torch.randn(lat.shape, generator=g, device=device, dtype=lat.dtype)
    t = torch.tensor([MID_TS], device=device)
    lat_t = pipe.scheduler.add_noise(lat, noise, t)
    emb, _ = pipe.encode_prompt(prompt, device, 1, True)
    if emb.shape[0] > 1:                # 有的版本返回 [负向; 正向] 拼接
        emb = emb[-1:]
    tap = TokenTap(pipe)
    pipe.unet(lat_t, MID_TS, encoder_hidden_states=emb)
    tap.close()
    if tap.captured is None:
        raise RuntimeError("没有抓到 mid-block token")
    toks = tap.captured[0].to(torch.float32)          # (N, D)
    return toks / toks.norm(dim=-1, keepdim=True).clamp_min(1e-6)


# ---------------------------------------------------------------- 注入器

class Injector:
    """在 mid-block 自注意力上做两种注入：
    raw   —— 把参考 token 拼到序列尾部（注意力长度 N+R）
    tower —— 闭式岭回归读出，作为残差加在图像 token 上（长度不变）
    """

    def __init__(self, pipe, arm: str, ref: torch.Tensor):
        self.arm = arm
        self.ref = ref.contiguous()          # (R, D) float32
        self.attn = pipe.unet.mid_block.attentions[0].transformer_blocks[0].attn1
        self.n_orig = 0
        self._hs = self.attn.register_forward_pre_hook(self._pre)
        self._hp = self.attn.register_forward_hook(self._post) if arm == "raw" else None

    # 塔：r = (q K^T) (K K^T + λ I)^{-1} V   —— 双对偶形式，避免 1280×1280 求逆
    # fp16 下 linalg.solve 未实现，统一在 float32 里做闭式读出
    def _mem_read(self, q: torch.Tensor) -> torch.Tensor:
        dtype = q.dtype
        shape = q.shape
        q32 = q.reshape(-1, shape[-1]).to(torch.float32)   # (B*N, D)
        K = self.ref.to(q.device, torch.float32)           # (R,D)
        g = q32 @ K.t()                                    # (B*N,R)
        gram = K @ K.t() + LAM * torch.eye(K.shape[0], device=q.device, dtype=torch.float32)
        coef = torch.linalg.solve(gram, g.t()).t()         # (B*N,R)
        return (coef @ K).reshape(shape).to(dtype)

    def _pre(self, module, args):
        hs = args[0]
        if hs.dim() != 3:
            return None
        if self.arm == "raw":
            self.n_orig = hs.shape[1]
            ref = self.ref.to(hs.device, hs.dtype).unsqueeze(0).expand(hs.shape[0], -1, -1)
            return (torch.cat([hs, ref], dim=1),) + tuple(args[1:])
        if self.arm in ("tower", "writeback"):
            return (hs + TOWER_W * self._mem_read(hs),) + tuple(args[1:])
        return None

    def _post(self, module, args, output):
        if self.arm == "raw" and self.n_orig:
            return output[:, : self.n_orig]
        return None

    def close(self):
        self._hs.remove()
        if self._hp is not None:
            self._hp.remove()


# ---------------------------------------------------------------- 生成与度量

def gen(pipe, prompt: str, seed: int, size: int) -> Image.Image:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(prompt, num_inference_steps=20, guidance_scale=7.5,
                height=size, width=size, generator=g).images[0]


def gen_img2img(i2i, prompt: str, seed: int, size: int, init: Image.Image,
                strength: float) -> Image.Image:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return i2i(prompt, image=init.resize((size, size)), strength=strength,
               num_inference_steps=20, guidance_scale=7.5, generator=g).images[0]


def diversity(imgs: Sequence[Image.Image]) -> float:
    if len(imgs) < 2:
        return 0.0
    px = [torch.frombuffer(im.resize((64, 64)).convert("RGB").tobytes(), dtype=torch.uint8)
          .view(64, 64, 3).float() for im in imgs]
    tot, cnt = 0.0, 0
    for i in range(len(px)):
        for j in range(i + 1, len(px)):
            tot += float((px[i] - px[j]).abs().mean())
            cnt += 1
    return tot / max(cnt, 1)


def run_arm(ctx: Dict[str, object], arm: str, size: int, n: int, out_dir: str) -> Dict[str, object]:
    pipe, clipm, clipp, tok = ctx["pipe"], ctx["clip"], ctx["clip_proc"], ctx["tokenizer"]
    ref_img: Image.Image = ctx["ref"]
    ref_emb = clip_image_emb(clipm, clipp, [ref_img])[0]

    inj = None
    ref_tok = None
    if arm in ("raw", "tower", "writeback"):
        ref_tok = ref_tokens(pipe, ref_img, REF_PROMPT, size)
        inj = Injector(pipe, arm, ref_tok)

    imgs: List[Image.Image] = []
    prompts: List[str] = []
    t0 = time.time()
    for k in range(n):
        prompt = f"{CHAR}, {SCENES[k % len(SCENES)]}"
        prompts.append(prompt)
        if arm == "none":
            im = gen(pipe, prompt, 1000 + k, size)
        elif arm == "copy":
            im = gen_img2img(ctx["i2i"], prompt, 1000 + k, size, ref_img, 0.15)
        elif arm == "img2img":
            im = gen_img2img(ctx["i2i"], prompt, 1000 + k, size, ref_img, 0.60)
        else:
            im = gen(pipe, prompt, 1000 + k, size)
        imgs.append(im)
        if arm == "writeback":
            # 用户提的"信息流返回中部"：把这张图自己的中层特征写回塔（只留最近 3 倍）
            new_tok = ref_tokens(pipe, im, prompt, size)
            inj.ref = torch.cat([inj.ref, new_tok], dim=0)[-(len(ref_tok) * 3):]
        im.save(os.path.join(out_dir, f"{arm}_{k}.png"))
    dt = (time.time() - t0) / n

    embs = clip_image_emb(clipm, clipp, imgs)
    txt = clip_text_emb(clipm, tok, prompts)
    id_cos = [float((e * ref_emb).sum()) for e in embs]
    clip_t = [float((e * t_).sum()) for e, t_ in zip(embs, txt)]
    if inj is not None:
        inj.close()
    return {
        "arm": arm,
        "id_cos_mean": round(sum(id_cos) / len(id_cos), 4),
        "id_cos": [round(c, 4) for c in id_cos],
        "clip_t_mean": round(sum(clip_t) / len(clip_t), 4),
        "diversity_l1": round(diversity(imgs), 2),
        "sec_per_img": round(dt, 2),
        "n_ref_tokens": 0 if ref_tok is None else int(len(ref_tok)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--arms", default="none,copy,img2img,raw,tower,writeback")
    ap.add_argument("--out", default=r"_ab_tmp\album")
    ap.add_argument("--quick", action="store_true", help="只跑 none/copy/raw，2 张")
    args = ap.parse_args(argv)
    if args.quick:
        args.n, args.arms = 2, "none,copy,raw"

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    pipe = load_pipe()
    clipm, clipp = load_clip()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(SD15, "tokenizer"))
    from diffusers import StableDiffusionImg2ImgPipeline
    i2i = StableDiffusionImg2ImgPipeline.from_pipe(pipe)
    print(f"[load] {time.time() - t0:.1f}s, arms={args.arms}, n={args.n}", flush=True)

    g = torch.Generator(device="cuda").manual_seed(7)
    ref = gen(pipe, REF_PROMPT, 7, args.size)
    ref.save(os.path.join(args.out, "reference.png"))
    ref_b = gen(pipe, REF_PROMPT_B, 11, args.size)
    ref_b.save(os.path.join(args.out, "reference_other.png"))

    # 指标自检：同一角色 vs 无关角色
    self_check = {}
    e_ref, e_other = clip_image_emb(clipm, clipp, [ref, ref_b])
    self_check["identity_same_image"] = round(float((e_ref * e_ref).sum()), 4)
    self_check["identity_other_char"] = round(float((e_ref * e_other).sum()), 4)
    print("[selfcheck]", json.dumps(self_check, ensure_ascii=False), flush=True)

    ctx = {"pipe": pipe, "clip": clipm, "clip_proc": clipp, "tokenizer": tok,
           "ref": ref, "i2i": i2i}
    results = []
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        r = run_arm(ctx, arm, args.size, args.n, args.out)
        print("[arm]", json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)

    summary = {"size": args.size, "n": args.n, "mid_ts": MID_TS, "lam": LAM,
               "tower_weight": TOWER_W, "chunk": CHAR, "selfcheck": self_check,
               "results": results}
    with open(os.path.join(args.out, "album_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("[summary_json] " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

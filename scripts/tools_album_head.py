"""A′ 阶段：训练一个小投影头（连接器），把参考图的中层 token 变成 SD1.5 的 cross-attn 上下文。

预注册判据（跑数字前写死，见 reports/ALBUM_TOWER.md「A′」）：
  H4：留出身份（训练时没见过）的 CLIP-I ≥ 0.80（介于 none 0.688 与 img2img 0.899 之间），
      且多样性 ≥ 40（不能塌成 12 张复制图）。两条都满足才算投影头学会了"认人"而不是"抄图"。

结构（总参数 ≈ 3.3M，只有它可训练）：
  ref 的 64 个 mid-block token (1280) → Linear(1280→768)+GELU → 均值池化 → Linear(768→4*768)
  → 4 个 cross-attn 上下文 token，拼在 77 个文本 token 之后。

训练监督：cond-eps 预测（标准扩散损失），目标图来自 img2img(ref, scene, 0.8)
—— 也就是"身份由参考图带过来、构图由场景词决定"的那批图。数据只有 2 个训练身份 × 6 场景。

用法：
    venv\\Scripts\\python.exe tools_album_head.py --stage all --steps 400
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from PIL import Image

import tools_album_tower as TA

CACHE = r"_ab_tmp\album_head"
HEAD_PATH = os.path.join(CACHE, "head.pt")
DATA_PATH = os.path.join(CACHE, "data.pt")
OUT_DIR = r"_ab_tmp\album_head_eval"

TRAIN_IDS = ["A", "B"]
EVAL_ID = "C"
N_SCENES = 6
N_CTX = 4                 # 追加的上下文 token 数
HID = 768
STEPS_DEFAULT = 400
LR = 1e-3
MIM_STRENGTH = 0.80       # 造训练目标时 img2img 的强度
SEED_BASE = 700


def id_prompt(key: str) -> str:
    return {
        "A": "portrait photo of sks person, short black hair, blue jacket, looking at camera, neutral background",
        "B": "portrait photo of sks man, long white beard, red coat, looking at camera, neutral background",
        "C": "portrait photo of sks woman, curly red hair, green dress, looking at camera, neutral background",
    }[key]


class Head(nn.Module):
    def __init__(self, d_in: int = 1280, d_hid: int = HID, n_ctx: int = N_CTX, d_out: int = 768):
        super().__init__()
        self.proj = nn.Linear(d_in, d_hid)
        self.out = nn.Linear(d_hid, n_ctx * d_out)
        self.n_ctx, self.d_out = n_ctx, d_out

    def forward(self, toks: torch.Tensor) -> torch.Tensor:
        """toks: (R, 1280) → (n_ctx, 768)"""
        h = torch.nn.functional.gelu(self.proj(toks))      # (R, 768)
        pooled = h.mean(dim=0)                             # (768,)
        return self.out(pooled).view(self.n_ctx, self.d_out)


# ---------------------------------------------------------------- 数据

@torch.no_grad()
def build_dataset(pipe, clip, size: int) -> Dict[str, object]:
    os.makedirs(CACHE, exist_ok=True)
    from diffusers import StableDiffusionImg2ImgPipeline
    i2i = StableDiffusionImg2ImgPipeline.from_pipe(pipe)

    data: Dict[str, object] = {"size": size, "ids": {}, "scenes": TA.SCENES[:N_SCENES]}
    for key in TRAIN_IDS + [EVAL_ID]:
        ref_prompt = id_prompt(key)
        g = torch.Generator(device="cuda").manual_seed(SEED_BASE + ord(key))
        ref = TA.gen(pipe, ref_prompt, SEED_BASE + ord(key) + 1, size)
        ref.save(os.path.join(CACHE, f"ref_{key}.png"))
        toks = TA.ref_tokens(pipe, ref, ref_prompt, size)      # (64,1280) 单位化
        items = []
        for si, scene in enumerate(TA.SCENES[:N_SCENES]):
            prompt = scene                                    # 身份由参考图（img2img 初值）带进来
            gimg = torch.Generator(device="cuda").manual_seed(SEED_BASE + si)
            tgt = i2i(prompt, image=ref.resize((size, size)), strength=MIM_STRENGTH,
                      num_inference_steps=20, guidance_scale=7.5, generator=gimg).images[0]
            tgt.save(os.path.join(CACHE, f"tgt_{key}_{si}.png"))
            px = pipe.image_processor.preprocess(tgt, height=size, width=size)
            with torch.no_grad():
                lat = pipe.vae.encode(px.to("cuda", pipe.vae.dtype)).latent_dist.mode() * 0.18215
            emb, _ = pipe.encode_prompt(prompt, "cuda", 1, False)
            items.append({"scene": prompt, "lat": lat.half().cpu(), "emb": emb.half().cpu()})
        data["ids"][key] = {"ref_tokens": toks.half().cpu(), "items": items}
        print(f"[data] {key}: ref + {len(items)} targets", flush=True)

    torch.save(data, DATA_PATH)
    return data


# ---------------------------------------------------------------- 训练

def train_head(pipe, data: Dict[str, object], steps: int) -> Head:
    unet, vae = pipe.unet, pipe.vae
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    unet.train()
    try:
        unet.enable_gradient_checkpointing()
    except Exception as exc:                                  # pragma: no cover
        print("[warn] gradient checkpointing 不可用:", exc, flush=True)

    head = Head().to("cuda", torch.float32)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.0)

    pool: List[Dict[str, object]] = []
    for key in TRAIN_IDS:
        d = data["ids"][key]
        for it in d["items"]:
            pool.append({"ref": d["ref_tokens"].to("cuda", torch.float32),
                         "lat": it["lat"].to("cuda"), "emb": it["emb"].to("cuda")})
    print(f"[train] {len(pool)} pairs, {steps} steps", flush=True)

    t0 = time.time()
    losses: List[float] = []
    for step in range(steps):
        item = pool[step % len(pool)]
        lat = item["lat"].to(torch.float16)
        emb = item["emb"].to(torch.float16)                       # (1,77,768)
        ctx = head(item["ref"]).to(torch.float16).unsqueeze(0)    # (1,4,768)
        cond = torch.cat([emb, ctx], dim=1)                       # (1,81,768)

        noise = torch.randn_like(lat)
        t = torch.randint(0, 1000, (1,), device="cuda").long()
        lat_t = pipe.scheduler.add_noise(lat, noise, t)
        pred = unet(lat_t, t, encoder_hidden_states=cond).sample
        loss = torch.nn.functional.mse_loss(pred.float(), noise.float())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0 or step == steps - 1:
            print(f"[step {step}] loss {sum(losses[-50:]) / len(losses[-50:]):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    unet.requires_grad_(True)
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, HEAD_PATH)
    print(f"[train] done in {time.time() - t0:.0f}s → {HEAD_PATH}", flush=True)
    return head


# ---------------------------------------------------------------- 评估

@torch.no_grad()
def generate_with_head(pipe, head: Head, ref_tokens: torch.Tensor, prompt: str,
                       seed: int, size: int) -> Image.Image:
    # 这个 diffusers 版本在 negative_prompt=None 时只返回正向；负向单独编码
    pos, _ = pipe.encode_prompt(prompt, "cuda", 1, False)
    neg, _ = pipe.encode_prompt("", "cuda", 1, False)
    ctx = head(ref_tokens.to("cuda", torch.float32)).to(pos.dtype).unsqueeze(0)
    pos = torch.cat([pos, ctx], dim=1)
    neg = torch.cat([neg, ctx], dim=1)
    g = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(prompt_embeds=pos, negative_prompt_embeds=neg,
                num_inference_steps=20, guidance_scale=7.5,
                height=size, width=size, generator=g).images[0]


@torch.no_grad()
def evaluate(pipe, clip, head: Head, data: Dict[str, object], size: int,
             keys: Sequence[str]) -> List[Dict[str, object]]:
    os.makedirs(OUT_DIR, exist_ok=True)
    clipm, clipp = clip
    results = []
    for key in keys:
        d = data["ids"][key]
        ref_img = Image.open(os.path.join(CACHE, f"ref_{key}.png"))
        ref_emb = TA.clip_image_emb(clipm, clipp, [ref_img])[0]
        imgs, prompts = [], []
        for si, scene in enumerate(TA.SCENES[:N_SCENES]):
            im = generate_with_head(pipe, head, d["ref_tokens"], scene, SEED_BASE + si, size)
            im.save(os.path.join(OUT_DIR, f"{key}_{si}.png"))
            imgs.append(im)
            prompts.append(scene)
        embs = TA.clip_image_emb(clipm, clipp, imgs)
        id_cos = [float((e * ref_emb).sum()) for e in embs]
        results.append({"id": key, "id_cos_mean": round(sum(id_cos) / len(id_cos), 4),
                        "id_cos": [round(c, 4) for c in id_cos],
                        "diversity_l1": round(TA.diversity(imgs), 2)})
        print("[eval]", json.dumps(results[-1], ensure_ascii=False), flush=True)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "build", "train", "eval"])
    ap.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args(argv)

    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    pipe = TA.load_pipe()
    clip = TA.load_clip()
    print(f"[load] {time.time() - t0:.1f}s", flush=True)

    if args.stage in ("all", "build") or not os.path.exists(DATA_PATH):
        data = build_dataset(pipe, clip, args.size)
    else:
        data = torch.load(DATA_PATH, weights_only=False)

    summary: Dict[str, object] = {"size": args.size, "steps": args.steps, "n_ctx": N_CTX,
                                  "mim_strength": MIM_STRENGTH, "train_ids": TRAIN_IDS,
                                  "eval_id": EVAL_ID}
    if args.stage in ("all", "train"):
        head = train_head(pipe, data, args.steps)
        summary["train_seconds"] = None
    if args.stage in ("all", "eval"):
        head = Head().to("cuda")
        head.load_state_dict(torch.load(HEAD_PATH, weights_only=True))
        head.eval()
        res = evaluate(pipe, clip, head, data, args.size, [EVAL_ID, TRAIN_IDS[0]])
        summary["results"] = res

    with open(os.path.join(CACHE, "head_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("[summary_json] " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


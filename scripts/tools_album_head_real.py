"""A″ 阶段：用真实人脸数据（CelebA）训练连接器，检验"数据量是否能让头不塌"。

与 A′ 的差别：
  - 数据从"合成 12 对"换成 CelebA 真人照片（同一人 6 张以上，跨姿态/光线/造型）；
  - 头从 3.3M 单层换成 ~9M 两层、8 个 context token；
  - 评估同时跑 `none` 基线（纯文本、无身份信号），把所有增益归给连接器。

阶段：prep → build → train → eval

    venv\\Scripts\\python.exe tools_album_head_real.py --stage prep
    venv\\Scripts\\python.exe tools_album_head_real.py --stage build
    venv\\Scripts\\python.exe tools_album_head_real.py --stage train --steps 2000
    venv\\Scripts\\python.exe tools_album_head_real.py --stage eval

预注册判据（先于数字写下）：
  H5：留出身份上，连接器 CLIP-I 比 `none` 高 ≥ 0.10 且多样性 ≥ 40；
  H6：训练身份上，连接器 CLIP-I 比 `none` 高 ≥ 0.15（先证明能拟合，再谈泛化）。
  两者任一不成立，就说明"小头 + 冻结模型"这条路在可承受的数据量内不通。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

import tools_album_tower as TA   # 复用 pipe/CLIP/ref_tokens/多样性工具

ROOT = r"_ab_tmp\celeba"
DATA = r"_ab_tmp\album_head_real"
HEAD_PATH = os.path.join(DATA, "head_real.pt")
PARQUET = r"_ab_tmp\celeba_test0.parquet"

N_TRAIN_IDS = 4
N_HELDOUT_IDS = 2
N_IMGS = 8                 # 每个身份取的图数（1 张当 ref，其余当监督目标）
N_CTX = 8
EVAL_SCENES = [
    "a photo of a person walking in a rainy city street, full body",
    "a photo of a person sitting in a cafe by the window",
    "a photo of a person standing on a beach at sunset",
    "a photo of a person in a snowy forest, wearing a thick coat",
]
PROMPT_NEUTRAL = "a photo of a person"


# ---------------------------------------------------------------- 头

class Head2(torch.nn.Module):
    """1280 → 768 → 均值池化 → 两层 MLP → N_CTX 个 context token。"""

    def __init__(self, d_in: int = 1280, d_hid: int = 768, n_ctx: int = N_CTX):
        super().__init__()
        self.n_ctx = n_ctx
        self.proj = torch.nn.Sequential(torch.nn.Linear(d_in, d_hid), torch.nn.GELU())
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_hid, d_hid), torch.nn.GELU(),
            torch.nn.Linear(d_hid, d_hid), torch.nn.GELU(),
            torch.nn.Linear(d_hid, n_ctx * d_hid),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.proj(tokens).mean(dim=0)                  # (D,)
        return self.mlp(h).view(self.n_ctx, -1)            # (n_ctx, D)


# ---------------------------------------------------------------- prep

def stage_prep(size: int = 512, n_rows: int = 700) -> Dict[str, object]:
    """流式读取 CelebA 前 n_rows 行（影像按 celeb_id 排序，前几百行就有几十个身份）。"""
    from collections import Counter

    import io

    from datasets import load_dataset

    os.makedirs(ROOT, exist_ok=True)
    ds = load_dataset("parquet",
                      data_files={"test": "hf://datasets/flwrlabs/celeba/"
                                           "img_align+identity+attr/test-00000-of-00003.parquet"},
                      split="test", streaming=True)
    buckets: Dict[str, List[bytes]] = {}
    seen = 0
    for row in ds:
        seen += 1
        key = str(row.get("celeb_id"))
        b = buckets.setdefault(key, [])
        if len(b) < N_IMGS:
            buf = io.BytesIO()
            row["image"].save(buf, format="PNG")
            b.append(buf.getvalue())
        if seen >= n_rows:
            break
    print(f"[prep] rows={seen} identities={len(buckets)}", flush=True)

    usable = [(k, v) for k, v in sorted(buckets.items()) if len(v) >= N_IMGS]
    picked = usable[: N_TRAIN_IDS + N_HELDOUT_IDS]
    print(f"[prep] usable(>= {N_IMGS} imgs)={len(usable)}, picked={len(picked)}", flush=True)
    if len(picked) < N_TRAIN_IDS + N_HELDOUT_IDS:
        raise SystemExit(f"可用身份只有 {len(picked)} 个，不够")

    manifest: Dict[str, object] = {"train_ids": [], "heldout_ids": [], "files": {}, "size": size}
    for i, (ident, blobs) in enumerate(picked):
        split = "train" if i < N_TRAIN_IDS else "heldout"
        key = f"{split}_{i}"
        files = []
        for k, blob in enumerate(blobs):
            path = os.path.join(ROOT, f"{key}_{k}.png")
            Image.open(io.BytesIO(blob)).convert("RGB").resize((size, size)).save(path)
            files.append(path)
        manifest["files"][key] = files
        (manifest["train_ids"] if split == "train" else manifest["heldout_ids"]).append(key)
        print(f"[prep] {key} <- celeb_id {ident}, {len(files)} imgs", flush=True)

    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[prep] train={manifest['train_ids']} heldout={manifest['heldout_ids']}", flush=True)
    return manifest


# ---------------------------------------------------------------- build

@torch.no_grad()
def stage_build(pipe, manifest: Dict[str, object], size: int) -> Dict[str, object]:
    from diffusers import StableDiffusionImg2ImgPipeline
    i2i = StableDiffusionImg2ImgPipeline(**pipe.components)
    data = {"mim_strength": 0.75, "ids": {}}
    os.makedirs(DATA, exist_ok=True)
    for key, files in manifest["files"].items():
        ref_img = Image.open(files[0]).convert("RGB")
        ref_tokens = TA.ref_tokens(pipe, ref_img, PROMPT_NEUTRAL, size)
        items = []
        for tgt_path in files[1:]:
            tgt_img = Image.open(tgt_path).convert("RGB")
            prompt = PROMPT_NEUTRAL
            g = torch.Generator(device="cuda").manual_seed(hash(tgt_path) % 10_000)
            target = i2i(prompt, image=tgt_img, strength=data["mim_strength"],
                         num_inference_steps=20, guidance_scale=7.5, generator=g).images[0]
            px = pipe.image_processor.preprocess(target, height=size, width=size)
            lat = pipe.vae.encode(px.to("cuda", torch.float16)).latent_dist.mode() * 0.18215
            emb, _ = pipe.encode_prompt(prompt, "cuda", 1, False)
            items.append({"scene": tgt_path, "lat": lat.half().cpu(), "emb": emb.half().cpu()})
            target.save(os.path.join(DATA, os.path.basename(tgt_path).replace(".png", "_tgt.png")))
        data["ids"][key] = {"ref_tokens": ref_tokens.cpu(),
                            "ref_img": files[0], "files": files, "items": items}
        print(f"[build] {key}: ref + {len(items)} targets", flush=True)
    torch.save(data, os.path.join(DATA, "data_real.pt"))
    print("[build] saved", os.path.join(DATA, "data_real.pt"), flush=True)
    return data


# ---------------------------------------------------------------- train

def stage_train(pipe, data: Dict[str, object], steps: int, lr: float = 1e-3) -> Head2:
    unet = pipe.unet
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    head = Head2().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)

    pool = []
    for key in data["train_keys"]:
        d = data["ids"][key]
        for it in d["items"]:
            pool.append({"ref": d["ref_tokens"].cuda().float(),
                         "lat": it["lat"].cuda(), "emb": it["emb"].cuda()})
    print(f"[train] {len(pool)} pairs, {steps} steps, params={sum(p.numel() for p in head.parameters())}", flush=True)

    t0 = time.time()
    hist: List[float] = []
    for step in range(steps):
        item = pool[step % len(pool)]
        lat = item["lat"].to(torch.float16)
        emb = item["emb"].to(torch.float16)
        ctx = head(item["ref"]).to(torch.float16).unsqueeze(0)
        cond = torch.cat([emb, ctx], dim=1)
        noise = torch.randn_like(lat)
        t = torch.randint(0, 1000, (1,), device="cuda").long()
        lat_t = pipe.scheduler.add_noise(lat, noise, t)
        pred = unet(lat_t, t, encoder_hidden_states=cond).sample
        loss = F.mse_loss(pred.float(), noise.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        hist.append(float(loss.detach()))
        if step % 100 == 0 or step == steps - 1:
            print(f"[step {step}] loss {sum(hist[-100:]) / len(hist[-100:]):.4f} ({time.time() - t0:.0f}s)", flush=True)

    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, HEAD_PATH)
    print("[train] saved", HEAD_PATH, flush=True)
    return head


# ---------------------------------------------------------------- eval

@torch.no_grad()
def generate(pipe, head, ref_tokens, scene: str, seed: int, size: int) -> Image.Image:
    pos, _ = pipe.encode_prompt(scene, "cuda", 1, False)
    neg, _ = pipe.encode_prompt("", "cuda", 1, False)
    if head is not None:
        ctx = head(ref_tokens.cuda().float()).to(pos.dtype).unsqueeze(0)
        pos, neg = torch.cat([pos, ctx], 1), torch.cat([neg, ctx], 1)
    g = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(prompt_embeds=pos, negative_prompt_embeds=neg,
                num_inference_steps=20, guidance_scale=7.5,
                height=size, width=size, generator=g).images[0]


@torch.no_grad()
def stage_eval(pipe, clip, manifest, head: Head2 | None, size: int) -> List[Dict[str, object]]:
    clipm, clipp = clip
    out_dir = os.path.join(DATA, "eval")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for key, files in manifest["files"].items():
        ref_img = Image.open(files[0]).convert("RGB")
        ref_tokens = TA.ref_tokens(pipe, ref_img, PROMPT_NEUTRAL, size)
        # 身份参照用"训练时当 ref 那张之外"的真实照片，避免同一张图的自匹配
        idimgs = [Image.open(p).convert("RGB") for p in files[1:4]]
        idemb = TA.clip_image_emb(clipm, clipp, idimgs).mean(0, keepdim=True)
        idemb = idemb / idemb.norm(dim=-1, keepdim=True)

        for tag, hd in (("head", head), ("none", None)):
            imgs = []
            for si, scene in enumerate(EVAL_SCENES):
                im = generate(pipe, hd, ref_tokens, scene, 4000 + si, size)
                im.save(os.path.join(out_dir, f"{key}_{tag}_{si}.png"))
                imgs.append(im)
            embs = TA.clip_image_emb(clipm, clipp, imgs)
            cos = [float((e * idemb[0]).sum()) for e in embs]
            row = {"id": key, "arm": tag, "clip_i": round(sum(cos) / len(cos), 4),
                   "per_image": [round(c, 4) for c in cos],
                   "diversity": round(TA.diversity(imgs), 2)}
            print("[eval]", json.dumps(row, ensure_ascii=False), flush=True)
            rows.append(row)
    return rows


# ---------------------------------------------------------------- main

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["prep", "build", "train", "eval", "all"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args(argv)
    os.makedirs(DATA, exist_ok=True)

    if args.stage in ("prep", "all"):
        manifest = stage_prep(args.size)
    else:
        manifest = json.load(open(os.path.join(DATA, "manifest.json"), encoding="utf-8"))

    if args.stage == "prep":
        return 0

    pipe = TA.load_pipe()
    clip = TA.load_clip()

    if args.stage in ("build", "all"):
        data = stage_build(pipe, manifest, args.size)
    else:
        data = torch.load(os.path.join(DATA, "data_real.pt"), weights_only=False)

    data["train_keys"] = manifest["train_ids"]
    data["heldout_keys"] = manifest["heldout_ids"]

    head = None
    if args.stage in ("train", "all"):
        head = stage_train(pipe, data, args.steps)
    if args.stage == "eval":
        head = Head2().cuda()
        head.load_state_dict(torch.load(HEAD_PATH, map_location="cuda"))
        head.eval()

    if args.stage in ("eval", "all"):
        rows = stage_eval(pipe, clip, manifest, head, args.size)
        summary = {"size": args.size, "steps": args.steps, "n_ctx": N_CTX,
                   "train_ids": manifest["train_ids"], "heldout_ids": manifest["heldout_ids"],
                   "note": "CLIP-I 相对该身份另外 3 张真实照片的均值；none = 纯文本无身份信号",
                   "results": rows}
        with open(os.path.join(DATA, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print("[summary_json] " + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

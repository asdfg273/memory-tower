"""数据规模曲线：同一个连接器（小投影头），训练身份数 4 / 12 / 24，看留出身份的身份保持率。

回答的问题：A′/A″ 的"学不动"是数据太少，还是结构/目标本身不可学？
- 身份一次 prep（流式），目标图一次 build；训练只在子集上做，条件可比。
- 对照：none（不放头）、头容量减半（d_hid 384）。

用法：
    venv\\Scripts\\python.exe tools_album_scaling.py --ns 4,12,24 --steps 2000
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

import tools_album_head_real as AH
import tools_album_tower as TA

ROOT = os.path.join("_ab_tmp", "album_scaling")
OUT = os.path.join("reports", "repro", "album_scaling.json")
N_HELDOUT = 2
N_IMGS = 6
EVAL_SCENES = AH.EVAL_SCENES[:6]


def prep(n_ids: int, size: int, n_rows: int = 8000) -> Dict[str, object]:
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
    usable = [(k, v) for k, v in sorted(buckets.items()) if len(v) >= N_IMGS]
    print(f"[prep] rows={seen} usable={len(usable)}", flush=True)
    if len(usable) < n_ids + N_HELDOUT:
        raise SystemExit(f"可用身份只有 {len(usable)} 个，不够 {n_ids + N_HELDOUT}")
    picked = usable[: n_ids + N_HELDOUT]
    manifest = {"train_ids": [], "heldout_ids": [], "files": {}}
    for i, (ident, blobs) in enumerate(picked):
        split = "train" if i < n_ids else "heldout"
        key = f"{split}_{i}"
        files = []
        for k, blob in enumerate(blobs):
            path = os.path.join(ROOT, f"{key}_{k}.png")
            Image.open(io.BytesIO(blob)).convert("RGB").resize((size, size)).save(path)
            files.append(path)
        manifest["files"][key] = files
        (manifest["train_ids"] if split == "train" else manifest["heldout_ids"]).append(key)
    json.dump(manifest, open(os.path.join(ROOT, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[prep] train={len(manifest['train_ids'])} heldout={len(manifest['heldout_ids'])}",
          flush=True)
    return manifest


@torch.no_grad()
def build(pipe, manifest, size: int) -> Dict[str, object]:
    from diffusers import StableDiffusionImg2ImgPipeline

    path = os.path.join(ROOT, "data.pt")
    if os.path.exists(path):
        data = torch.load(path, weights_only=False)
        if len(data.get("ids", {})) >= len(manifest["files"]):
            print("[build] cache hit", flush=True)
            return data
    i2i = StableDiffusionImg2ImgPipeline(**pipe.components)
    emb, _ = pipe.encode_prompt(AH.PROMPT_NEUTRAL, "cuda", 1, False)
    data = {"ids": {}}
    for key, files in manifest["files"].items():
        ref_img = Image.open(files[0]).convert("RGB")
        ref_tokens = TA.ref_tokens(pipe, ref_img, AH.PROMPT_NEUTRAL, size)
        items = []
        for tgt_path in files[1:]:
            tgt = Image.open(tgt_path).convert("RGB")
            g = torch.Generator(device="cuda").manual_seed(abs(hash(tgt_path)) % 10_000)
            out = i2i(AH.PROMPT_NEUTRAL, image=tgt, strength=0.75, num_inference_steps=20,
                      guidance_scale=7.5, generator=g).images[0]
            px = pipe.image_processor.preprocess(out, height=size, width=size)
            lat = pipe.vae.encode(px.to("cuda", torch.float16)).latent_dist.mode() * 0.18215
            items.append({"lat": lat.half().cpu(), "emb": emb.half().cpu()})
        data["ids"][key] = {"ref_tokens": ref_tokens.cpu(), "items": items, "ref_img": files[0],
                            "files": files}
        print(f"[build] {key} ref + {len(items)}", flush=True)
    torch.save(data, path)
    return data


def train(pipe, data, ids: List[str], steps: int, d_hid: int, seed: int = 0,
          lr: float = 1e-3):
    torch.manual_seed(seed)
    unet = pipe.unet
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    head = AH.Head2(d_hid=d_hid).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    pool = []
    for key in ids:
        d = data["ids"][key]
        for it in d["items"]:
            pool.append({"ref": d["ref_tokens"].cuda().float(),
                         "lat": it["lat"].cuda(), "emb": it["emb"].cuda()})
    n_par = sum(p.numel() for p in head.parameters())
    print(f"[train] ids={len(ids)} pairs={len(pool)} steps={steps} params={n_par:,}", flush=True)
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
        if step % 250 == 0 or step == steps - 1:
            print(f"  [step {step}] loss {sum(hist[-250:]) / len(hist[-250:]):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    head.eval()
    return head, n_par, float(sum(hist[-250:]) / len(hist[-250:]))


@torch.no_grad()
def evaluate(pipe, clip, data, manifest, head, size: int, tag: str) -> Dict[str, float]:
    clipm, clipp = clip
    per_id, acc = {}, []
    for key in manifest["heldout_ids"]:
        d = data["ids"][key]
        ref_tokens = d["ref_tokens"].cuda().float()
        idimgs = [Image.open(p).convert("RGB") for p in d["files"][1:4]]
        idemb = TA.clip_image_emb(clipm, clipp, idimgs).mean(0, keepdim=True)
        idemb = idemb / idemb.norm(dim=-1, keepdim=True)
        cos = []
        for si, scene in enumerate(EVAL_SCENES):
            im = AH.generate(pipe, head, ref_tokens, scene, 4000 + si, size)
            if si < 2:
                im.save(os.path.join(ROOT, f"_eval_{tag}_{key}_{si}.png"))
            e = TA.clip_image_emb(clipm, clipp, [im])
            cos.append(round(float((e * idemb).sum()), 4))
        per_id[key] = {"clip_i": round(sum(cos) / len(cos), 4), "per_scene": cos}
        acc.append(sum(cos) / len(cos))
    mean = sum(acc) / max(len(acc), 1)
    print(f"[eval] {tag}: CLIP-I {mean:.4f} {per_id}", flush=True)
    return {"mean_clip_i": round(mean, 4), "per_id": per_id}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="4,12,24")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    size = args.size

    pipe = TA.load_pipe()
    clip = TA.load_clip()
    manifest = prep(max(ns), size)
    data = build(pipe, manifest, size)

    results: Dict[str, object] = {"size": size, "steps": args.steps, "n_imgs": N_IMGS,
                                  "evals": {}, "scales": {}}
    results["evals"]["none"] = evaluate(pipe, clip, data, manifest, None, size, "none")

    for n in ns:
        ids = manifest["train_ids"][:n]
        head, n_par, loss = train(pipe, data, ids, args.steps, 768, seed=n)
        ev = evaluate(pipe, clip, data, manifest, head, size, f"n{n}")
        results["evals"][f"n{n}"] = ev
        results["scales"][str(n)] = {"train_ids": n, "pairs": n * (N_IMGS - 1),
                                     "params": n_par, "final_loss": round(loss, 4),
                                     "mean_clip_i": ev["mean_clip_i"]}
        print(f"[scale n={n}] CLIP-I {ev['mean_clip_i']:.4f} "
              f"(none {results['evals']['none']['mean_clip_i']:.4f})", flush=True)

    n_big = max(ns)
    head_s, n_par_s, loss_s = train(pipe, data, manifest["train_ids"][:n_big], args.steps,
                                    384, seed=99)
    ev_s = evaluate(pipe, clip, data, manifest, head_s, size, "small")
    results["evals"]["small"] = ev_s
    results["small_head"] = {"d_hid": 384, "params": n_par_s, "final_loss": round(loss_s, 4),
                             "mean_clip_i": ev_s["mean_clip_i"]}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[done] -> " + args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

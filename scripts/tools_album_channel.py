"""5.16 channel-capacity experiment: injection width W x frozen/LoRA consumer.

Question: why do 4-8 injected context tokens fail to carry identity --
is the channel too narrow, or is the readout inert (no gradient path)?

Grid: W in {8,32,128} x consumer in {frozen UNet, LoRA on cross-attn only (rank 8)}
Controls: zero (readout zeroed) / rand (same-shape random readout)

Kill criterion H9 (pre-registered): if LoRA + W=128 + readout-in-the-loop cannot beat
the plain-text-prompt baseline on held-out identities, the "memory injection" line is
declared dead at this compute tier and we stop iterating on structure.

Usage:
    venv\\Scripts\\python.exe tools_album_channel.py --arms grid --ws 8,32,128 --steps 800
    venv\\Scripts\\python.exe tools_album_channel.py --arms ctrl --w 128 --steps 800
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

import tools_album_head_real as A
import tools_album_tower as T

TMP = os.path.join("_ab_tmp", "channel")

# 纯文本基线的身份描述：人对参考图一次性写下的"打字版身份"（不看记忆接口能看到的那点信息）
DESCS = {
    "train_0": "a young woman with long blonde hair, fair skin and blue eyes",
    "train_1": "a young woman with long dark brown hair, olive skin and dark eyes",
    "train_2": "a man with short spiky bleached blonde hair and blue eyes",
    "train_3": "an older woman with short curly brown hair and bright red lipstick",
    "heldout_4": "a blonde woman with long wavy hair, heavy eye makeup and dark eyebrows",
    "heldout_5": "a woman with shoulder-length dark auburn hair and a wide smile",
}


class LoRALinear(torch.nn.Module):
    """y = base(x) + scale * B(A(x)); base frozen, A/B kept in fp32."""

    def __init__(self, base: torch.nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / rank
        # LoRA params stay fp32: in fp16 with lr 5e-4 they overflow to nan on step 1.
        self.A = torch.nn.Linear(base.in_features, rank, bias=False).float()
        self.B = torch.nn.Linear(rank, base.out_features, bias=False).float()
        torch.nn.init.zeros_(self.B.weight)
        for p in list(self.A.parameters()) + list(self.B.parameters()):
            p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.scale * self.B(self.A(x.float()))
        return out + delta.to(out.dtype)


def strip_lora(unet) -> int:
    """把上一臂包上去的 LoRALinear 摘掉；否则下一个 arm 会继承/嵌套上一层 LoRA。"""
    n = 0
    for mod in unet.modules():
        if not isinstance(mod, torch.nn.Module) or not hasattr(mod, "attn2"):
            continue
        inner = mod.attn2
        for attr in ("to_q", "to_k", "to_v"):
            cur = getattr(inner, attr)
            if isinstance(cur, LoRALinear):
                setattr(inner, attr, cur.base)
                n += 1
        cur = inner.to_out[0]
        if isinstance(cur, LoRALinear):
            inner.to_out[0] = cur.base
            n += 1
    if n:
        print(f"[strip] 摘掉 {n} 个 LoRA 层", flush=True)
    return n


def add_lora(unet, rank: int = 8, alpha: int = 16) -> List[torch.nn.Parameter]:
    """Wrap q/k/v/out of every cross-attention (attn2) module."""
    hits = [(n, m) for n, m in unet.named_modules()
            if n.endswith("attn2") and hasattr(m, "to_q")]
    params: List[torch.nn.Parameter] = []
    for _, mod in hits:
        for attr in ("to_q", "to_k", "to_v"):
            cur = getattr(mod, attr)
            lora = LoRALinear(cur, rank, alpha).to(cur.weight.device)
            setattr(mod, attr, lora)
            params += list(lora.parameters())
        out0 = mod.to_out[0]
        lora = LoRALinear(out0, rank, alpha).to(out0.weight.device)
        mod.to_out[0] = lora
        params += list(lora.parameters())
    print(f"[lora] {len(hits)} cross-attn modules, "
          f"{sum(p.numel() for p in params)} params", flush=True)
    return params


def _text_tokenizer():
    """CLIP text tokenizer: the local CLIP cache lacks tokenizer files, so borrow SD1.5's
    (same 49408 vocabulary)."""
    from transformers import CLIPTokenizer
    try:
        return CLIPTokenizer.from_pretrained(
            os.path.join("_ab_tmp", "sd15", "tokenizer"), local_files_only=True)
    except Exception as exc:                      # noqa: BLE001
        print("[warn] no CLIP tokenizer, CLIP-T will be nan:", exc, flush=True)
        return None


def make_ctx(head, ref: torch.Tensor, w: int, mode: str) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros(1, w, 768, device="cuda", dtype=torch.float16)
    if mode == "rand":
        g = torch.Generator(device="cuda").manual_seed(1234)
        v = torch.randn(w, 768, device="cuda", dtype=torch.float16, generator=g) * 0.5
        return v.unsqueeze(0)
    return head(ref.cuda().float()).to(torch.float16).unsqueeze(0)


def train_arm(pipe, data, w: int, lora: bool, steps: int, mode: str = "real",
              seed: int = 1, lr_head: float = 1e-3, lr_lora: float = 5e-4):
    torch.manual_seed(seed)
    unet = pipe.unet
    strip_lora(unet)                    # 每个 arm 都从干净底座开始
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    head = A.Head2(n_ctx=w).cuda()
    groups = [{"params": list(head.parameters()), "lr": lr_head}]
    if lora:
        groups.append({"params": add_lora(unet, 8, 16), "lr": lr_lora})
    opt = torch.optim.AdamW(groups)
    pool = []
    for k in data["train_keys"]:
        d = data["ids"][k]
        for it in d["items"]:
            pool.append({"ref": d["ref_tokens"].cuda().float(),
                         "lat": it["lat"].cuda(), "emb": it["emb"].cuda()})

    fixed_ctx = make_ctx(head, pool[0]["ref"], w, mode) if mode in ("zero", "rand") else None
    t0, hist = time.time(), []
    tag = f"w{w}{'L' if lora else 'F'}{'' if mode == 'real' else '-' + mode}"
    for step in range(steps):
        it = pool[step % len(pool)]
        lat, emb = it["lat"].to(torch.float16), it["emb"].to(torch.float16)
        c = fixed_ctx if fixed_ctx is not None else make_ctx(head, it["ref"], w, mode)
        cond = torch.cat([emb, c.to(emb.dtype)], dim=1)
        noise = torch.randn_like(lat)
        t = torch.randint(0, 1000, (1,), device="cuda").long()
        lat_t = pipe.scheduler.add_noise(lat, noise, t)
        pred = unet(lat_t, t, encoder_hidden_states=cond).sample
        loss = F.mse_loss(pred.float(), noise.float())
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print(f"[warn] {tag} step {step} non-finite loss, skipped", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        hist.append(float(loss.detach()))
        if step % 100 == 0 or step == steps - 1:
            print(f"[{tag}] step {step} loss {sum(hist[-100:]) / len(hist[-100:]):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return head


@torch.no_grad()
def eval_arm(pipe, clip, manifest, head, w: int, mode: str, size: int,
             tag: str) -> Dict[str, object]:
    clipm, clipp = clip
    tok = _text_tokenizer()
    os.makedirs(TMP, exist_ok=True)
    rows = []
    for key in manifest["heldout_ids"]:
        files = manifest["files"][key]
        ref = Image.open(files[0]).convert("RGB")
        ref_tokens = T.ref_tokens(pipe, ref, A.PROMPT_NEUTRAL, size)
        idimgs = [Image.open(p).convert("RGB") for p in files[1:4]]
        idemb = T.clip_image_emb(clipm, clipp, idimgs)
        idemb = idemb / idemb.norm(dim=-1, keepdim=True)
        imgs = []
        for si, scene in enumerate(A.EVAL_SCENES):
            pos, _ = pipe.encode_prompt(scene, "cuda", 1, False)
            neg, _ = pipe.encode_prompt("", "cuda", 1, False)
            c = make_ctx(head, ref_tokens, w, mode).to(pos.dtype)
            pos, neg = torch.cat([pos, c], 1), torch.cat([neg, c], 1)
            g = torch.Generator(device="cuda").manual_seed(4000 + si)
            im = pipe(prompt_embeds=pos, negative_prompt_embeds=neg, num_inference_steps=20,
                      guidance_scale=7.5, height=size, width=size, generator=g).images[0]
            im.save(os.path.join(TMP, f"{tag}_{key}_{si}.png"))
            imgs.append(im)
        embs = T.clip_image_emb(clipm, clipp, imgs)
        cos = [float((e * idemb[0]).sum()) for e in embs]
        if tok is not None:
            tembs = T.clip_text_emb(clipm, tok, A.EVAL_SCENES)
            tcos = [float((t_ / t_.norm()) @ (e / e.norm())) for t_, e in zip(tembs, embs)]
        else:
            tcos = [float("nan")]
        row = {"id": key, "clip_i": round(sum(cos) / len(cos), 4),
               "clip_t": round(sum(tcos) / len(tcos), 4),
               "diversity": round(T.diversity(imgs), 2),
               "per_image": [round(c, 4) for c in cos]}
        print("[eval]", json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
    mean_i = sum(r["clip_i"] for r in rows) / len(rows)
    mean_t = sum(r["clip_t"] for r in rows) / len(rows)
    return {"per_id": rows, "mean_clip_i": round(mean_i, 4),
            "mean_clip_t": round(mean_t, 4)}


@torch.no_grad()
def eval_text(pipe, clip, manifest, size: int, tag: str) -> Dict[str, object]:
    """纯文本基线：把身份写进 prompt，不经过记忆接口、不训任何东西。"""
    clipm, clipp = clip
    tok = _text_tokenizer()
    os.makedirs(TMP, exist_ok=True)
    rows = []
    for key in manifest["heldout_ids"]:
        files = manifest["files"][key]
        idimgs = [Image.open(p).convert("RGB") for p in files[1:4]]
        idemb = T.clip_image_emb(clipm, clipp, idimgs)
        idemb = idemb / idemb.norm(dim=-1, keepdim=True)
        imgs, prompts = [], []
        for si, scene in enumerate(A.EVAL_SCENES):
            prompt = f"{DESCS.get(key, 'a person')}, {scene}"
            prompts.append(prompt)
            g = torch.Generator(device="cuda").manual_seed(4000 + si)
            im = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=7.5,
                      height=size, width=size, generator=g).images[0]
            im.save(os.path.join(TMP, f"{tag}_{key}_{si}.png"))
            imgs.append(im)
        embs = T.clip_image_emb(clipm, clipp, imgs)
        cos = [float((e * idemb[0]).sum()) for e in embs]
        if tok is not None:
            tembs = T.clip_text_emb(clipm, tok, prompts)
            tcos = [float((t_ / t_.norm()) @ (e / e.norm())) for t_, e in zip(tembs, embs)]
        else:
            tcos = [float("nan")]
        row = {"id": key, "clip_i": round(sum(cos) / len(cos), 4),
               "clip_t": round(sum(tcos) / len(tcos), 4),
               "diversity": round(T.diversity(imgs), 2),
               "per_image": [round(c, 4) for c in cos],
               "prompt0": prompts[0]}
        print("[text]", json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
    mean_i = sum(r["clip_i"] for r in rows) / len(rows)
    mean_t = sum(r["clip_t"] for r in rows) / len(rows)
    return {"per_id": rows, "mean_clip_i": round(mean_i, 4),
            "mean_clip_t": round(mean_t, 4)}


@torch.no_grad()
def eval_sep(pipe, clip, manifest, head, w: int, size: int, n: int = 10) -> Dict[str, object]:
    """固定 LoRA+头，只换读出（身份 A vs B），看 CLIP 嵌入能不能把两组分开。

    分不开 → 通道是死的，路损曲线一概不用看。
    """
    clipm, clipp = clip
    os.makedirs(TMP, exist_ok=True)
    scene = A.EVAL_SCENES[0]
    feats: Dict[str, torch.Tensor] = {}
    for key in manifest["heldout_ids"]:
        ref = Image.open(manifest["files"][key][0]).convert("RGB")
        ref_tokens = T.ref_tokens(pipe, ref, A.PROMPT_NEUTRAL, size)
        c = make_ctx(head, ref_tokens, w, "real")
        pos, _ = pipe.encode_prompt(scene, "cuda", 1, False)
        neg, _ = pipe.encode_prompt("", "cuda", 1, False)
        pos = torch.cat([pos, c.to(pos.dtype)], 1)
        neg = torch.cat([neg, c.to(neg.dtype)], 1)
        imgs = []
        for i in range(n):
            g = torch.Generator(device="cuda").manual_seed(9000 + i)
            im = pipe(prompt_embeds=pos, negative_prompt_embeds=neg,
                      num_inference_steps=20, guidance_scale=7.5,
                      height=size, width=size, generator=g).images[0]
            im.save(os.path.join(TMP, f"sep_{key}_{i}.png"))
            imgs.append(im)
        feats[key] = T.clip_image_emb(clipm, clipp, imgs).float()

    keys = list(feats)
    cent = {k: feats[k].mean(0) for k in keys}
    d_cross = float((cent[keys[0]] - cent[keys[1]]).norm())
    intra = float(torch.stack([(feats[k] - cent[k]).norm(dim=-1).mean() for k in keys]).mean())
    # 最近质心留一分类
    correct = 0
    total = 0
    for k in keys:
        for i in range(feats[k].shape[0]):
            others = [j for j in keys if j != k]
            d_own = float((feats[k][i] - cent[k]).norm())
            d_other = min(float((feats[k][i] - cent[j]).norm()) for j in others)
            correct += int(d_own < d_other)
            total += 1
    out = {"ids": keys, "cross_centroid": round(d_cross, 4),
           "intra_spread": round(intra, 4), "ratio": round(d_cross / max(intra, 1e-6), 3),
           "nearest_centroid_acc": round(correct / total, 3), "n_per_id": n}
    print("[sep]", json.dumps(out, ensure_ascii=False), flush=True)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="grid", choices=["grid", "ctrl", "text", "sep"])
    ap.add_argument("--ws", default="8,32,128")
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(TMP, "arms.json"))
    args = ap.parse_args(argv)
    os.makedirs(TMP, exist_ok=True)

    pipe = T.load_pipe()
    clip = T.load_clip()
    mp = os.path.join(A.DATA, "manifest.json")
    manifest = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else A.stage_prep(args.size)
    data_path = os.path.join(A.DATA, "data_real.pt")
    if os.path.exists(data_path):
        data = torch.load(data_path, weights_only=False)
    else:
        data = A.stage_build(pipe, manifest, args.size)
    data["train_keys"] = manifest["train_ids"]

    if args.arms == "grid":
        jobs = [(w, lora, "real") for w in [int(x) for x in args.ws.split(",")]
                for lora in (False, True)]
    elif args.arms == "ctrl":
        jobs = [(args.w, True, m) for m in ("zero", "rand")]
    elif args.arms == "sep":
        jobs = [(args.w, True, "real")]
    else:
        jobs = []

    res: Dict[str, object] = {"size": args.size, "steps": args.steps, "arms": {}}
    if args.arms == "sep":
        w, lora, mode = jobs[0]
        head = train_arm(pipe, data, w, lora, args.steps, mode=mode)
        res["arms"]["sep"] = eval_sep(pipe, clip, manifest, head, w, args.size)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("[done] ->", args.out, flush=True)
        return 0
    if args.arms == "text":
        res["arms"]["text_prompt"] = eval_text(pipe, clip, manifest, args.size, "text")
        print(f"[arm text_prompt] CLIP-I {res['arms']['text_prompt']['mean_clip_i']:.4f}",
              flush=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("[done] ->", args.out, flush=True)
        return 0
    for w, lora, mode in jobs:
        tag = f"w{w}{'_lora' if lora else '_frozen'}" + ("" if mode == "real" else "_" + mode)
        head = train_arm(pipe, data, w, lora, args.steps, mode=mode)
        ev = eval_arm(pipe, clip, manifest, head, w, mode, args.size, tag)
        res["arms"][tag] = ev
        print(f"[arm {tag}] CLIP-I {ev['mean_clip_i']:.4f} "
              f"CLIP-T {ev['mean_clip_t']:.4f}", flush=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print("[done] ->", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

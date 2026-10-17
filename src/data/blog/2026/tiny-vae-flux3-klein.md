---
title: "A Tiny VAE for FLUX 3 Klein: 30× Smaller, 11× Faster Decoding"
author: Tanmay
pubDatetime: 2026-10-17T04:00:00Z
slug: tiny-vae-flux3-klein
featured: true
draft: false
tags:
  - diffusion
  - vae
  - distillation
  - flux
  - image-generation
  - taesd
description: "Distilling the FLUX 3 Image Klein autoencoder into a 2.7M-parameter decoder: what the latent space really is, how the model was trained, and how it compares with the full 84M-parameter VAE on real FLUX 3 generations."
canonicalURL: "https://tanmaypatil123.github.io/posts/2026/tiny-vae-flux3-klein"
---

> **TL;DR.** FLUX 3 Image Klein ships the FLUX.2 autoencoder unchanged, so a tiny decoder trained for that latent space works for FLUX 3. I fine-tuned [TAEF2](https://huggingface.co/madebyollin/taef2) on 25.8k real photos with L1 + LPIPS + a light PatchGAN loss and weight averaging. The result, **Tiny Klein 3 VAE**, has **2.7M parameters instead of 84M**, decodes **11× faster at 1024²** with about **4× less memory**, and its perceptual gap to the full VAE (LPIPS) is **26–32% smaller than TAEF2's** on FLUX 3 generations. It is still visibly softer than the full VAE in the finest textures. It also works for image editing, including bounding-box edits, as long as the full VAE encodes the reference images. Drag the sliders below and judge for yourself.

<style>
.tv{--tv-line:color-mix(in srgb,var(--foreground) 14%,transparent)}
.tv-fig{margin:1.6rem 0;border:1px solid var(--border);border-radius:14px;padding:1rem;background:color-mix(in srgb,var(--foreground) 3%,transparent)}
.tv-fig figcaption,.tv-cap{font-size:.82rem;opacity:.75;margin-top:.6rem;line-height:1.45}
.tv-cmp{position:relative;width:100%;overflow:hidden;border-radius:10px;user-select:none;touch-action:pan-y;background:#111;cursor:ew-resize}
.tv-cmp:focus-within .tv-knob{outline:2px solid var(--accent);outline-offset:2px}
.tv-cmp img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;margin:0!important;border-radius:0!important;pointer-events:none}
.tv-cmp .tv-top{clip-path:inset(0 0 0 50%)}
.tv-cmp .tv-bar{position:absolute;top:0;bottom:0;left:50%;width:2px;margin-left:-1px;background:#fff;box-shadow:0 0 6px rgba(0,0,0,.6);pointer-events:none}
.tv-cmp .tv-knob{position:absolute;top:50%;left:50%;width:34px;height:34px;margin:-17px 0 0 -17px;border-radius:50%;background:#fff;color:#111;display:flex;align-items:center;justify-content:center;font-size:.8rem;font-weight:700;box-shadow:0 2px 8px rgba(0,0,0,.5);pointer-events:none}
.tv-cmp .tv-lab{position:absolute;top:.5rem;padding:.15rem .5rem;border-radius:6px;background:rgba(0,0,0,.65);color:#fff;font-size:.72rem;font-weight:600;pointer-events:none}
.tv-cmp input{position:absolute;left:0;bottom:0;width:100%;height:1px;opacity:0;margin:0;pointer-events:none}
.tv-crops{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin-top:.7rem}
.tv-crops figure{margin:0}
.tv-crops img{width:100%;margin:0!important;border-radius:8px;image-rendering:auto}
.tv-crops span{display:block;font-size:.74rem;color:var(--foreground)!important;opacity:.85;margin-top:.3rem;text-align:center}
.tv-head{display:flex;flex-wrap:wrap;gap:.4rem .9rem;align-items:baseline;justify-content:space-between;margin-bottom:.6rem;color:var(--foreground)}
.tv-head b{font-size:.95rem;color:var(--foreground)!important}
.tv-head span{font-size:.78rem;color:var(--foreground)!important;opacity:.85;font-variant-numeric:tabular-nums}
.tv-prompt,.tv-fig figcaption{color:var(--foreground)!important}
.tv-tabs{display:flex;flex-wrap:wrap;gap:.35rem;margin-bottom:.8rem}
.tv-tabs button{font:inherit;font-size:.78rem;border:1px solid var(--border);background:var(--background);color:var(--foreground);border-radius:999px;padding:.25rem .7rem}
.tv-tabs button[aria-pressed="true"]{background:var(--accent);color:var(--background);border-color:var(--accent)}
.tv-prompt{font-size:.76rem;opacity:.7;margin-top:.6rem}
.tv-chart svg{width:100%;height:auto;display:block}
.tv-chart text{fill:var(--foreground);font-size:11px}
.tv-legend{display:flex;flex-wrap:wrap;gap:1rem;font-size:.76rem;color:var(--foreground)!important;margin-top:.4rem}
.tv-legend span{color:var(--foreground)!important}
.tv-legend i{display:inline-block;width:.9rem;height:.2rem;border-radius:2px;margin-right:.35rem;vertical-align:middle}
.tv-kpi{display:grid;grid-template-columns:repeat(2,1fr);gap:.6rem;margin:1.2rem 0}
@media(min-width:640px){.tv-kpi{grid-template-columns:repeat(4,1fr)}}
.tv-kpi div{border:1px solid var(--border);border-radius:10px;padding:.7rem .8rem;margin:0}
.tv-kpi dt{margin:0!important;padding:0!important;font-size:.68rem;text-transform:uppercase;letter-spacing:.04em;color:var(--foreground)!important;opacity:.7;font-weight:500}
.tv-kpi dd{margin:.2rem 0 0!important;padding:0!important;font-size:1.4rem;line-height:1.2;font-weight:700;color:var(--accent)!important;font-variant-numeric:tabular-nums;white-space:nowrap}
</style>

<dl class="tv-kpi">
<div><dt>Parameters (vs 84M)</dt><dd>2.7M</dd></div>
<div><dt>Faster decode, 1024²</dt><dd>11.2×</dd></div>
<div><dt>Less memory, 2048²</dt><dd>3.8×</dd></div>
<div><dt>LPIPS gap vs TAEF2</dt><dd>−26–32%</dd></div>
</dl>

## Table of contents

## 1. Why a tiny VAE?

FLUX 3 Image Klein is a 5.6B-parameter text-to-image and editing model. Like every latent diffusion model it never touches pixels while it denoises: the transformer works on a compressed **latent** grid, and a separate autoencoder (the **VAE**) turns that grid into an image at the end. FLUX 3 Klein's VAE has **84M parameters** (34M encoder, 50M decoder) and a self-attention block in the middle of the decoder.

For a single 1024² image from the 50-step base model the decode is cheap: about 0.16 s out of roughly 40 s of generation on an L40S. A tiny decoder is not about that case. It matters when:

- **You preview every step.** Decoding all 50 intermediate latents with the full VAE adds about 8 s per image; with the tiny decoder it is about 0.7 s.
- **You generate large images.** FLUX 3 Klein generates natively up to 4K. The full decoder's attention grows quadratically with resolution: decoding 4096² without tiling took **78 s and 24 GB** of extra memory in my test, versus **0.31 s** for the tiny decoder.
- **You are short on GPU memory** or run the distilled 16-step Klein, where decode becomes a bigger share of the total time.

The goal: a decoder **small enough to run everywhere**, reading **exactly the same latents**, losing as little quality as possible.

## 2. FLUX 3 Klein reuses the FLUX.2 VAE

A tiny decoder only works if it reads the same latent space the transformer writes. So the first question was whether FLUX 3 changed the autoencoder.

It did not. The Hugging Face LFS metadata gives the SHA-256 of each weight file without downloading it:

| File | SHA-256 |
| --- | --- |
| `flux-3-image-klein-base/ae.safetensors` | `868fe7b3…ce8f3` |
| `FLUX.2-dev/ae.safetensors` | `868fe7b3…ce8f3` |

The diffusers copies (`vae/diffusion_pytorch_model.safetensors`) also match each other, and the two `vae/config.json` files differ only in the diffusers version stamp. Both describe a 32-channel latent at 1/8 resolution, patchified 2×2 into 128 channels for the transformer.

So anything trained against the FLUX.2 VAE (including TAEF2) already speaks FLUX 3 Klein's latent format.

## 3. The latent-space trap

My first test fed the full VAE's encoder output straight into TAEF2's decoder. It scored **17.9 dB** PSNR, which is badly broken. The latent spread gave it away: the encoder's latents had a standard deviation of about 1.7, TAEF2's own about 0.85.

The FLUX pipeline does not hand the raw VAE latent to the transformer. It patchifies it and **normalizes each of the 128 channels with the VAE's stored batch-norm statistics**, and it undoes that just before decoding:

$$
z_{\text{norm}} = \operatorname{unpatchify}\!\left(\frac{\operatorname{patchify}(z) - \mu_{\text{bn}}}{\sqrt{\sigma^2_{\text{bn}} + \epsilon}}\right)
$$

TAEF2 was trained on $z_{\text{norm}}$, the transformer's space, not on the raw $z$. Its diffusers wrapper swaps in an identity batch-norm, so the pipeline's de-normalization becomes a no-op. Applying the normalization fixed it at once:

| Input to TAEF2's decoder | PSNR (Kodak) | LPIPS |
| --- | ---: | ---: |
| raw VAE latent $z$ | 17.89 dB | 0.224 |
| normalized latent $z_{\text{norm}}$ | 30.69 dB | 0.038 |

Everything below uses $z_{\text{norm}}$.

## 4. Method

### Architecture

I kept TAEF2's architecture (the `flux_2` variant of [TAESD](https://github.com/madebyollin/taesd)) and changed only its weights. The decoder is a plain convolutional network:

- A `tanh` soft-clamp on the latent, then a 3×3 conv to 64 channels.
- Four stages of three residual blocks (3×3 conv → ReLU → 3×3 conv → ReLU → 3×3 conv), with nearest-neighbour 2× upsampling + conv between stages. That gives 8× upsampling.
- In the three blocks at latent resolution, a small 1×1-conv **GroupNorm branch** that injects image-wide statistics. TAEF2 added this to match the FLUX.2 latent accurately.
- No attention and no normalization at pixel resolution. Every layer is 64 channels wide.

That is **1.34M parameters** in the decoder, the same again in the encoder, against 49.6M in the full decoder.

### Distillation setup

The full VAE encoder is the **teacher**: it turns each training photo $x$ into $z_{\text{norm}}$. The tiny decoder $D_\theta$ is the **student** and learns to rebuild the original photo from those latents. Only the decoder is trained. The encoder stays as TAEF2 shipped it; it is only needed for image-to-image and editing.

$$
\mathcal{L}(\theta) = \lVert D_\theta(\tilde z) - x \rVert_1 \;+\; 0.5\,\mathrm{LPIPS}_{\text{VGG}}\big(D_\theta(\tilde z), x\big) \;+\; 0.01\,\mathcal{L}_{\text{adv}}
$$

- **L1** keeps colours and structure right. **LPIPS** (perceptual distance in VGG feature space) punishes the blur that L1 alone produces.
- **$\mathcal{L}_{\text{adv}}$** comes from a small spectral-norm PatchGAN discriminator trained with the hinge loss. It pushes towards real texture rather than an average.
- **Latent jitter:** in 30% of samples, $\tilde z = z_{\text{norm}} + u\cdot 0.1\cdot\varepsilon$ with $u\sim\mathcal U(0,1)$, $\varepsilon\sim\mathcal N(0,I)$. Real FLUX outputs are never perfectly clean encodings.
- **EMA weights:** I keep $\bar\theta \leftarrow d\,\bar\theta + (1-d)\,\theta$ with $d=\min(0.999, \tfrac{1+t}{10+t})$, and evaluate and save $\bar\theta$.

**Data.** 800 DIV2K photos plus 24,997 Unsplash Lite photos at 2048 px. Real photos only, no FLUX outputs. Each sample is resized so its short side is random between the crop size and 3× the crop size, then randomly cropped and flipped. The model sees both fine native-resolution detail and whole-object structure.

**Schedule** (one NVIDIA L40S, bf16 autocast, AdamW, cosine decay):

| Stage | Steps | Crop | Batch | Peak LR |
| --- | ---: | ---: | ---: | ---: |
| 1. warm-up fine-tune (L1 + LPIPS, DIV2K only) | 3,000 | 512 | 16 | 1e-4 |
| 2a. full data, GAN weight 0.05 | 2,000* | 512 | 16 | 1e-4 |
| 2b. full data, GAN weight 0.01 | 1,000* | 512 | 16 | 8e-5 |
| 3. + EMA, main run | 16,000 | 512 | 16 | 6e-5 |
| 4. high-resolution finish | 3,000 | 1024 | 4 | 2e-5 |

\*Stages 2a and 2b are the kept parts of two runs I stopped early; each later stage starts from the best checkpoint of the one before. Section 7 has the details. All in, about 6.5 GPU-hours including the abandoned steps, or roughly $6 of rented compute.

The checkpoint kept at each stage is the one with the lowest LPIPS on the 24 Kodak photos.

## 5. Results

### Quality on real FLUX 3 generations

These are the numbers that matter. I generated images with FLUX 3 Klein base (50 steps), captured the **exact latent** the pipeline hands to its VAE, and decoded it three ways. The full-VAE decode is the reference. None of these images was used for training or for picking checkpoints.

| Decoder | Set A: 6 images | | Set B: 8 images (gallery) | |
| --- | ---: | ---: | ---: | ---: |
| | PSNR ↑ | LPIPS ↓ | PSNR ↑ | LPIPS ↓ |
| TAEF2 | 30.00 | 0.0342 | 28.87 | 0.0363 |
| **Tiny Klein 3 VAE (ours)** | **31.45** | **0.0234** | **30.33** | **0.0267** |
| *gap closed vs TAEF2 (LPIPS)* | | *−32%* | | *−26%* |

Set A is mixed sizes from 1024² to 1088×1920; set B is the 8 gallery images below. Ours beats TAEF2 on all 14 images, by both metrics.

### Quality on photographs (Kodak)

Here the reference is the original photo, so the full VAE's own error is visible too. These numbers are a little optimistic for the tiny models, because Kodak was also used to pick checkpoints.

| Decoder | Params | PSNR ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: |
| Full FLUX 3 VAE | 49.6M (dec) | 34.38 | 0.0137 |
| TAEF2 | 1.34M (dec) | 30.69 | 0.0380 |
| **Tiny Klein 3 VAE (ours)** | 1.34M (dec) | **31.98** | **0.0306** |

### Speed and memory

Decoder only, batch 1, bf16, NVIDIA L40S, no tiling. "Extra memory" is the peak above the model weights.

| Resolution | Full VAE | Tiny Klein 3 VAE | Speed-up |
| --- | --- | --- | ---: |
| 1024² | 161 ms · 2.4 GB | 14.4 ms · 0.6 GB | **11.2×** |
| 2048² | 760 ms · 9.5 GB | 73 ms · 2.5 GB | **10.4×** |
| 4096² | 78,249 ms · 24.0 GB | 311 ms · 10.0 GB | **251×** |

The 4096² row is the full decoder's attention block running out of headroom. A tiled decode would avoid most of it, but tiling the full VAE trades time for seams and complexity.

### Gallery

Each image is a FLUX 3 Klein base generation (seed 7, 50 steps). **Left of the line: full VAE. Right: our tiny decoder.** Drag to compare. Under each is the 384×384 region with the most fine detail, picked automatically, decoded both ways. Numbers are against the full-VAE decode.

<figure class="tv-fig">
<div class="tv-tabs" id="tv-tabs" role="toolbar" aria-label="Choose a sample"></div>
<div id="tv-gallery"></div>
</figure>

## 6. Does it work for image editing?

FLUX 3 Klein also edits images: it takes up to 10 reference images and a caption, which can include a table of bounding boxes. Editing uses the VAE twice. The **encoder** turns each reference image into latents the transformer conditions on, and the **decoder** turns the result back into pixels. My model only replaces the decoder, so I tested three setups:

- **A:** full VAE encoder and decoder (the baseline).
- **B:** full VAE encoder, tiny decoder. It decodes **the same latent as A**, so any difference is decoder error only.
- **C:** tiny encoder and tiny decoder, an all-tiny pipeline with its own generation.

The tests: a plain edit (recolor a sweater), a text edit, a bounding-box **add** (put a cat inside a given box) and **remove** (take out a lamp), and a two-reference composition. "Preserved region" is a part of the image the edit should leave alone, compared with the reference. Same seed everywhere, 50 steps, FLUX 3 Klein base.

| Edit | B vs A | C vs A | Preserved region: A / B / C |
| --- | ---: | ---: | ---: |
| Recolor sweater | 0.021 | 0.064 | 0.015 / 0.027 / 0.068 |
| Change chalk text | 0.025 | 0.047 | 0.033 / 0.043 / 0.059 |
| Bbox: add a cat | 0.026 | 0.040 | 0.012 / 0.028 / 0.031 |
| Bbox: remove a lamp | 0.025 | 0.038 | 0.012 / 0.027 / 0.031 |
| Two references | 0.017 | **0.305** | — |

All numbers are LPIPS, lower is better.

<figure class="tv-fig">
<div class="tv-tabs" id="tv-edit-tabs" role="toolbar" aria-label="Choose an edit"></div>
<div id="tv-edits"></div>
<figcaption>Reference, then the three setups. A and B share one generated latent; C is its own generation from tiny-encoded references.</figcaption>
</figure>

- **With the full encoder (B), editing works exactly like text-to-image.** Every edit lands the same way: the cat is in the same box, the new text is spelled right, the same lamp is gone. The only difference is the decoder's usual softness, LPIPS 0.017–0.026 against A, the same range as for text-to-image. Bounding-box edits behave no differently, because the boxes live in the caption, which the VAE never sees.
- **The preserved regions get slightly softer, not changed.** With B, the untouched parts sit about 0.01–0.015 LPIPS further from the reference than with A. That is the tiny decoder's own error, not the edit leaking.
- **Don't use the tiny encoder for editing.** Its latents only roughly match the full encoder's (cosine 0.90–0.94), and the transformer conditions on them. On single edits that adds drift. With two references the model composed **a different picture** (different hand pose and framing, LPIPS 0.31), with latents at cosine 0.76 to A's. I never fine-tuned the encoder, so this is expected. A distilled encoder would be the next thing to train.

### Editing the same image again and again

Editing an edit compounds errors. I ran three edits in a row on the portrait (recolor the sweater → add glasses → swap the background), each round feeding its own output back as the next reference:

| Round | B vs A | C vs A |
| --- | ---: | ---: |
| 1. recolor | 0.021 | 0.064 |
| 2. + glasses | 0.055 | 0.098 |
| 3. + new background | 0.105 | 0.193 |

<figure class="tv-fig">
<div class="tv-crops" style="grid-template-columns:repeat(3,1fr)">
<figure><img src="/assets/tiny-vae-flux3/edits/chain3_A.webp" alt="Round 3 of chained edits, full VAE" loading="lazy"><span>A · full VAE</span></figure>
<figure><img src="/assets/tiny-vae-flux3/edits/chain3_B.webp" alt="Round 3 of chained edits, full encoder plus tiny decoder" loading="lazy"><span>B · tiny decoder</span></figure>
<figure><img src="/assets/tiny-vae-flux3/edits/chain3_C.webp" alt="Round 3 of chained edits, tiny encoder and decoder" loading="lazy"><span>C · all tiny</span></figure>
</div>
<figcaption>After three rounds of editing. B and C have drifted in color and contrast: warmer skin, more orange hair, deeper shadows.</figcaption>
</figure>

The drift is not just softness. By round 3, B and C show a visible **color and contrast shift**. Each round's small decoder error becomes part of the next reference, and the model faithfully keeps it ("everything else stays unchanged"). **For multi-round editing, use the tiny decoder for previews but feed the full-VAE decode back as the next reference.** That breaks the loop for the cost of one full decode per round.

## 7. What worked and what didn't

Kodak LPIPS of the best checkpoint after each change, lower is better:

| Step | Kodak LPIPS | PSNR |
| --- | ---: | ---: |
| TAEF2, as released | 0.0380 | 30.69 |
| + fine-tune with L1 + LPIPS on DIV2K (3k steps) | 0.0358 | 32.05 |
| + full data and GAN weight **0.05** | 0.0356 → 0.0377 | 31.71 → 31.24 |
| restart with GAN weight **0.01** | 0.0339 | 31.70 |
| + EMA, 16k steps | 0.0310 | 31.94 |
| + 1024px finish, 3k steps | **0.0306** | **31.98** |

- **A strong GAN loss hurt.** At weight 0.05 the adversarial term grew as large as L1 and LPIPS combined, and both metrics got worse after 3k steps. At 0.01 it sharpens textures without fighting the reconstruction losses.
- **EMA gave the biggest gain.** Without it, the Kodak score jumped around by ±0.002 between checkpoints (0.0339 → 0.0356 → 0.0333 → 0.0343). With it, the averaged weights improved almost every evaluation, and beat the raw weights at 15 of the 16 stage-3 checkpoints.
- **The PSNR–LPIPS trade-off is real.** The first L1-heavy fine-tune had the best PSNR (32.05) but worse LPIPS. The final model gives up 0.07 dB of that PSNR for 15% better LPIPS. I chose by LPIPS because it tracks what people see.
- **The 1024px finish helped a little** (0.0310 → 0.0306). The model is fully convolutional, so most of what it learns transfers from 512 crops.

<figure class="tv-fig tv-chart">
<div id="tv-chart"></div>
<div class="tv-legend"><span><i style="background:var(--accent)"></i>EMA weights</span><span><i style="background:color-mix(in srgb,var(--foreground) 45%,transparent)"></i>raw weights</span><span><i style="background:#2f9e6e"></i>TAEF2 = 0.0380</span></div>
<figcaption>Kodak LPIPS during stage 3 (steps 0–16k, 512px) and stage 4 (16k–19k, 1024px). Lower is better. The raw weights jump around while the EMA copy improves steadily.</figcaption>
</figure>

## 8. Limitations

- **It is still a 1.34M-parameter decoder.** On Kodak it reaches 0.031 LPIPS against the full VAE's 0.014. Look at the gallery crops: hair, skin pores, neon text and small high-contrast detail come out slightly smoother. For final renders the full VAE is still better. For previews, drafts and memory-limited setups the tiny one is the better trade.
- **Tiling is not seamless out of the box.** The GroupNorm branch computes image-wide statistics, so decoding tiles separately gives each tile slightly different statistics. Compute those once over the whole latent, or blend overlapping tiles.
- **Only the decoder was trained.** The encoder is TAEF2's original and is not good enough to feed editing references (section 6). Keep the full VAE encoder for editing.
- **Chained edits drift.** Re-editing the tiny decoder's output compounds into a visible color and contrast shift after a few rounds. Feed full-VAE decodes back instead.
- **Evaluation is small.** 24 Kodak photos plus 14 generated images, one seed each. The differences are consistent across images, but the exact percentages would move with a bigger set.
- **The latent format is inferred, not documented.** The VAE file is byte-identical to FLUX.2's today. A future FLUX 3 VAE update would need a re-check.

## 9. Reproduce it

- **Weights and code:** [Patil/tiny-flux-klien-3-vae](https://huggingface.co/Patil/tiny-flux-klien-3-vae). It contains `tiny_klein3_v2.safetensors`, every intermediate checkpoint with its discriminator, the training, evaluation and generation scripts, and the logs.
- **Use it** like TAEF2: build `TAESD(latent_channels=32, arch_variant="flux_2")` from [`taesd.py`](https://github.com/madebyollin/taesd), load the weights, and call `decoder(z_norm)`. In diffusers, use the wrapper from the [TAEF2 model card](https://huggingface.co/madebyollin/taef2) with this file in place of `taef2.safetensors`.
- **Credits:** TAESD and TAEF2 by Ollin Boer Bohan (MIT). FLUX 3 Image Klein by Black Forest Labs (FLUX Kommunity License). Training photos from DIV2K and the Unsplash Lite dataset. This project is not affiliated with or endorsed by Black Forest Labs.

<script>
(function () {
  var BASE = "/assets/tiny-vae-flux3/";
  var S = [
    { id: "portrait", t: "Portrait", w: 1024, h: 1024, ps: 36.33, lp: 0.0221, tp: 34.14, tl: 0.0348, p: "Editorial close-up portrait, freckles, curly auburn hair, cream knit sweater, soft window light." },
    { id: "street_night", t: "Neon street", w: 768, h: 1344, ps: 25.43, lp: 0.0450, tp: 23.79, tl: 0.0590, p: "Rain-soaked Tokyo street at night, dense neon signs incl. “RAMEN 24H”, red lanterns, clear umbrella." },
    { id: "menu_board", t: "Chalk text", w: 1024, h: 1024, ps: 30.55, lp: 0.0259, tp: 28.61, tl: 0.0410, p: "Hand-lettered chalkboard cafe menu on a brick wall, “TODAY'S SPECIALS”." },
    { id: "macro_spider", t: "Macro", w: 1344, h: 768, ps: 27.17, lp: 0.0222, tp: 25.95, tl: 0.0306, p: "Extreme macro of a jumping spider's eyes, iridescent hairs, water droplets, lichen twig." },
    { id: "anime", t: "Anime", w: 1024, h: 1024, ps: 30.54, lp: 0.0222, tp: 29.97, tl: 0.0276, p: "Anime key visual: silver-haired girl in a yellow raincoat on a rooftop at sunset." },
    { id: "food", t: "Food", w: 832, h: 1216, ps: 30.74, lp: 0.0289, tp: 28.90, tl: 0.0410, p: "Overhead photo of a ramen bowl with jammy eggs, chashu, nori, sesame, chili oil." },
    { id: "library", t: "Interior", w: 1344, h: 768, ps: 28.43, lp: 0.0273, tp: 27.43, tl: 0.0293, p: "Sunlit library reading room, arched windows, green banker's lamps, spiral staircase." },
    { id: "camera", t: "Product", w: 1024, h: 1024, ps: 33.48, lp: 0.0197, tp: 32.20, tl: 0.0273, p: "Studio product shot of a vintage rangefinder camera, engraved dials, glass lens." }
  ];

  var tabs = document.getElementById("tv-tabs");
  var gal = document.getElementById("tv-gallery");
  if (tabs && gal) {
    function show(i) {
      var s = S[i];
      tabs.querySelectorAll("button").forEach(function (b, j) { b.setAttribute("aria-pressed", j === i ? "true" : "false"); });
      gal.innerHTML =
        '<div class="tv-head"><b>' + s.t + ' · ' + s.w + '×' + s.h + '</b>' +
        '<span>ours: ' + s.ps.toFixed(2) + ' dB · LPIPS ' + s.lp.toFixed(4) +
        ' &nbsp;|&nbsp; TAEF2: ' + s.tp.toFixed(2) + ' dB · LPIPS ' + s.tl.toFixed(4) + '</span></div>' +
        '<div class="tv-cmp" style="aspect-ratio:' + s.w + '/' + s.h + ';max-height:80vh;margin:0 auto;max-width:calc(80vh*' + (s.w / s.h).toFixed(4) + ')">' +
        '<img src="' + BASE + s.id + '_full.webp" alt="' + s.t + ', full VAE decode" loading="lazy">' +
        '<img class="tv-top" src="' + BASE + s.id + '_tiny.webp" alt="' + s.t + ', tiny VAE decode" loading="lazy">' +
        '<div class="tv-bar"></div><div class="tv-knob">↔</div>' +
        '<span class="tv-lab" style="left:.5rem">Full VAE</span><span class="tv-lab" style="right:.5rem">Tiny (ours)</span>' +
        '<input type="range" min="0" max="100" value="50" aria-label="Comparison divider position">' +
        '</div>' +
        '<div class="tv-crops">' +
        '<figure><img src="' + BASE + s.id + '_full_crop.webp" alt="detail crop, full VAE" loading="lazy"><span>Full VAE (84M) · detail crop</span></figure>' +
        '<figure><img src="' + BASE + s.id + '_tiny_crop.webp" alt="detail crop, tiny VAE" loading="lazy"><span>Tiny Klein 3 VAE (2.7M) · same crop</span></figure>' +
        '</div>' +
        '<p class="tv-prompt"><b>Prompt (abridged):</b> ' + s.p + '</p>';
      var box = gal.querySelector(".tv-cmp");
      var top = box.querySelector(".tv-top"), bar = box.querySelector(".tv-bar"), knob = box.querySelector(".tv-knob");
      var range = box.querySelector("input");
      function set(v) {
        v = Math.max(0, Math.min(100, v));
        range.value = v;
        top.style.clipPath = "inset(0 0 0 " + v + "%)";
        bar.style.left = v + "%";
        knob.style.left = v + "%";
      }
      function fromEvent(e) {
        var r = box.getBoundingClientRect();
        set((e.clientX - r.left) / r.width * 100);
      }
      var dragging = false;
      box.addEventListener("pointerdown", function (e) { dragging = true; box.setPointerCapture(e.pointerId); fromEvent(e); range.focus({ preventScroll: true }); });
      box.addEventListener("pointermove", function (e) { if (dragging) fromEvent(e); });
      box.addEventListener("pointerup", function () { dragging = false; });
      box.addEventListener("pointercancel", function () { dragging = false; });
      range.addEventListener("input", function () { set(+range.value); });
    }
    S.forEach(function (s, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = s.t;
      b.addEventListener("click", function () { show(i); });
      tabs.appendChild(b);
    });
    show(0);
  }

  // editing comparison: reference + three VAE setups
  var E = [
    { id: "recolor", t: "Recolor", d: "“Change the cream sweater to deep emerald green”" },
    { id: "text", t: "Text", d: "“Change the heading to ‘WEEKEND BRUNCH’”" },
    { id: "bbox_add", t: "Bbox add", d: "Add a sleeping orange cat inside a given box on the table" },
    { id: "bbox_remove", t: "Bbox remove", d: "Remove the green lamp in a given box" },
    { id: "multi_ref", t: "Two references", d: "Woman from image 1 holding the camera from image 2 (only image 1 shown)" }
  ];
  var et = document.getElementById("tv-edit-tabs"), eg = document.getElementById("tv-edits");
  if (et && eg) {
    var EB = "/assets/tiny-vae-flux3/edits/";
    var cell = function (id, p, l) {
      return '<figure><img src="' + EB + id + "_" + p + '.webp" alt="' + l + '" loading="lazy"><span>' + l + "</span></figure>";
    };
    var showE = function (i) {
      var e = E[i];
      et.querySelectorAll("button").forEach(function (b, j) { b.setAttribute("aria-pressed", j === i ? "true" : "false"); });
      eg.innerHTML =
        '<p class="tv-prompt" style="margin:0 0 .6rem"><b>Edit:</b> ' + e.d + "</p>" +
        '<div class="tv-crops">' +
        cell(e.id, "ref", "Reference") + cell(e.id, "A", "A · full VAE") +
        cell(e.id, "B", "B · full encoder + tiny decoder") + cell(e.id, "C", "C · tiny encoder + tiny decoder") +
        "</div>";
    };
    E.forEach(function (e, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = e.t;
      b.addEventListener("click", function () { showE(i); });
      et.appendChild(b);
    });
    showE(0);
  }

  // Kodak LPIPS during the EMA run (stage 3: 0-16k @512, stage 4: 16k-19k @1024)
  var chart = document.getElementById("tv-chart");
  if (chart) {
    var ema = [[0,.0339],[1,.0322],[2,.0323],[3,.0323],[4,.0320],[5,.0320],[6,.0317],[7,.0316],[8,.0314],[9,.0314],[10,.0313],[11,.0312],[12,.0312],[13,.0311],[14,.0311],[15,.0310],[16,.0310],[16.5,.0306],[17,.0308],[17.5,.0307],[18,.0306],[18.5,.0306],[19,.0306]];
    var raw = [[0,.0339],[1,.0358],[2,.0333],[3,.0343],[4,.0334],[5,.0328],[6,.0324],[7,.0318],[8,.0345],[9,.0325],[10,.0314],[11,.0314],[12,.0310],[13,.0312],[14,.0314],[15,.0313],[16,.0312],[16.5,.0320],[17,.0317],[17.5,.0321],[18,.0308],[18.5,.0311],[19,.0306]];
    var W = 640, H = 260, L = 52, R = 12, T = 14, B = 34, y0 = .0300, y1 = .0385, x1 = 19;
    var X = function (v) { return L + v / x1 * (W - L - R); };
    var Y = function (v) { return T + (y1 - v) / (y1 - y0) * (H - T - B); };
    var path = function (d) { return d.map(function (p, i) { return (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1); }).join(" "); };
    var grid = "", lbl = "";
    [.030, .032, .034, .036, .038].forEach(function (v) {
      grid += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(v) + '" y2="' + Y(v) + '" stroke="var(--tv-line)"/>';
      lbl += '<text x="' + (L - 6) + '" y="' + (Y(v) + 4) + '" text-anchor="end">' + v.toFixed(3) + '</text>';
    });
    [0, 4, 8, 12, 16, 19].forEach(function (v) {
      lbl += '<text x="' + X(v) + '" y="' + (H - 14) + '" text-anchor="middle">' + v + 'k</text>';
    });
    chart.innerHTML =
      '<svg class="tv" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Kodak LPIPS by training step: EMA weights fall steadily from 0.0339 to 0.0306; raw weights fluctuate.">' +
      grid +
      '<rect x="' + X(16) + '" y="' + T + '" width="' + (X(19) - X(16)) + '" height="' + (H - T - B) + '" fill="var(--accent)" opacity=".07"/>' +
      '<text x="' + ((X(16) + X(19)) / 2) + '" y="' + (T + 12) + '" text-anchor="middle" opacity=".7">1024px</text>' +
      '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(.0380) + '" y2="' + Y(.0380) + '" stroke="#2f9e6e" stroke-dasharray="5 4" stroke-width="1.5"/>' +
      '<text x="' + (L + 6) + '" y="' + (Y(.0380) - 5) + '" fill="#2f9e6e" style="fill:#2f9e6e">TAEF2 as released</text>' +
      '<path d="' + path(raw) + '" fill="none" stroke="var(--foreground)" stroke-opacity=".4" stroke-width="1.5"/>' +
      '<path d="' + path(ema) + '" fill="none" stroke="var(--accent)" stroke-width="2.5"/>' +
      lbl +
      '<text x="' + ((L + W - R) / 2) + '" y="' + (H - 1) + '" text-anchor="middle" opacity=".7">training step</text>' +
      '</svg>';
  }
})();
</script>

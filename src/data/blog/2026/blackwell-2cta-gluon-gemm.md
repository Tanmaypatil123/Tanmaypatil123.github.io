---
title: "Inside a Blackwell 2-CTA GEMM: A Gluon Kernel Tour"
author: Tanmay
pubDatetime: 2026-06-18T10:00:00Z
slug: blackwell-2cta-gluon-gemm
featured: true
draft: false
tags:
  - gemm
  - matmul
  - gluon
  - triton
  - blackwell
  - tcgen05
  - tma
  - tensor-memory
  - gpu-kernels
  - cuda
description: "A line-by-line, interactive walk through a 2-CTA Gluon GEMM for NVIDIA Blackwell (B200): TMA, Tensor Memory, tcgen05 MMA, mbarriers, and software pipelining."
canonicalURL: "https://tanmaypatil123.github.io/posts/2026/blackwell-2cta-gluon-gemm"
---

> **TL;DR**: We tour a single GEMM kernel (`C = A @ B`) written in **Gluon** for the NVIDIA **Blackwell B200**. It uses two CTAs cooperating on one matrix-multiply (`tcgen05`), pulls tiles from HBM with the **Tensor Memory Accelerator (TMA)**, accumulates in **Tensor Memory (TMEM)**, and overlaps loads with compute using an `mbarrier`-driven **software pipeline**. There are three interactive diagrams below — step through them. By the end you'll be able to read every line of the kernel and know *why* it's there.

The kernel lives here: [`kernels/gluon_gemm.py`](https://github.com/Tanmaypatil123/kronos/blob/main/kernels/gluon_gemm.py). This post explains it as a learning exercise — if you've written a Triton kernel before but never touched Blackwell's new primitives, this is for you.

<style>
.gw{border:1px solid var(--border);border-radius:14px;padding:1rem 1.1rem;margin:1.5rem 0;background:color-mix(in srgb,var(--foreground) 4%,transparent);color:var(--foreground);font-size:.9rem;line-height:1.45}
.gw code{color:var(--accent);background:color-mix(in srgb,var(--foreground) 8%,transparent);padding:.05em .35em;border-radius:4px;font-size:.85em}
.gw b{color:var(--foreground)}
.gw *{box-sizing:border-box}
.gw-blue{background:#2563a8;color:#fff}
.gw-green{background:#1f8a5f;color:#fff}
.gw-amber{background:var(--accent);color:var(--background)}
.gw-gray{background:color-mix(in srgb,var(--foreground) 12%,transparent);color:var(--foreground)}
.gw-head{display:flex;flex-wrap:wrap;align-items:center;gap:.6rem;margin-bottom:.85rem}
.gw-title{font-weight:700;letter-spacing:.01em}
.gw-spacer{flex:1}
.gw-step{font-variant-numeric:tabular-nums;opacity:.75;font-size:.82rem}
.gw-btn{font:inherit;font-size:.82rem;border:1px solid var(--border);background:var(--background);color:var(--foreground);border-radius:8px;padding:.32rem .7rem;transition:background .12s,color .12s}
.gw-btn:hover:not(:disabled){background:var(--accent);color:var(--background)}
.gw-btn:disabled{opacity:.38;cursor:not-allowed}
.gw-desc{margin:.5rem 0 .75rem;min-height:3.2em}
.gw-code{margin:0;border-radius:10px;background:#0d1117;color:#c9d1d9;padding:.7rem .85rem;font-size:.78rem;line-height:1.5;overflow-x:auto;white-space:pre;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.gw-bufs{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;margin-bottom:.85rem}
.gw-buf{border:1px solid var(--border);border-radius:10px;padding:.6rem;text-align:center;transition:transform .15s,box-shadow .15s}
.gw-buf .lab{font-size:.72rem;opacity:.7;margin-bottom:.4rem}
.gw-pill{display:inline-block;border-radius:999px;padding:.22rem .7rem;font-weight:700;font-size:.8rem;min-width:6.2em}
.gw-buf.act{transform:translateY(-3px);box-shadow:0 4px 14px color-mix(in srgb,var(--accent) 35%,transparent)}
.gw-prog{height:8px;border-radius:999px;background:color-mix(in srgb,var(--foreground) 12%,transparent);overflow:hidden;margin:.2rem 0 .9rem}
.gw-prog>span{display:block;height:100%;background:#1f8a5f;width:0;transition:width .25s}
.gw-legend{display:flex;flex-wrap:wrap;gap:.9rem;font-size:.76rem;opacity:.85;margin-top:.4rem}
.gw-legend i{display:inline-block;width:.8rem;height:.8rem;border-radius:3px;margin-right:.35rem;vertical-align:-1px}
.gw-tile{display:grid;gap:.7rem;align-items:start;grid-template-columns:120px 1fr;grid-template-areas:". b" "a c"}
.gw-cap{font-size:.72rem;text-align:center;opacity:.7;margin-top:.3rem}
.gw-cell{border-radius:8px;display:flex;align-items:center;justify-content:center;text-align:center;font-weight:600;font-size:.8rem;padding:.5rem;min-height:48px;transition:opacity .15s,filter .15s}
.gw-dim{opacity:.22;filter:grayscale(.4)}
.gw-flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:.2rem}
.gw-node{flex:1 1 120px;min-width:120px;border:1px solid var(--border);border-radius:10px;padding:.7rem .6rem;text-align:center;cursor:pointer;transition:transform .15s,box-shadow .15s,opacity .15s}
.gw-node .t{font-weight:700;font-size:.86rem}
.gw-node .s{font-size:.72rem;opacity:.75;margin-top:.15rem}
.gw-node.on{transform:translateY(-3px);box-shadow:0 4px 16px color-mix(in srgb,var(--accent) 40%,transparent)}
.gw-node.off{opacity:.4}
.gw-arrow{display:flex;flex-direction:column;align-items:center;justify-content:center;min-width:78px;font-size:.66rem;opacity:.8;text-align:center;padding:0 .1rem}
.gw-arrow b{font-size:1.1rem;line-height:1}
@media(max-width:560px){.gw-arrow{transform:rotate(90deg);min-width:auto;padding:.3rem 0}.gw-flow{flex-direction:column}}
</style>

## Table of contents

## What problem are we solving?

A GEMM — **GE**neral **M**atrix **M**ultiply — computes:

$$
C = A \cdot B, \qquad A \in \mathbb{R}^{M\times K},\; B \in \mathbb{R}^{K\times N},\; C \in \mathbb{R}^{M\times N}
$$

Each output element is a dot product over the shared `K` dimension:

$$
C_{ij} = \sum_{k=1}^{K} A_{ik}\, B_{kj}
$$

This is the single most important kernel in deep learning: every linear layer, attention projection, and MLP is a GEMM. So the entire game is keeping the **tensor cores** fed with data fast enough that they never stall. The kernel we're studying is a hand-pipelined, hardware-accelerated answer to that on Blackwell.

> #### 🧠 GPU concept: why tiling?
> Matrices don't fit in fast memory. The trick of every fast GEMM is **tiling**: chop `A`, `B`, and `C` into small blocks that *do* fit in on-chip [shared memory](https://modal.com/gpu-glossary/device-software/shared-memory), multiply the blocks, and accumulate. We stream `K` in chunks of `BK` so we only ever hold a thin slice of `A` and `B` on-chip at once. The output tile (`BM × BN`) stays resident and gets accumulated into across all `K` steps.

This kernel runs each output tile on a **cluster of two CTAs** that split the tile's rows in half. Use the buttons to see who owns what (defaults `BM=256, BN=256, BK=64`):

<div class="gw" id="gw-tile"><div class="gw-head"><span class="gw-title">🧩 2-CTA tiling — who computes what</span><span class="gw-spacer"></span><button class="gw-btn" data-h="0">Highlight CTA 0</button><button class="gw-btn" data-h="1">Highlight CTA 1</button><button class="gw-btn" data-h="2">Show both</button></div><div class="gw-tile"><div style="grid-area:b"><div class="gw-cap">Matrix B · [BK × BN] — shared by both CTAs</div><div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem"><div class="gw-cell gw-blue" data-cta="0">B (left half)</div><div class="gw-cell gw-green" data-cta="1">B (right half)</div></div></div><div style="grid-area:a"><div class="gw-cap">A · [BM × BK]</div><div class="gw-cell gw-blue" data-cta="0" style="border-radius:8px 8px 0 0">CTA 0 rows</div><div class="gw-cell gw-green" data-cta="1" style="border-radius:0 0 8px 8px">CTA 1 rows</div></div><div style="grid-area:c"><div class="gw-cap">Output C · [BM × BN] = A @ B</div><div class="gw-cell gw-blue" data-cta="0" style="border-radius:8px 8px 0 0;min-height:70px">CTA 0 → top 128 output rows</div><div class="gw-cell gw-green" data-cta="1" style="border-radius:0 0 8px 8px;min-height:70px">CTA 1 → bottom 128 output rows</div></div></div><div class="gw-desc" id="gw-tile-desc" style="min-height:2.4em">Each grid program is a <b>2-CTA cluster</b>. The 256-row output tile is split: CTA&nbsp;0 owns the top 128 rows, CTA&nbsp;1 the bottom 128. They cooperate on one <code>tcgen05</code> MMA, so the B tile is shared between them.</div><div class="gw-legend"><span><i class="gw-blue"></i>CTA 0</span><span><i class="gw-green"></i>CTA 1</span></div></div>

The launch grid has one program (one cluster) per output tile:

```python
grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
```

So `program_id(0)` picks the row-band of `C` and `program_id(1)` picks the column-band.

---

## Why Gluon (and not plain Triton)?

You've probably seen Triton kernels: you write `tl.load`, `tl.dot`, `tl.store`, and the compiler decides how to lay out data in shared memory, how to pipeline, and how to schedule warps. That's wonderful for productivity but it hides the hardware.

**Gluon** is Triton's experimental lower-level dialect. It exposes the Blackwell primitives *directly*: you allocate shared-memory buffers yourself, you place memory barriers yourself, you issue the async copy and the tensor-core MMA as explicit instructions, and you own the pipeline. It's closer to writing CUTLASS than to writing Triton — but in Python.

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, tma, mbarrier,
    tcgen05_mma, tcgen05_mma_barrier_count, fence_async_shared, get_tmem_reg_layout,
)
```

That import block is a great table of contents for what's special here:

| Primitive | What it is |
| --- | --- |
| `TensorDescriptor` | A handle describing a tensor + tile shape for **TMA** copies |
| `tma` | The **Tensor Memory Accelerator** — async bulk DMA between HBM and SMEM |
| `mbarrier` | Hardware async **barriers** to coordinate producers/consumers |
| `allocate_tensor_memory` / `TensorMemoryLayout` | Blackwell's new **Tensor Memory (TMEM)** for accumulators |
| `tcgen05_mma` | The Blackwell 5th-gen **tensor core** matrix-multiply instruction |

Let's introduce each one before we read the kernel.

---

## A 60-second tour of the Blackwell memory path

The whole kernel is a choreography of data moving through a hierarchy, from slow-and-big to fast-and-small: **HBM** (global, tens of GB) → **SMEM** (shared, ~228 KB per SM) → **TMEM** (tensor memory) → back out. Three pieces of that path are brand-new on Blackwell or unfamiliar from textbook CUDA, so let's give each a short box.

> #### 🧠 GPU concept: TMA — the Tensor Memory Accelerator
> Classically, to load a tile you'd have every [thread](https://modal.com/gpu-glossary/device-software/thread) in the block compute an address and issue a `load`, then `__syncthreads()`. That burns registers and instruction slots. **TMA** (introduced on Hopper) is a dedicated copy engine: you hand it a *descriptor* ("this 256×64 tile of A, starting at these coordinates") and it streams the whole tile from [HBM](https://modal.com/gpu-glossary/device-hardware/gpu-ram) into [shared memory](https://modal.com/gpu-glossary/device-software/shared-memory) **asynchronously**, in the background, with a single instruction. It even handles out-of-bounds tiles (ragged shapes) by zero-filling — which is why this kernel passes on sizes like `4000×4096`. When the copy finishes, TMA signals an `mbarrier`.

> #### 🧠 GPU concept: Tensor Memory (TMEM)
> On Hopper and earlier, tensor-core accumulators lived in [registers](https://modal.com/gpu-glossary/device-hardware/registers), spread across a warp. Blackwell adds **Tensor Memory** — a dedicated on-chip memory bank physically next to the tensor cores, addressed in a 2-D `(rows × columns)` layout. The big `BM × BN` fp32 accumulator now sits in TMEM instead of hogging the register file, which frees registers and lets the MMA run wider. You allocate it explicitly (`allocate_tensor_memory`) and copy results out with `acc.load(...)` when you're done.

> #### 🧠 GPU concept: tcgen05 — the 5th-gen tensor core MMA
> [Tensor cores](https://modal.com/gpu-glossary/device-hardware/tensor-core) are the units that do the actual `D = A·B + C` on small matrix tiles. `tcgen05` is Blackwell's instruction family for them. Two things make it special here: it reads its operands **straight from shared memory** (no manual register staging), and it can run in **2-CTA mode**, where two thread blocks cooperate on one larger MMA (the split you toggled in the diagram above).

> #### 🧠 GPU concept: CTAs and clusters
> A **CTA** (Cooperative Thread Array) is what CUDA calls a [thread block](https://modal.com/gpu-glossary/device-software/cooperative-thread-array) — a group of warps that share SMEM and run on one [SM](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor). Hopper introduced the [**thread-block cluster**](https://modal.com/gpu-glossary/device-software/thread-block-cluster): a small group of CTAs (here, 2) that run on neighboring SMs and can read each other's shared memory and share barriers. This kernel launches with `num_ctas=2`, so each grid program is a 2-CTA cluster.

With that vocabulary, the kernel reads like prose. Let's go.

---

## The kernel, section by section

### 1. Deriving the tile shapes

```python
@gluon.jit
def _gemm_2cta_kernel(a_desc, b_desc, c_desc, NB: gl.constexpr, num_warps: gl.constexpr):
    cluster_m: gl.constexpr = a_desc.block_type.shape[0]   # BM = 256 (rows for the whole cluster)
    BK: gl.constexpr = a_desc.block_type.shape[1]          # 64  (K-chunk size)
    tile_n: gl.constexpr = b_desc.block_type.shape[1]      # BN  (output columns)
    cta_m: gl.constexpr = cluster_m // 2                   # 128 (rows each CTA owns)
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]
    num_k = (K + BK - 1) // BK                             # number of K-steps to loop over
```

The kernel reads its tile shapes *out of the descriptors* instead of taking them as arguments — the host already baked `BM/BK/BN` into the TMA descriptors. The key line is `cta_m = cluster_m // 2`: the cluster owns 256 rows, each of its two CTAs owns 128. `num_k` is how many `BK`-wide slices of the `K` dimension we'll stream through.

### 2. Picking this program's output tile

```python
    pid_m = gl.program_id(0); pid_n = gl.program_id(1)
    off_m = pid_m * cluster_m; off_n = pid_n * tile_n
```

`(off_m, off_n)` is the top-left corner of the `BM × BN` block of `C` this cluster is responsible for. Every TMA copy below will be relative to these offsets.

### 3. Allocating the staging buffers and barriers

```python
    a_bufs = gl.allocate_shared_memory(dtype, [NB] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [NB] + b_desc.block_type.shape, b_desc.layout)
    ready = mbarrier.allocate_mbarrier(batch=NB, two_ctas=True)
    empty = mbarrier.allocate_mbarrier(batch=NB)
    cnt: gl.constexpr = tcgen05_mma_barrier_count([a_bufs.index(0), b_bufs.index(0)], False)
    for i in gl.static_range(NB):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(empty.index(i), count=cnt)
```

This is the heart of the pipeline setup. We allocate **`NB` copies** of the A-tile and B-tile buffers in shared memory — a *ring buffer* with `NB` slots (default `NB=3`, i.e. triple-buffering). While the tensor cores chew on slot 0, TMA can be filling slots 1 and 2.

Two sets of `mbarrier`s coordinate the ring:

- **`ready[i]`** — flips when the TMA load into slot `i` has fully arrived. The MMA waits on this before consuming. (`two_ctas=True` because in 2-CTA mode both CTAs must see the data.)
- **`empty[i]`** — flips when the MMA is *done* with slot `i`, so TMA is allowed to overwrite it with the next K-tile.

`cnt` is how many arrivals the `empty` barrier should expect per MMA — the helper `tcgen05_mma_barrier_count` computes it from the operand shapes so we don't hard-code it.

> #### 🧠 GPU concept: mbarriers and the producer/consumer dance
> An `mbarrier` is a hardware barrier sitting in shared memory with a **phase bit** that flips each time the expected number of arrivals lands. Async engines (TMA, tensor cores) "arrive" on it when they finish; threads "wait" on a phase. This is how you build a lock-free producer/consumer pipeline: the producer (TMA) signals `ready`, the consumer (MMA) signals `empty`, and nobody spins on a global lock. It's the GPU-native version of a bounded queue.

### 4. Allocating the accumulator in Tensor Memory

```python
    acc_layout: gl.constexpr = TensorMemoryLayout((cta_m, tile_n), col_stride=1,
                                                  cta_split_num=(2, 1), two_ctas=True)
    acc = allocate_tensor_memory(gl.float32, [cluster_m, tile_n], acc_layout)
    a_pc: gl.constexpr = a_desc.block_type.nbytes // 2     # bytes of A per CTA (half the tile)
    b_pc: gl.constexpr = b_desc.block_type.nbytes // 2
```

The accumulator is a `256 × BN` fp32 tile in **TMEM**. `cta_split_num=(2, 1)` says: split it across **2 CTAs along the M dimension, 1 along N** — so each CTA physically holds its own `128 × BN` half, but the pair addresses it as one `256 × BN` logical tile. Accumulating in fp32 (even though A/B are fp16/bf16) is what keeps the result accurate.

`a_pc` / `b_pc` are "bytes per CTA" — half of a full tile, because in 2-CTA mode each CTA's TMA only pulls *its* half. We'll feed these to `mbarrier.expect` so the barrier knows exactly how many bytes constitute "done."

### 5. Priming the pipeline (prologue)

```python
    for i in gl.static_range(NB):
        mbarrier.expect(ready.index(i), a_pc + b_pc)
        tma.async_copy_global_to_shared(a_desc, [off_m, i * BK], ready.index(i), a_bufs.index(i))
        tma.async_copy_global_to_shared(b_desc, [i * BK, off_n], ready.index(i), b_bufs.index(i))
```

Before the main loop, we **kick off the first `NB` loads** so the pipeline starts full. For each slot `i`: arm the barrier with `expect` ("await this many bytes"), then fire two async TMA copies — A's tile at K-offset `i*BK` and B's tile at the same offset. These calls **return immediately**; the copies run in the background.

### 6. The main loop — load/compute overlap

This is where the magic happens:

```python
    for k in range(num_k):
        buf = k % NB; ph = (k // NB) & 1
        mbarrier.wait(ready.index(buf), ph, deps=[a_bufs.index(buf), b_bufs.index(buf)])
        tcgen05_mma(a_bufs.index(buf), b_bufs.index(buf), acc, use_acc=(k > 0),
                    mbarriers=[empty.index(buf)])
        kk = k + NB
        if kk < num_k:
            mbarrier.wait(empty.index(buf), ph)
            mbarrier.expect(ready.index(buf), a_pc + b_pc)
            tma.async_copy_global_to_shared(a_desc, [off_m, kk * BK], ready.index(buf), a_bufs.index(buf))
            tma.async_copy_global_to_shared(b_desc, [kk * BK, off_n], ready.index(buf), b_bufs.index(buf))
```

Rather than describe it in prose, **step through it**. The widget below runs the kernel for a `K` split into 6 chunks with `NB=3` buffers. Watch how the tensor cores (green) always have a buffer ready, because TMA (amber) filled it three steps earlier:

<div class="gw" id="gw-pipe"><div class="gw-head"><span class="gw-title">⛓️ Software pipeline — step through it</span><span class="gw-spacer"></span><span class="gw-step" id="gw-pipe-step"></span><button class="gw-btn" id="gw-pipe-back">◀ Back</button><button class="gw-btn" id="gw-pipe-next">Step ▶</button><button class="gw-btn" id="gw-pipe-reset">Reset</button></div><div class="gw-bufs" id="gw-pipe-bufs"><div class="gw-buf"><div class="lab">buffer 0</div><span class="gw-pill"></span></div><div class="gw-buf"><div class="lab">buffer 1</div><span class="gw-pill"></span></div><div class="gw-buf"><div class="lab">buffer 2</div><span class="gw-pill"></span></div></div><div class="gw-prog"><span id="gw-pipe-prog"></span></div><div class="gw-desc" id="gw-pipe-desc"></div><pre class="gw-code" id="gw-pipe-code"></pre><div class="gw-legend"><span><i class="gw-amber"></i>loading (TMA)</span><span><i class="gw-blue"></i>ready</span><span><i class="gw-green"></i>MMA (compute)</span><span><i class="gw-gray"></i>free</span></div></div>

The tensor cores never wait for HBM, because by the time the MMA needs slot `k % NB`, the TMA filled it `NB` steps ago. **That overlap is the entire point** — a GEMM that stalls on memory runs at a fraction of peak. A couple of details from the code:

- `use_acc=(k > 0)` is a neat trick — on the **first** K-step we *overwrite* `acc` (no garbage init needed), and on every step after we *accumulate*.
- The `if kk < num_k` block is the **prefetch**: as soon as the MMA is issued, we queue the load `NB` steps ahead into the same slot — but only after `mbarrier.wait(empty[buf], ph)` confirms the previous MMA finished reading it.

> #### 🧠 GPU concept: software pipelining
> Memory is slow (hundreds of cycles); compute is fast. If you load-then-compute serially, the tensor cores idle during every load. **Software pipelining** breaks that dependency by running `NB` iterations "in flight" at once: iteration `k`'s compute overlaps iterations `k+1..k+NB-1`'s loads. More stages (`NB`) hides more latency but costs more shared memory — which is exactly why `NB` is one of the autotuned knobs.

### 7. Draining the pipeline (epilogue)

```python
    last = (num_k - 1) % NB; lph = ((num_k - 1) // NB) & 1
    mbarrier.wait(empty.index(last), lph)
```

After the loop issues the final MMA, we must **wait for it to actually finish** before touching `acc`. We compute which slot and phase the last K-step used and wait on its `empty` barrier. Now the accumulator holds the complete `C` tile.

### 8. Writing C back out

```python
    reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, [cluster_m, tile_n], acc_layout,
                                                   num_warps, cga_layout=[(1, 0)])
    out = acc.load(reg_layout)                  # TMEM → registers
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(out.to(dtype))                 # registers → SMEM, cast fp32 → fp16/bf16
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)   # SMEM → HBM
    tma.store_wait(pendings=0)                  # make sure the store finished
```

The output journey reverses the input one. Press **Play** to watch the tile travel from Tensor Memory back to global memory — and click any stage for detail:

<div class="gw" id="gw-flow"><div class="gw-head"><span class="gw-title">📤 Writing C back: TMEM → HBM</span><span class="gw-spacer"></span><button class="gw-btn" id="gw-flow-play">▶ Play</button><button class="gw-btn" id="gw-flow-reset">Reset</button></div><div class="gw-flow"><div class="gw-node" data-i="0"><div class="t">TMEM</div><div class="s">acc · fp32</div></div><div class="gw-arrow"><b>→</b><span>acc.load()</span></div><div class="gw-node" data-i="1"><div class="t">registers</div><div class="s">out</div></div><div class="gw-arrow"><b>→</b><span>c_smem.store()<br>+ to(dtype)</span></div><div class="gw-node" data-i="2"><div class="t">shared mem</div><div class="s">c_smem · fp16</div></div><div class="gw-arrow"><b>→</b><span>tma.async_copy<br>shared → global</span></div><div class="gw-node" data-i="3"><div class="t">global</div><div class="s">C matrix</div></div></div><div class="gw-desc" id="gw-flow-desc" style="min-height:2.6em">fp32 accumulate the whole way, narrowed to fp16/bf16 only at the shared-memory hop. Press <b>Play</b> or click a stage.</div></div>

`fence_async_shared()` ensures the SMEM writes are visible to the TMA engine before the store launches, and `tma.store_wait(pendings=0)` blocks until the store drains, so the kernel doesn't exit with an in-flight copy.

---

## The host side

### Setting up descriptors and launching

```python
def gemm(A, B, C=None, *, BM=256, BN=256, BK=64, NB=3, num_warps=4):
    """C = A @ B on B200 via 2-CTA tcgen05. A,B 2-D fp16/bf16. Returns C.
    BM must be 2*instrM (256); tcgen05 caps the MMA N (=BN) at 256."""
    assert A.dtype == B.dtype and A.dtype in (torch.float16, torch.bfloat16)
    M, K = A.shape; K2, N = B.shape; assert K == K2
    if C is None:
        C = torch.empty(M, N, device=A.device, dtype=A.dtype)
    gd: gl.constexpr = gl.float16 if A.dtype == torch.float16 else gl.bfloat16
    a_layout = gl.NVMMASharedLayout.get_default_for([BM, BK], gd, cga_layout=[(1, 0)])
    b_layout = gl.NVMMASharedLayout.get_default_for([BK, BN], gd, cga_layout=[(0, 1)])
    c_layout = gl.NVMMASharedLayout.get_default_for([BM, BN], gd, cga_layout=[(1, 0)])
    a_desc = TensorDescriptor.from_tensor(A, [BM, BK], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BK, BN], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BM, BN], c_layout)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _gemm_2cta_kernel[grid](a_desc, b_desc, c_desc, NB=NB, num_warps=num_warps, num_ctas=2)
    return C
```

A few things worth calling out:

- **`NVMMASharedLayout`** picks a swizzled shared-memory layout that the tensor cores can read without bank conflicts. The `cga_layout` argument tells it how the tile is split across the 2-CTA cluster: A is split along M (`[(1,0)]`), B along N (`[(0,1)]`).
- **`TensorDescriptor.from_tensor(A, [BM, BK], ...)`** builds the TMA descriptor: "this tensor, tiled into `BM×BK` blocks." This object is what the kernel's `tma.async_copy_*` calls dereference.
- The launch passes **`num_ctas=2`** — that single argument is what turns each grid point into a 2-CTA cluster. The constraints in the docstring (`BM=256`, `BN≤256`) come straight from the `tcgen05` 2-CTA instruction's fixed shapes.

### Autotuning

The fastest `(BM, BN, BK, NB)` depends on the matrix shape, so there's a tiny autotuner:

```python
_SPACE = [(256, 256, 64, 2), (256, 256, 64, 3), (256, 256, 64, 4),
          (256, 128, 64, 3), (256, 128, 64, 4), (256, 128, 64, 6)]
_BEST = {}

def gemm_auto(A, B, C=None):
    """Autotuned 2-CTA GEMM. First call for a shape sweeps `_SPACE` and caches."""
    M, K = A.shape; N = B.shape[1]; key = (M, N, K, A.dtype)
    cfg = _BEST.get(key)
    if cfg is None:
        flush = torch.empty(128 * 1024 * 1024, device=A.device, dtype=torch.int8)
        best = None
        for (BM, BN, BK, NB) in _SPACE:
            try:
                for _ in range(5): gemm(A, B, C, BM=BM, BN=BN, BK=BK, NB=NB)
                torch.cuda.synchronize()
                t = 1e9
                for _ in range(3):
                    flush.zero_()
                    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
                    for _ in range(20): gemm(A, B, C, BM=BM, BN=BN, BK=BK, NB=NB)
                    e.record(); torch.cuda.synchronize()
                    t = min(t, s.elapsed_time(e))
                if best is None or t < best[0]: best = (t, (BM, BN, BK, NB))
            except Exception:
                pass
        cfg = best[1]; _BEST[key] = cfg
    BM, BN, BK, NB = cfg
    return gemm(A, B, C, BM=BM, BN=BN, BK=BK, NB=NB)
```

Two details that separate a *toy* benchmark from an *honest* one:

- **`flush.zero_()`** on a 128 MB buffer before each timed run wipes the **L2 cache**. Without it, the second iteration would find `A`/`B` already cached and report fantasy throughput.
- It uses **CUDA events** (not Python `time`) and takes the **min** over repeats, which rejects noise from clock-boost warmup and scheduler jitter.

The `try/except` is there because some configs (e.g. a `BN` that doesn't divide cleanly, or one that overflows shared memory) simply fail to compile or launch — we skip them and keep the ones that work.

### Correctness against ragged shapes

```python
    shapes = [(256, 128, 128), (4096, 4096, 4096), (2048, 2048, 2048),
              (4000, 4096, 4096), (4096, 4096, 4000), (1536, 6144, 2048)]
```

Notice `4000×4096×4096` and `4096×4096×4000`: dimensions that **aren't** multiples of the tile size. These exist on purpose — they test that TMA's **boundary handling** (zero-padding the out-of-range part of edge tiles) is correct. The check compares against an fp32 reference and asserts the relative error is below a dtype-appropriate tolerance:

```python
    ref = A.float() @ B.float()
    rel = ((C.float() - ref).abs().max() / ref.abs().max()).item()
    tol = 5e-2 if dt == torch.float16 else 1e-1
```

The tolerances look loose, but remember we're accumulating thousands of fp16/bf16 products — that's expected rounding, not a bug. The fp32 *accumulator* is what keeps it this tight.

---

## Why this design is fast

Putting it together, every performance-critical idea in modern GEMM is present in this one file:

| Technique | Where | Pays off by… |
| --- | --- | --- |
| **Tiling** | `BM×BN×BK` blocks | keeping working set in fast on-chip memory |
| **TMA async copy** | `tma.async_copy_*` | freeing threads from address math; overlapping DMA |
| **Software pipelining** | `NB`-slot ring + `mbarrier`s | hiding HBM latency behind tensor-core compute |
| **2-CTA tcgen05** | `num_ctas=2`, `cta_split_num` | one MMA spanning two SMs → bigger tiles, more reuse |
| **TMEM accumulation** | `allocate_tensor_memory` | a huge fp32 accumulator without register pressure |
| **fp32 accumulate** | `gl.float32` acc | accuracy despite fp16/bf16 inputs |
| **L2-flushed autotuning** | `gemm_auto` | picking the best tiling per shape, measured honestly |

The benchmark in `main()` reports throughput as a percentage of cuBLAS — which is the right yardstick, since cuBLAS is the heavily-tuned vendor library. Getting within striking distance of it with a readable Python kernel is the whole appeal of Gluon.

---

## Where to go next

- Read the [Modal GPU Glossary](https://modal.com/gpu-glossary) end-to-end — it's the best free reference for the hardware terms used above ([SM](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor), [tensor core](https://modal.com/gpu-glossary/device-hardware/tensor-core), [thread-block cluster](https://modal.com/gpu-glossary/device-software/thread-block-cluster), [shared memory](https://modal.com/gpu-glossary/device-software/shared-memory)).
- Step through the pipeline widget again with `NB` in mind — fewer buffers means the green MMA stage would stall waiting on amber loads.
- Compare this with a plain-Triton `tl.dot` GEMM to see exactly which decisions the compiler was making for you — and which ones Gluon hands back.

The full kernel is on GitHub: [`Tanmaypatil123/kronos · kernels/gluon_gemm.py`](https://github.com/Tanmaypatil123/kronos/blob/main/kernels/gluon_gemm.py). Clone it, run it on a B200, and watch the tensor cores stay fed.

<script>
(function () {
  // ---- Widget 1: 2-CTA tiling highlight ----
  var tile = document.getElementById("gw-tile");
  if (tile) {
    var tdesc = document.getElementById("gw-tile-desc");
    var msgs = {
      0: "<b>CTA 0</b> (blue) computes the <b>top 128 rows</b> of the output tile. It loads its own half of A and shares the B tile with CTA 1.",
      1: "<b>CTA 1</b> (green) computes the <b>bottom 128 rows</b>. Its A-half is different, but the B columns are the same — so the cluster reuses B across both halves.",
      2: "Each grid program is a <b>2-CTA cluster</b>. The 256-row output tile is split: CTA&nbsp;0 owns the top 128 rows, CTA&nbsp;1 the bottom 128. They cooperate on one <code>tcgen05</code> MMA, so the B tile is shared between them."
    };
    var cells = tile.querySelectorAll(".gw-cell");
    function paint(h) {
      cells.forEach(function (c) {
        var own = c.getAttribute("data-cta");
        if (h === "2" || own === h) c.classList.remove("gw-dim");
        else c.classList.add("gw-dim");
      });
      tdesc.innerHTML = msgs[h];
    }
    tile.querySelectorAll("[data-h]").forEach(function (b) {
      b.addEventListener("click", function () { paint(b.getAttribute("data-h")); });
    });
    paint("2");
  }

  // ---- Widget 2: pipeline stepper ----
  var pipe = document.getElementById("gw-pipe");
  if (pipe) {
    var L = "load", R = "ready", M = "mma", F = "free";
    var steps = [
      { tag: "prologue", done: 0,
        txt: "Prologue — fire async TMA loads for K-chunks 0, 1 and 2 into the three ring buffers. The tensor cores haven't run yet; we are just filling the pipeline.",
        code: "for i in static_range(NB):          # NB = 3\n    mbarrier.expect(ready[i], a_pc + b_pc)\n    tma.async_copy_global_to_shared(a_desc, [off_m, i*BK], ready[i], a_bufs[i])\n    tma.async_copy_global_to_shared(b_desc, [i*BK, off_n], ready[i], b_bufs[i])",
        bufs: [[L, 0], [L, 1], [L, 2]] },
      { tag: "k = 0", done: 1,
        txt: "Buffer 0 has arrived. Issue the tcgen05 MMA on chunk 0 with use_acc=False, so it *initializes* the accumulator. Then prefetch chunk 3 into buffer 0.",
        code: "mbarrier.wait(ready[0], ph)\ntcgen05_mma(a_bufs[0], b_bufs[0], acc, use_acc=False, mbarriers=[empty[0]])\n# kk = 0 + 3 = 3 < 6  ->  prefetch chunk 3 into buffer 0",
        bufs: [[M, 0], [R, 1], [R, 2]] },
      { tag: "k = 1", done: 2,
        txt: "MMA chunk 1 into the accumulator (use_acc=True now — we add on top). Meanwhile buffer 0 is refilling with chunk 3 in the background.",
        code: "mbarrier.wait(ready[1], ph)\ntcgen05_mma(a_bufs[1], b_bufs[1], acc, use_acc=True, mbarriers=[empty[1]])\n# prefetch chunk 4 into buffer 1",
        bufs: [[L, 3], [M, 1], [R, 2]] },
      { tag: "k = 2", done: 3,
        txt: "MMA chunk 2. Buffer 1 refills with chunk 4. Notice the tensor cores never wait — every buffer they need was filled NB steps ago.",
        code: "mbarrier.wait(ready[2], ph)\ntcgen05_mma(a_bufs[2], b_bufs[2], acc, use_acc=True, mbarriers=[empty[2]])\n# prefetch chunk 5 into buffer 2  (last prefetch)",
        bufs: [[R, 3], [L, 4], [M, 2]] },
      { tag: "k = 3", done: 4,
        txt: "Buffer 0 now holds chunk 3 — MMA it. kk = 3 + 3 = 6 is NOT < num_k, so there is nothing left to prefetch; the pipeline begins to drain.",
        code: "mbarrier.wait(ready[0], ph)\ntcgen05_mma(a_bufs[0], b_bufs[0], acc, use_acc=True, mbarriers=[empty[0]])\n# kk = 6  ->  no prefetch",
        bufs: [[M, 3], [R, 4], [L, 5]] },
      { tag: "k = 4", done: 5,
        txt: "MMA chunk 4. Buffer 0 has no more work, so it goes idle. Buffer 2 finished loading chunk 5 and is ready for the final step.",
        code: "mbarrier.wait(ready[1], ph)\ntcgen05_mma(a_bufs[1], b_bufs[1], acc, use_acc=True, mbarriers=[empty[1]])",
        bufs: [[F, -1], [M, 4], [R, 5]] },
      { tag: "k = 5", done: 6,
        txt: "Final MMA, chunk 5. All six chunks are now folded into the accumulator.",
        code: "mbarrier.wait(ready[2], ph)\ntcgen05_mma(a_bufs[2], b_bufs[2], acc, use_acc=True, mbarriers=[empty[2]])",
        bufs: [[F, -1], [F, -1], [M, 5]] },
      { tag: "epilogue", done: 6,
        txt: "All 6 chunks summed. The full [BM × BN] result now sits in the accumulator (TMEM). Wait for the final MMA, then write the tile back out to C.",
        code: "mbarrier.wait(empty[last], lph)   # drain last MMA\nout = acc.load(reg_layout)        # TMEM -> registers",
        bufs: [[F, -1], [F, -1], [F, -1]] }
    ];
    var cls = { load: "gw-amber", ready: "gw-blue", mma: "gw-green", free: "gw-gray" };
    var lab = { load: "loading", ready: "ready", mma: "MMA", free: "free" };
    var bufEls = pipe.querySelectorAll("#gw-pipe-bufs .gw-buf");
    var pills = pipe.querySelectorAll("#gw-pipe-bufs .gw-pill");
    var stepEl = document.getElementById("gw-pipe-step");
    var descEl = document.getElementById("gw-pipe-desc");
    var codeEl = document.getElementById("gw-pipe-code");
    var progEl = document.getElementById("gw-pipe-prog");
    var backB = document.getElementById("gw-pipe-back");
    var nextB = document.getElementById("gw-pipe-next");
    var resetB = document.getElementById("gw-pipe-reset");
    var i = 0;
    function render() {
      var s = steps[i];
      stepEl.textContent = "step " + (i + 1) + " / " + steps.length + " · " + s.tag;
      descEl.innerHTML = s.txt;
      codeEl.textContent = s.code;
      progEl.style.width = (100 * s.done / 6) + "%";
      s.bufs.forEach(function (b, j) {
        var state = b[0], chunk = b[1];
        var p = pills[j];
        p.className = "gw-pill " + cls[state];
        p.textContent = chunk >= 0 ? lab[state] + " · c" + chunk : lab[state];
        if (state === "mma") bufEls[j].classList.add("act");
        else bufEls[j].classList.remove("act");
      });
      backB.disabled = i === 0;
      nextB.disabled = i === steps.length - 1;
    }
    nextB.addEventListener("click", function () { if (i < steps.length - 1) { i++; render(); } });
    backB.addEventListener("click", function () { if (i > 0) { i--; render(); } });
    resetB.addEventListener("click", function () { i = 0; render(); });
    render();
  }

  // ---- Widget 3: writeback flow ----
  var flow = document.getElementById("gw-flow");
  if (flow) {
    var fdesc = document.getElementById("gw-flow-desc");
    var nodes = flow.querySelectorAll(".gw-node");
    var detail = [
      "<b>TMEM</b> — the fp32 accumulator we summed every K-chunk into. <code>acc.load(reg_layout)</code> pulls it into registers using a thread↔element mapping consistent with num_warps.",
      "<b>Registers</b> — the tile now lives per-thread as <code>out</code> (still fp32). Next we narrow it: <code>out.to(dtype)</code> casts down to fp16/bf16.",
      "<b>Shared memory</b> — <code>c_smem</code> holds the cast tile. <code>fence_async_shared()</code> makes these writes visible to the TMA engine before the store.",
      "<b>Global memory</b> — <code>tma.async_copy_shared_to_global</code> streams the BM×BN tile to C at (off_m, off_n); <code>tma.store_wait(pendings=0)</code> waits for it to drain."
    ];
    var playB = document.getElementById("gw-flow-play");
    var resetB2 = document.getElementById("gw-flow-reset");
    var timer = null;
    function show(idx) {
      nodes.forEach(function (n, k) {
        n.classList.toggle("on", k === idx);
        n.classList.toggle("off", idx >= 0 && k !== idx);
      });
      fdesc.innerHTML = idx >= 0 ? detail[idx] : "fp32 accumulate the whole way, narrowed to fp16/bf16 only at the shared-memory hop. Press <b>Play</b> or click a stage.";
    }
    nodes.forEach(function (n) {
      n.addEventListener("click", function () {
        if (timer) { clearInterval(timer); timer = null; }
        show(parseInt(n.getAttribute("data-i"), 10));
      });
    });
    playB.addEventListener("click", function () {
      if (timer) clearInterval(timer);
      var k = 0; show(0);
      timer = setInterval(function () {
        k++;
        if (k >= nodes.length) { clearInterval(timer); timer = null; return; }
        show(k);
      }, 1100);
    });
    resetB2.addEventListener("click", function () {
      if (timer) { clearInterval(timer); timer = null; }
      show(-1);
    });
    show(-1);
  }
})();
</script>

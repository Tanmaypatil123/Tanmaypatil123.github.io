---
title: "RMSNorm Backward: From Derivation to a Triton Kernel"
author: Tanmay
pubDatetime: 2025-11-01T13:30:00Z
slug: rmsnorm-backward-derivation-triton-kernel
featured: true
draft: false
tags:
  - rmsnorm
  - normalization
  - backpropagation
  - triton
  - pytorch
  - gpu-kernels
  - deep-learning
ogImage: ../../../assets/images/rms_norm.png
description: "Derive RMSNorm backward step by step and implement a Triton kernel with PyTorch, numerics tips"
canonicalURL: "https://tanmaypatil123.github.io/posts/2025/rmsnorm-backward-derivation-triton-kernel"  # add your canonical URL if cross-posted/
---

> **TL;DR**: We derive RMSNorm backward in plain math (no scary notation), implement a forward+backward Triton kernel.

## Why RMSNorm ?

RMSNorm introduces No mean subtraction, fewer ops, simpler math; used in modern LLMs. Good “real world” relevance over LayerNorm.

---

## Cheat-sheet: tiny math facts we’ll use

- Square: $$ \frac{\partial}{\partial x}(x^2)=2x $$
- Square root:$$  \frac{d}{dz}\sqrt{z}=\frac{1}{2\sqrt{z}} $$
- Reciprocal: $$ \frac{d}{dz}\frac{1}{z}=-\frac{1}{z^2} $$
- Product rule: $$ (uv)'=u'v+uv' $$
- Kronecker delta: $$\frac{\partial x_i}{\partial x_j}=\delta_{ij} $$ (1 if \(i=j\), else 0)

---

## Forward pass: RMSNorm

We’ll start with one feature vector $ x \in \mathbb{R}^{N} $ (think a single row of shape `[N]`).  
RMSNorm scales by the **root-mean-square** and (optionally) applies per-feature scale $\gamma$ and shift $\beta$ .

## Equations
<!-- $$ \frac{\partial}{\partial x}(x^2)=2x $$ -->
$$
\tag{1} a \;=\; \frac{1}{N}\sum_{i=1}^{N} x_i^2 \qquad\text{(mean of squares)}
$$

$$
\tag{2} \mathrm{rms} \;=\; \sqrt{a + \epsilon} \qquad\text{(root-mean-square with stability \(\epsilon>0\))}
$$

$$
\tag{3} \widehat{x}_i \;=\; \frac{x_i}{\mathrm{rms}} \qquad\text{(normalized activations)}
$$

$$
\tag{4} y_i \;=\; \gamma_i\,\widehat{x}_i \;+\; \beta_i \qquad\text{(scale \& shift)}
$$

### What each symbol means

- $ x \in \mathbb{R}^{N} $: input vector (last/feature dimension length $N$)  
- $a \in \mathbb{R}$: mean of squared elements of $x$  
- $\epsilon \in \mathbb{R}_{>0}$: small constant for numerical stability  
- $\mathrm{rms} \in \mathbb{R}_{>0}$: root-mean-square of $x$  
- $\widehat{x} \in \mathbb{R}^{N}$: normalized $x$  
- $\gamma, \beta \in \mathbb{R}^{N}$ (or scalars): per-feature scale and shift  
- $y \in \mathbb{R}^{N}$: output

## Triton Kernel : Forward Pass

```python
## Import things which are needed.
import triton
import triton.language as tl
import torch

@triton.jit
def rms_norm_forward(
    Y, Y_stride:tl.constexpr,
    X,X_stride : tl.constexpr,
    gamma, gamma_stride : tl.constexpr,
    r,r_stride : tl.constexpr,
    N:tl.constexpr,
    eps : tl.constexpr,
    BLOCK_SIZE : tl.constexpr
):
    pid = tl.program_id(0)
    offs = tl.arange(0,BLOCK_SIZE)
    mask = offs < N

    # finding accurate pointers to the memeory location.
    Y += pid * Y_stride
    X += pid * X_stride
    r += pid * r_stride

    # loading x and gamma to smem from hbm
    x_row = tl.load(X+offs , mask = mask , other = 0 ).to(tl.float32)
    gamma_row = tl.load(gamma + offs , mask = mask , other = 0)

    # sum of x ^ 2 / N
    row_var = tl.sum(x_row * x_row,axis = 0) / N

    # 1 / (a + eps) ^ 1/2
    inv_rms = tl.rsqrt(row_var + eps)

    # storing inv_rms for backward pass we will need this again.
    tl.store(r , inv_rms)

    # final calculation
    norm = x_row * inv_rms * gamma_row
    tl.store(Y + offs , norm , mask = mask)
```



## Backward Pass : RMSNorm

We work per row: $ x \in \mathbb{R}^N $.
Everything flows through
$ a \rightarrow \mathrm{rms} \rightarrow \mathrm{inv} \rightarrow \hat{x}$

1. $a=\frac{1}{N}\sum_i x_i^2 
\;\;\Rightarrow\;\;
\frac{\partial a}{\partial x_j}=\frac{2}{N}x_j.$

2. $\mathrm{rms}=\sqrt{a+\epsilon}
\;\;\Rightarrow\;\;
\frac{\partial\,\mathrm{rms}}{\partial x_j}
= \frac{1}{2\sqrt{a+\epsilon}}\cdot\frac{\partial a}{\partial x_j}
= \frac{1}{2\sqrt{a+\epsilon}}\cdot\frac{2}{N}x_j
= \frac{x_j}{N\,\mathrm{rms}}.$

3. $\mathrm{inv}=\frac{1}{\mathrm{rms}}
\;\;\Rightarrow\;\;
\frac{\partial\,\mathrm{inv}}{\partial x_j}
= -\frac{1}{\mathrm{rms}^2}\,\frac{\partial\,\mathrm{rms}}{\partial x_j}
= -\frac{x_j}{N\,\mathrm{rms}^3}.$
---


Let $x\in\mathbb{R}^N$ and

$$
a=\frac{1}{N}\sum_{k=1}^N x_k^2,\qquad
\mathrm{rms}=\sqrt{a+\varepsilon},\qquad
\mathrm{inv}=\frac{1}{\mathrm{rms}},\qquad
\hat{x}=x\cdot \mathrm{inv}.
$$

Jacobian of $\hat{x}$ w.r.t.\ $x$.
For each pair $(i,j)$,
$$
\frac{\partial \hat{x}_i}{\partial x_j}
= \frac{\partial (x_i\,\mathrm{inv})}{\partial x_j}
= \left(\frac{\partial x_i}{\partial x_j}\right)\mathrm{inv}
    + x_i \left(\frac{\partial\,\mathrm{inv}}{\partial x_j}\right)
= \delta_{ij}\,\mathrm{inv} + x_i\!\left(-\,\frac{x_j}{N\,\mathrm{rms}^3}\right),
$$
i.e.
$$
\boxed{\;
\frac{\partial \hat{x}_i}{\partial x_j}
= \frac{\delta_{ij}}{\mathrm{rms}}
   - \frac{x_i x_j}{N\,\mathrm{rms}^3}
\;}
$$

Matrix form.
Stacking $\partial \hat{x}_i/\partial x_j$ gives the Jacobian
$$
\boxed{\;
\frac{\partial \hat{x}}{\partial x}
= \frac{1}{\mathrm{rms}}\,I
  - \frac{1}{N\,\mathrm{rms}^3}\,x x^{\!\top}
\;}
$$

Index / layout convention,
We treat $x$ as a $\emph{column vector}$ ($N\times 1$).  
Then the Jacobian $J=\partial \hat{x}/\partial x$ is an $N\times N$ matrix with
$$
J_{ij}=\frac{\partial \hat{x}_i}{\partial x_j},
$$
so index $i$ corresponds to the $\textbf{row}$ (output component) and index $j$ to the $\textbf{column}$ (input component).  
If you prefer a row-vector convention for $x^\top$, the Jacobian appears transposed accordingly.

### Chain rule → \(dX\)

$$
\frac{\partial \mathcal{L}}{\partial x_j}
= \sum_{i=1}^N h_i \left(\frac{\delta_{ij}}{\mathrm{rms}} - \frac{x_i x_j}{N\,\mathrm{rms}^3}\right)
= \frac{h_j}{\mathrm{rms}} - \frac{x_j}{N\,\mathrm{rms}^3}\sum_i h_i x_i.
$$

Vector form:
$$
\boxed{\, dX \;=\; \frac{h}{\mathrm{rms}} \;-\; \frac{x\,\langle h,x\rangle}{N\,\mathrm{rms}^3} \,}
\quad\text{where}\quad
\langle h,x\rangle=\sum_i h_i x_i.
$$

Using $\hat{x}=x\cdot\text{inv}$ and $\sum_i h_i \hat{x}_i=\langle h,x\rangle\cdot\text{inv}$,
$$
dX \;=\; \frac{\text{inv}}{N}\Big(Nh - \hat{x}\sum_i h_i\hat{x}_i\Big).
$$

in the derivation 
$ℎ$ is just a shorthand for the upstream grad ($dy$) scaled by the weight ($gamma$).
## Triton Kernel : Backward Pass

```python
@triton.jit
def rms_norm_backward(
    dY, dY_stride : tl.constexpr,
    dX, dX_stride : tl.constexpr,
    X,   X_stride : tl.constexpr,
    gamma,  gamma_stride : tl.constexpr,
    r,   r_stride : tl.constexpr,
    N     : tl.constexpr,
    eps        : tl.constexpr,
    BLOCK_SIZE : tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    dY += pid * dY_stride
    X  += pid *  X_stride
    r  += pid *  r_stride

    dX = dY

    dY_row = tl.load(dY + offs, mask = mask, other = 0).to(tl.float32)
    X_row  = tl.load(X  + offs, mask = mask, other = 0).to(tl.float32)
    gamma_row  = tl.load(gamma  + offs, mask = mask, other = 0).to(tl.float32)

    # Get saved row variance
    inv_var = tl.load(r).to(tl.float32)
    # normed = x * inv
    normed = X_row * inv_var
    # h = y * r
    dY_gamma = dY_row * gamma_row
    # sum of h * normed
    rowsum_dY_normed = tl.sum(dY_gamma * normed, axis = 0)

    # inv / N (N * h - normed * sum of h and normed)
    output = inv_var/N * (N*dY_gamma - normed*rowsum_dY_normed)

    #storing back the output
    tl.store(dX + offs, output, mask = mask)
```

## Use Triton kernels with pytorch.

```python
class RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, X: torch.Tensor,W : torch.Tensor ,eps : float 
    ):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1,dim)
        n_rows , n_cols = X.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        device = X.device
        Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = device)
        r = torch.empty(n_rows, dtype = torch.float32, device = device)

        with torch.cuda.device(device):
            rms_norm_forward[(n_rows,)](
                Y, Y.stride(0),
                X, X.stride(0),
                W, W.stride(0),
                r, r.stride(0),
                n_cols, eps,
                BLOCK_SIZE = BLOCK_SIZE
            )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)
    @staticmethod
    def backward(ctx, dY : torch.Tensor):
        shape = dY.shape
        dim : int = shape[-1]
        dY = dY.view(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows : int
        n_cols : int
        n_rows, n_cols = dY.shape
        dX = dY

        with torch.cuda.device(dY.device):
            rms_norm_backward[(n_rows,)](
                dY, dY.stride(0),
                dX, dX.stride(0),
                X,  X .stride(0),
                W,  W .stride(0),
                r,  r .stride(0),
                n_cols, ctx.eps,
                BLOCK_SIZE = ctx.BLOCK_SIZE,
            )
        dX = dX.view(*shape)
        return dX, None, None, None
```
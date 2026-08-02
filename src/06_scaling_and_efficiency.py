# %% [markdown]
# # 06 — Scaling Laws & Making It Fast
#
# **Goal:** answer "how big a model should I train, for how long?" with
# arithmetic rather than vibes — then make the training run actually fast.
#
# **Time:** 45 min.
#
# ## In plain language
#
# **What you're doing:** learning to answer two questions with arithmetic instead
# of guesswork — *how big should my model be?* and *why is my training so slow?*
#
# **The everyday version of scaling laws.** You have a fixed budget and you're
# opening a restaurant. Do you spend it on one brilliant expensive chef, or on a
# decent chef plus far better ingredients?
#
# Turns out this has a *measured answer* for language models. If you have a fixed
# amount of GPU time, there's a mathematically best split between "bigger model"
# and "more data" — and for years everybody got it wrong in the same direction,
# building models far too large for the data they were fed. The correction
# (Chinchilla, 2022) says roughly **20 tokens of text per parameter of model**.
#
# That single number tells you a 124M-parameter model wants about 2.5 billion
# tokens. Not 100 million (undertrained), not 50 billion (wasted compute).
#
# **The everyday version of the speed half.** Your GPU has a top speed, and your
# training loop probably runs at a fraction of it — usually because the GPU is
# sitting idle waiting for something else, exactly like a fast chef waiting on a
# slow dishwasher. This notebook teaches you to measure that fraction and find
# the dishwasher.
#
# **What you'll have at the end:**
#
# - a formula that tells you what model size to train for a given budget
# - a measurement of how much of your GPU you're actually using
# - three optimizations, in order of how much they pay off
#
# **What to expect:** the optimizations are worth roughly **2–4× total speed** on
# a real run. That turns a 5-hour job into 1.5 hours. It's the difference between
# trying three ideas today and trying one.
#
# **This notebook is mostly arithmetic**, and it's the arithmetic that stops you
# wasting a weekend. Read the numbers, not the code.
#
# ## Part A — Scaling laws
#
# ### Chinchilla, and what it actually says
#
# The Chinchilla result (Hoffmann et al., 2022) fits loss as a function of
# parameters `N` and tokens `D`:
#
# ```
# L(N, D) = E + A/N^alpha + B/D^beta
#           ^     ^          ^
#           |     |          `- finite data
#           |     `- finite model capacity
#           `- irreducible entropy of language
# ```
#
# Minimizing loss subject to a compute budget `C ≈ 6ND` gives the famous
# result: **scale N and D equally**, at roughly **20 tokens per parameter**.
#
# This corrected GPT-3, which at 175B params and 300B tokens (1.7 tok/param)
# was badly undertrained — a Chinchilla-optimal model at that budget would have
# been ~4× smaller and much better.

# %%
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True

# Coefficients from the Chinchilla paper (Approach 3).
E, A, B = 1.69, 406.4, 410.7
ALPHA, BETA = 0.34, 0.28


def predicted_loss(n_params: float, n_tokens: float) -> float:
    return E + A / n_params**ALPHA + B / n_tokens**BETA


def compute_flops(n_params: float, n_tokens: float) -> float:
    return 6 * n_params * n_tokens


print("Chinchilla-optimal points:\n")
print(f"{'params':>10}{'tokens':>12}{'tok/param':>11}{'pred loss':>11}{'FLOPs':>11}")
print("-" * 55)
for n in [1e8, 4e8, 1e9, 7e9, 7e10]:
    d = 20 * n
    print(f"{n/1e9:>9.2f}B{d/1e9:>11.1f}B{d/n:>11.0f}{predicted_loss(n, d):>11.3f}"
          f"{compute_flops(n, d):>11.1e}")

# %% [markdown]
# ### The part everyone gets wrong
#
# Chinchilla optimizes **training** compute. It says nothing about inference.
#
# If you'll serve a model billions of times, a *smaller model trained far past
# Chinchilla* is the better deal: slightly worse loss, but permanently cheaper
# and faster to run. That's why:
#
# | model | params | tokens | tok/param |
# |---|---|---|---|
# | Chinchilla | 70B | 1.4T | 20 |
# | Llama 3 8B | 8B | 15T | **1875** |
# | SmolLM2 1.7B | 1.7B | 11T | **6470** |
#
# These are ~100× past "optimal" and it is absolutely the right call — they're
# meant to be deployed, not to win a training-efficiency benchmark.
#
# **The lesson for you:** you are not compute-constrained in the Chinchilla
# sense, you're *time* constrained. Train a small model on as many tokens as your
# patience allows.

# %%
print("What can a single RTX 5090 do?\n")
GPU_FLOPS = 200e12       # ~200 TFLOP/s sustained bf16, a realistic (not peak) figure
MFU = 0.40               # model FLOPs utilisation; 35-50% is good for a solo setup

print(f"{'budget':>9}{'usable FLOPs':>15}{'Chinchilla N':>15}{'Chinchilla D':>14}")
print("-" * 53)
for hours in [1, 6, 24, 168]:
    flops = GPU_FLOPS * MFU * hours * 3600
    # C = 6ND and D = 20N  ->  N = sqrt(C/120)
    n_opt = math.sqrt(flops / 120)
    label = f"{hours}h" if hours < 168 else "1 week"
    print(f"{label:>9}{flops:>15.1e}{n_opt/1e6:>14.0f}M{20*n_opt/1e9:>13.1f}B")

# %% [markdown]
# So: **6 hours buys a compute-optimal ~120M model; a full day buys ~240M.**
# That's why this course targets 124M — it's genuinely what one consumer GPU
# pretrains in an afternoon, not an arbitrary choice.
#
# Note these are *optimal* points under Chinchilla. As argued above, you'd
# often rather train the 120M model for 24 hours than the 240M model — same
# compute, worse training loss, but a model that's 2× cheaper forever after.
#
# For anything bigger, you **fine-tune someone else's pretrained model** — which
# is exactly what notebooks 08 onward do.

# %% [markdown]
# ## Part B — Measuring what you've got: MFU
#
# **Model FLOPs Utilisation** is the fraction of your GPU's theoretical peak
# that you're actually using. It's the single best number for "is my training
# loop leaving performance on the table?"
#
# ### Why you need a *ratio*, not a speed
#
# "4,200 tokens/sec" tells you nothing on its own. Is that good? It depends on
# the model size, the sequence length, and the card. Halve your model and tokens
# per second doubles — you have not improved anything, you have just done less
# work per token.
#
# MFU normalizes all of that away:
#
# ```
#           FLOPs your model actually needs, per second
# MFU  =   ---------------------------------------------
#              FLOPs your GPU can theoretically do
# ```
#
# The numerator is arithmetic — a known function of your architecture, computed
# below. The denominator is a spec-sheet number. The ratio answers one question:
# **of the maths your GPU could have done this second, what fraction went into
# your model?** Everything else — waiting on data, launching kernels, moving
# memory, recomputing activations — is the gap.
#
# It is a *speedometer relative to the speed limit*, and unlike tokens/sec it is
# comparable across model sizes, sequence lengths, and even across GPUs.
#
# | MFU | verdict |
# |---|---|
# | <15% | something is badly wrong — data loading, tiny batches, no autocast |
# | 20–35% | typical unoptimized loop |
# | 35–50% | good for a single consumer GPU |
# | 50–60% | excellent; what well-tuned large runs achieve |
#
# **Nobody gets 100%, and you should not chase it.** Peak FLOPS assumes every
# clock cycle is a fused multiply-add on data already sitting in registers. Real
# training reads and writes memory constantly, and memory bandwidth — not
# arithmetic — is usually the actual ceiling. Above ~50% on a consumer card you
# are into diminishing returns; spend the effort on a better dataset instead.
#
# **Where the missing MFU usually hides**, in the order worth checking:
#
# | symptom | likely cause | fix |
# |---|---|---|
# | MFU < 15%, GPU util spiky | data loader starving the GPU | `.bin` on the Linux FS (notebook 01), memmap |
# | MFU ~20%, GPU util high | fp32 instead of bf16 | `autocast` |
# | MFU ~25% on a small model | kernel launch overhead dominates | `torch.compile`, larger batch |
# | MFU drops when you raise `block_size` | attention's T² term | expected — it is real work, not waste |
#
# That last row matters: MFU falling as sequence length grows is not necessarily
# a regression. The quadratic attention term is genuine computation, and the
# formula below counts it, which is why this estimate is more honest than the
# "6ND" rule of thumb you will see quoted elsewhere.

# %%
def model_flops_per_token(n_layer, n_embd, n_head, block_size, vocab_size) -> float:
    """Forward+backward FLOPs per token (the PaLM-paper estimate)."""
    # 12 * L * d^2 counts the four attention projections (4d^2) and the MLP
    # (8d^2). The factor 6 = 2 (fwd) + 4 (bwd).
    dense = 6 * 12 * n_layer * n_embd**2
    # Attention's quadratic term: scores and the value-weighted sum.
    attn = 6 * 2 * n_layer * block_size * n_embd
    # The vocabulary projection is significant for small models with big vocabs.
    head = 6 * 2 * n_embd * vocab_size
    return dense + attn + head


def report_mfu(tokens_per_sec: float, cfg: dict, peak_flops: float = 200e12) -> float:
    f = model_flops_per_token(**cfg)
    achieved = tokens_per_sec * f
    return achieved / peak_flops


cfg_124m = dict(n_layer=12, n_embd=768, n_head=12, block_size=1024, vocab_size=50257)
print(f"FLOPs per token (124M model): {model_flops_per_token(**cfg_124m):.2e}\n")
print(f"{'tok/s':>10}{'MFU':>9}")
print("-" * 19)
for tps in [20_000, 50_000, 85_000, 120_000]:
    print(f"{tps:>10,}{100*report_mfu(tps, cfg_124m):>8.1f}%")

# %% [markdown]
# Run notebook 04's loop, note its tok/s, and compute your MFU. If it's under
# 20%, the optimizations below are where your speed is hiding.

# %% [markdown]
# ## Part C — The optimizations, in order of payoff
#
# ### 1. Mixed precision (bf16) — biggest single win
#
# 2× the memory bandwidth and access to tensor cores. Already in notebook 04.
#
# ### 2. `torch.compile` — 1.3–2×, one line
#
# Traces your model into a graph and generates fused Triton kernels. The main
# win is **kernel fusion**: instead of ten separate elementwise kernels each
# reading and writing HBM, you get one. Costs 1–2 minutes of compile time.
#
# ### 3. FlashAttention — already yours via SDPA
#
# `F.scaled_dot_product_attention` dispatches to FlashAttention when shapes
# allow. It never materializes the (B, nh, T, T) score matrix, making attention
# memory O(T) instead of O(T²).
#
# ### 4. Gradient checkpointing — trades ~30% speed for ~60% activation memory
#
# Don't store activations; recompute them during the backward pass. Use it when
# you're memory-bound and can't otherwise fit the batch you want.
#
# ### 5. Fused optimizer — a few percent, free
#
# `AdamW(..., fused=True)`. Already in notebook 04.

# %%
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))
from llmfs.model import GPT, GPTConfig  # noqa: E402


def benchmark_step(model, B, T, vocab, n_steps=12, autocast=True, warmup=4):
    """Median tokens/sec over n_steps of forward+backward+step."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4,
                            fused=device.startswith("cuda"))
    x = torch.randint(0, vocab, (B, T), device=device)
    y = torch.randint(0, vocab, (B, T), device=device)

    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if autocast and device == "cuda" else torch.autocast(device_type="cpu", enabled=False))

    times = []
    for i in range(n_steps + warmup):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with ctx:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:                 # discard warmup + compile time
            times.append(time.time() - t0)

    med = sorted(times)[len(times) // 2]
    return (B * T) / med


if device == "cuda":
    B, T, V = 8, 512, 50257
    small = GPTConfig(vocab_size=V, block_size=T, n_layer=6, n_head=6, n_embd=384)

    print(f"{'configuration':<34}{'tok/s':>12}{'speedup':>10}")
    print("-" * 56)

    m = GPT(small).to(device)
    base = benchmark_step(m, B, T, V, autocast=False)
    print(f"{'fp32 baseline':<34}{base:>12,.0f}{1.0:>9.2f}x")

    m = GPT(small).to(device)
    amp = benchmark_step(m, B, T, V, autocast=True)
    print(f"{'+ bf16 autocast':<34}{amp:>12,.0f}{amp/base:>9.2f}x")

    m = torch.compile(GPT(small).to(device))
    comp = benchmark_step(m, B, T, V, autocast=True, warmup=8)
    print(f"{'+ torch.compile':<34}{comp:>12,.0f}{comp/base:>9.2f}x")

    best_cfg = dict(n_layer=6, n_embd=384, n_head=6, block_size=T, vocab_size=V)
    print(f"\nMFU at best: {100*report_mfu(comp, best_cfg):.1f}%")

    # This is the one benchmark in the course that is purely about your machine,
    # so it is also the one where a recorded history is most useful: a driver
    # update, a new PyTorch, or a different batch size all move it.
    from llmfs.bench import log_run

    log_run(
        stage="06_scaling_and_efficiency",
        metrics={
            "tokens_per_sec": comp,
            "mfu": report_mfu(comp, best_cfg),
            "speedup_bf16": amp / base,
            "speedup_compile": comp / base,
            "tokens_per_sec_fp32": base,
        },
        key="tokens_per_sec",
        config={"n_layer": 6, "n_embd": 384, "batch": B, "block_size": T,
                "vocab_size": V},
        notes="fp32 -> bf16 -> torch.compile",
    )
else:
    print("(benchmark requires CUDA — the relative gains are the point,")
    print(" and they don't reproduce meaningfully on CPU)")
    print("Nothing recorded to BENCHMARK.md: a CPU number here would not be")
    print("comparable to the GPU runs it would sit next to.")

# %% [markdown]
# ## Gradient checkpointing
#
# The trade: recompute activations in the backward pass instead of storing them.
# Memory drops from `O(n_layers)` to roughly `O(1)` in the layer dimension, at
# the cost of one extra forward pass (~30% slower).
#
# **Use it when** you want a longer sequence or bigger micro-batch than fits.
# **Don't use it when** you already fit comfortably — it's pure slowdown.

# %%
from torch.utils.checkpoint import checkpoint


class CheckpointedGPT(GPT):
    """GPT with activation checkpointing on every block."""

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            if self.training:
                # use_reentrant=False is the modern implementation; the old
                # reentrant version interacts badly with autocast and hooks.
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


if device == "cuda":
    cfg = GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768)
    print(f"{'model':<26}{'peak VRAM':>12}{'tok/s':>11}")
    print("-" * 49)
    for name, klass in [("standard", GPT), ("checkpointed", CheckpointedGPT)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m = klass(cfg).to(device)
        m.train()
        tps = benchmark_step(m, 4, 1024, 50257, n_steps=6)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{name:<26}{peak:>11.2f}G{tps:>11,.0f}")
        del m
    torch.cuda.empty_cache()
else:
    print("(needs CUDA to show the memory difference)")

# %% [markdown]
# ## The Muon optimizer
#
# Worth knowing because it currently holds the nanoGPT speedrun records
# (modded-nanogpt got GPT-2-quality down to minutes on 8×H100 largely thanks to
# it).
#
# **The idea:** AdamW treats every parameter independently. But a weight
# *matrix* has structure — an update whose singular values are wildly uneven
# effectively moves in only a few directions. Muon orthogonalizes the momentum
# matrix (via a few Newton–Schulz iterations) so the update has roughly equal
# singular values, spreading the step across all directions.
#
# Applies to **2-D parameters only**. Embeddings, the LM head, norms, and biases
# stay on AdamW.

# %%
@torch.no_grad()
def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal factor of G's polar decomposition.

    Cheap because it's only matmuls — no SVD. The quintic coefficients are the
    ones tuned by Keller Jordan for fast convergence in bf16.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B_ = b * A + c * A @ A
        X = a * X + B_ @ X
    return (X.T if transposed else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Minimal Muon. Use it ONLY for 2-D params; keep AdamW for the rest."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                g = newton_schulz_orthogonalize(g, group["ns_steps"])
                # Scale by the shape ratio so the update norm is comparable
                # across differently-shaped matrices.
                p.add_(g, alpha=-group["lr"] * max(1.0, p.size(0) / p.size(1)) ** 0.5)
        return loss


# Verify the orthogonalization does what it claims.
G = torch.randn(128, 64)
O = newton_schulz_orthogonalize(G, steps=5)
sv_before = torch.linalg.svdvals(G)
sv_after = torch.linalg.svdvals(O.float())
print("singular values before:", f"max {sv_before.max():.3f}  min {sv_before.min():.3f}  "
      f"ratio {sv_before.max()/sv_before.min():.1f}")
print("singular values after: ", f"max {sv_after.max():.3f}  min {sv_after.min():.3f}  "
      f"ratio {sv_after.max()/sv_after.min():.1f}")
print(f"\ncondition number improved {sv_before.max()/sv_before.min():.1f} -> "
      f"{sv_after.max()/sv_after.min():.1f}")
print("Not exactly 1.0 — 5 Newton-Schulz steps is a deliberate approximation,")
print("trading exactness for speed. But the update is now far more evenly")
print("spread across directions instead of dominated by a few singular values.")
print("Raise ns_steps to see it converge closer to 1.0, at more cost per step.")

# %% [markdown]
# ### Wiring Muon into a real model
#
# ```python
# hidden = [p for n, p in model.named_parameters()
#           if p.dim() == 2 and "wte" not in n and "head" not in n]
# rest   = [p for n, p in model.named_parameters()
#           if p.dim() != 2 or "wte" in n or "head" in n]
#
# optimizers = [
#     Muon(hidden, lr=0.02, momentum=0.95),
#     torch.optim.AdamW(rest, lr=3e-4, betas=(0.9, 0.95), fused=True),
# ]
# # in the loop: for opt in optimizers: opt.step(); opt.zero_grad(set_to_none=True)
# ```
#
# Reported gains are ~1.5–2× fewer steps to a target loss. Worth trying once
# your AdamW baseline is solid — not before, or you won't know what helped.

# %% [markdown]
# ## Quick reference: what to do when
#
# | symptom | first thing to try |
# |---|---|
# | OOM | halve micro_batch, double grad_accum |
# | still OOM | gradient checkpointing |
# | still OOM | shorter block_size, or a smaller model |
# | slow, low MFU | `torch.compile`, confirm bf16 is on |
# | slow, GPU idle in `nvidia-smi` | data loading — is the `.bin` on `/mnt/c/`? |
# | loss spikes | lower LR, tighten grad_clip |
# | want faster convergence | try Muon on the 2-D params |
#
# ## Exercises
#
# 1. **Measure your MFU** on notebook 04's real run. Then enable
#    `torch.compile` and measure again.
# 2. **Verify the Chinchilla ratio.** Train three 10M models on 50M / 200M /
#    800M tokens. Plot final val loss against tokens on a log-log axis and
#    compare the slope to Chinchilla's `beta = 0.28`.
# 3. **Muon vs AdamW.** Same model, same steps, both optimizers. Plot val loss.
#
# **Next:** `07_sft_from_scratch.ipynb` — the model can complete text. Now teach
# it to follow instructions.

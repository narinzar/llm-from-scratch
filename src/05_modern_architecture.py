# %% [markdown]
# # 05 — From GPT-2 (2019) to Llama (2024+)
#
# **Goal:** implement the four architecture changes that separate a modern LLM
# from GPT-2, understand the problem each one solves, and measure the difference.
#
# **Time:** 45–60 min.
#
# ## In plain language
#
# **What you're doing:** taking the transformer you built in notebook 03 — which
# is a 2019 design — and upgrading four parts of it to what everyone actually
# uses in 2024+.
#
# **The everyday version.** Cars in 1990 and 2024 have the same layout: engine,
# four wheels, steering wheel. Nobody reinvented the car. But fuel injection
# replaced carburettors, discs replaced drums, and the result is meaningfully
# better while looking identical from outside.
#
# Transformers are the same story. The 2017 design was right. Four parts got
# swapped, each fixing one specific annoyance, and modern models are the same
# machine with better components.
#
# **What you'll have at the end:** a Llama-style model — the same architecture
# family as Llama, Qwen and Mistral — with each of the four changes implemented
# and measured, so you can see what each one actually bought.
#
# **What to expect:** none of these is dramatic on its own. RMSNorm makes norms
# ~10% cheaper. GQA cuts inference memory 4×. The quality gains are small enough
# that you can't see them on a toy model. **Don't expect a revelation** — expect
# to understand why the code in every modern model's repo looks the way it does,
# and to be able to read it without confusion.
#
# **The single most useful thing here** is probably RoPE. It's why modern models
# can handle 128k-token contexts when GPT-2 was stuck at 1024. You'll implement
# it and then *demonstrate* the extrapolation yourself.
#
# ## What actually changed
#
# The transformer block is remarkably unchanged since 2017. Four things were
# swapped, each fixing a specific, identifiable weakness:
#
# | GPT-2 (2019) | Modern (Llama/Qwen/Mistral) | fixes |
# |---|---|---|
# | learned position embeddings | **RoPE** | can't extrapolate past training length |
# | LayerNorm | **RMSNorm** | ~10–15% of norm compute is wasted |
# | GELU MLP | **SwiGLU** | slightly worse quality per parameter |
# | multi-head attention | **GQA** | KV cache dominates inference memory |
#
# None is revolutionary. Together they're worth a meaningful chunk of quality
# and a large chunk of inference efficiency.
#
# ### The one-line intuition for each
#
# Before the maths, here is what each change is *for*. Come back to this after
# you have implemented them and it should read as obvious.
#
# **RoPE — "position as a rotation, not a lookup."** GPT-2 learned a separate
# vector for position 0, 1, 2, … up to 1024, then stopped. Ask it about position
# 1025 and there is simply nothing there — like a book with numbered pages where
# page 1025 was never printed. RoPE instead *rotates* each query and key by an
# angle proportional to its position. Rotation is a formula, not a table, so
# position 5000 is as computable as position 5. Better still, the dot product
# between two rotated vectors depends only on the **difference** of their angles
# — so attention naturally sees *relative* distance, which is what actually
# matters. "The word three tokens back" means the same thing at the start of a
# document and 4,000 tokens in.
#
# **RMSNorm — "the mean subtraction wasn't doing anything."** LayerNorm centres
# activations (subtract the mean) then scales them (divide by the standard
# deviation). Someone checked whether the centring step mattered. It mostly does
# not — the scaling is what stabilises training. Dropping it removes a mean, a
# subtraction, and a bias term from every norm in the network. Same quality,
# ~10-15% less norm compute, and norms run constantly. This is the least
# interesting change and the easiest free win.
#
# **SwiGLU — "let the network decide what to let through."** A standard MLP
# applies the same fixed nonlinearity to everything. A *gated* MLP computes two
# projections and multiplies them: one is the content, the other is a learned
# gate deciding how much of that content passes. Think of a dimmer switch per
# dimension, set by the input itself, instead of one fixed rule for all inputs.
# It costs a third matrix, so implementations shrink the hidden dimension to
# `(8/3)d` to keep the parameter count matched — and still come out ahead.
#
# **GQA — "the KV cache is eating your VRAM."** At inference the model caches a
# key and value vector for every past token, for every head, in every layer.
# With 32 heads that cache dominates memory and grows with every token you
# generate. GQA has several query heads *share* one key/value pair — 32 query
# heads over 8 KV pairs cuts the cache 4×. The queries still differ, so the heads
# still ask different questions; they just consult a shared index. Quality cost
# is small; memory saving is enormous, and it is what makes long-context serving
# affordable.
#
# Notice the pattern: **three of the four are about inference, not training.**
# The field learned that a model is trained once and served billions of times, so
# architecture drifted toward whatever makes serving cheap.

# %%
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} on {device}")

# %% [markdown]
# ## 1. RoPE — Rotary Position Embeddings
#
# ### The problem with learned positions
#
# GPT-2 has an `nn.Embedding(block_size, n_embd)` — one learned vector per
# position. Three consequences:
#
# 1. **Hard length limit.** Position 1025 has no embedding. The model cannot
#    process longer sequences at all.
# 2. **No extrapolation.** Even if you added rows, they'd be untrained noise.
# 3. **Absolute, not relative.** What matters linguistically is usually "3
#    tokens back", not "at index 47". The model has to learn relative distance
#    indirectly from absolute positions.
#
# ### The RoPE idea
#
# Don't *add* anything to the embeddings. Instead, **rotate** the query and key
# vectors by an angle proportional to their position.
#
# Treat consecutive pairs of dimensions as 2-D points and rotate each pair by
# `m·θ_i`, where `m` is the position and `θ_i` is a per-pair frequency.
#
# The magic: the dot product of a query at position `m` and a key at position
# `n` depends only on `(m − n)`. Rotating both by the same amount changes
# nothing; rotating by different amounts encodes exactly their difference.
# **Absolute rotations produce relative attention, for free.**

# %%
def build_rope_cache(head_dim: int, max_seq: int, base: float = 10000.0, device="cpu"):
    """Precompute cos/sin tables. Depends only on position, so compute once."""
    # Frequencies: theta_i = base^(-2i/d). Low i -> fast rotation (local
    # detail), high i -> slow rotation (long-range position). It's a
    # multi-resolution positional clock.
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(positions, inv_freq)          # (max_seq, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of dimensions. x is (B, n_heads, T, head_dim)."""
    T = x.size(2)
    cos, sin = cos[:T], sin[:T]                       # (T, hd/2)
    # Split into even/odd dims: these are the (x, y) of each 2-D pair.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    # Standard 2-D rotation:  x' = x cos - y sin ;  y' = x sin + y cos
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.stack([rot1, rot2], dim=-1).flatten(-2)


# Demonstrate the defining property: attention scores depend on relative offset.
hd, max_seq = 64, 128
cos, sin = build_rope_cache(hd, max_seq)

# To isolate the POSITION effect we must hold CONTENT constant: put the exact
# same vector at every position. (Using randn per position would let content
# differences swamp the positional signal and the demo would prove nothing.)
vec = torch.randn(hd)
q = vec.view(1, 1, 1, hd).expand(1, 1, max_seq, hd).contiguous()
k = q.clone()
qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

print("score between positions holding CONTENT identical:")
print(f"{'pair':<16}{'offset':>8}{'score':>10}")
print("-" * 34)
for (i, j) in [(10, 10), (11, 10), (20, 19), (50, 49), (15, 10), (55, 50), (60, 10)]:
    s = torch.dot(qr[0, 0, i], kr[0, 0, j]).item()
    print(f"{f'({i},{j})':<16}{i-j:>8}{s:>10.4f}")

# %% [markdown]
# Look at the pairs with the same offset — (11,10), (20,19), (50,49) all have
# offset 1 and **the same score**, despite being at completely different
# absolute positions. Same for offset 5: (15,10) and (55,50) match.
#
# That's translation invariance, and it's why RoPE extrapolates: a model that
# learned "offset 5 matters" at positions 10–15 applies the same knowledge at
# positions 5000–5005.

# %%
# Verify it numerically rather than by eyeballing.
def score(i: int, j: int) -> float:
    return torch.dot(qr[0, 0, i], kr[0, 0, j]).item()


for offset in [1, 5, 20]:
    scores = [score(base + offset, base) for base in [5, 20, 40, 60, 80]]
    spread = max(scores) - min(scores)
    print(f"offset {offset:>3}: spread {spread:.2e}  scores {[f'{s:.4f}' for s in scores]}")
print("\nspreads are ~1e-5 (float noise) -> the score is a pure function of (i-j).")
print("Also notice scores DECAY as offset grows: RoPE gives a mild built-in")
print("locality bias, which is a reasonable prior for language.")

# %% [markdown]
# ### Context extension: the `base` parameter
#
# A model trained at 4k context can be stretched to 32k+ by increasing `base`
# (usually called "rope theta scaling"). Larger base = slower rotation = the
# same angular range covers more positions.
#
# This is how Llama 3 went from 8k to 128k context. It isn't free — you need a
# short fine-tune at the new length — but it beats retraining. Variants: NTK-aware
# scaling, YaRN, linear position interpolation.

# %%
print(f"{'base':>10}{'full rotation of slowest dim after':>38}")
print("-" * 48)
for base in [10_000, 100_000, 1_000_000]:
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    slowest = inv[-1].item()
    print(f"{base:>10}{2*math.pi/slowest:>32,.0f} pos")

# %% [markdown]
# ## 2. RMSNorm — LayerNorm minus the mean
#
# LayerNorm does two things: **re-center** (subtract mean) and **re-scale**
# (divide by std). RMSNorm asks: is the centering doing anything?
#
# ```
# LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
# RMSNorm(x)   = x / sqrt(mean(x^2) + eps) * gamma
# ```
#
# It turns out re-centering contributes almost nothing to quality, and it costs
# an extra pass over the data plus the `beta` parameters. Removing it is ~10–15%
# faster with no measurable quality loss — so everyone did.

# %%
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # note: no bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the norm in fp32 even under bf16 autocast: squaring bf16
        # values loses precision badly and this is a reduction over the whole
        # feature dim. This upcast is standard in every real implementation.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


x = torch.randn(4, 128, 512, device=device)
ln = nn.LayerNorm(512).to(device)
rms = RMSNorm(512).to(device)

print(f"LayerNorm params: {sum(p.numel() for p in ln.parameters()):,} (weight + bias)")
print(f"RMSNorm params:   {sum(p.numel() for p in rms.parameters()):,} (weight only)")


def bench(fn, x, n=200):
    for _ in range(20):
        fn(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t) / n * 1e6


if device == "cuda":
    t_ln, t_rms = bench(ln, x), bench(rms, x)
    print(f"\nLayerNorm {t_ln:7.1f} us/call")
    print(f"RMSNorm   {t_rms:7.1f} us/call   ({t_ln/t_rms:.2f}x)")
else:
    print("\n(skipping the timing benchmark on CPU — it is not representative.)")

# %% [markdown]
# **A caveat on that benchmark, so you don't draw the wrong conclusion.**
#
# Our `RMSNorm` is written for clarity, not speed: the explicit `.float()`
# upcast allocates a full-size fp32 temporary. PyTorch's `nn.LayerNorm` is a
# single fused C++/CUDA kernel. So this comparison can easily show RMSNorm
# *losing*, especially on CPU — that's an artifact of unfused Python, not a
# property of the algorithm.
#
# The real-world ~10–15% gain comes from a **fused** RMSNorm kernel (as in
# `apex`, `flash-attn`'s layer_norm, or `torch.compile`'s generated Triton),
# where the fp32 accumulation happens in registers with no extra memory traffic.
#
# The honest, unambiguous win visible here is the **parameter count**: RMSNorm
# has no bias, so half the parameters, and one less reduction pass over the data.
# Measure fused-vs-fused if you want the true speed number.

# %% [markdown]
# ## 3. SwiGLU — a gated MLP
#
# GPT-2's MLP: `down(gelu(up(x)))` — two matrices.
#
# SwiGLU uses **three**: two parallel projections, one of which acts as a
# multiplicative *gate*:
#
# ```
# SwiGLU(x) = down( silu(gate(x)) * up(x) )
# ```
#
# The gate lets the network modulate information multiplicatively per element,
# not just additively. Empirically it's a consistent quality win.
#
# **The 2/3 detail:** three matrices instead of two would be 1.5× the
# parameters, so implementations shrink the hidden dimension to `⅔ × 4d ≈ 2.67d`
# to keep the count matched. If you skip that, you're comparing a bigger model to
# a smaller one and learning nothing.

# %%
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None, multiple_of: int = 256) -> None:
        super().__init__()
        if hidden is None:
            hidden = int(2 * (4 * dim) / 3)
            # Round up to a multiple of 256 — GPU matmul kernels are far more
            # efficient on nicely-aligned dimensions.
            hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.hidden = hidden
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # silu(z) = z * sigmoid(z), a.k.a. swish
        return self.down(F.silu(self.gate(x)) * self.up(x))


d = 768
gelu_mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False))
swiglu = SwiGLU(d)

n_gelu = sum(p.numel() for p in gelu_mlp.parameters())
n_swi = sum(p.numel() for p in swiglu.parameters())
print(f"GELU MLP  hidden={4*d:<6} params {n_gelu:>12,}")
print(f"SwiGLU    hidden={swiglu.hidden:<6} params {n_swi:>12,}")
print(f"ratio {n_swi/n_gelu:.3f}  (should be close to 1.0 — that's the point of the 2/3)")

# %% [markdown]
# ## 4. GQA — Grouped-Query Attention
#
# ### The problem: the KV cache
#
# At inference you cache keys and values for every past token so you don't
# recompute them each step. That cache is:
#
# ```
# 2 (K and V) × n_layers × n_kv_heads × head_dim × seq_len × batch × dtype_bytes
# ```
#
# For a 7B model at 32k context this **exceeds the model weights**. Since
# generation is memory-bandwidth bound, the cache is often the binding
# constraint on how many users you can serve.
#
# ### The fix
#
# Queries need to be diverse — that's what gives you multiple attention
# patterns. But keys and values can be **shared across groups of query heads**.
#
# ```
# MHA:  32 Q heads, 32 KV heads   <- GPT-2, biggest cache
# GQA:  32 Q heads,  8 KV heads   <- Llama 2/3, Qwen: 4x smaller cache
# MQA:  32 Q heads,  1 KV head    <- smallest cache, some quality loss
# ```
#
# GQA at ratio 4–8 is nearly free in quality and hugely cheaper in memory,
# which is why it's now universal.

# %%
def kv_cache_gb(n_layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per=2):
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per / 1024**3


print("KV cache for a 7B-class model (32 layers, head_dim 128), bf16, batch 1:\n")
print(f"{'context':>9}{'MHA (32 kv)':>14}{'GQA (8 kv)':>13}{'MQA (1 kv)':>13}")
print("-" * 49)
for seq in [2048, 8192, 32768, 131072]:
    print(
        f"{seq:>9}"
        f"{kv_cache_gb(32, 32, 128, seq):>13.2f}G"
        f"{kv_cache_gb(32, 8, 128, seq):>12.2f}G"
        f"{kv_cache_gb(32, 1, 128, seq):>12.2f}G"
    )
print("\n(a 7B model's bf16 weights are ~14 GB, for comparison)")

# %% [markdown]
# At 128k context, MHA needs **64 GB** of cache for a single sequence — more
# than four times the model's own weights, and impossible on any single consumer
# GPU. GQA cuts it to 16 GB, MQA to 2 GB. This table is the entire reason GQA
# exists, and it's why serving throughput is usually a memory problem rather
# than a compute problem.

# %%
class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_seq: int = 4096) -> None:
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads      # query heads per kv head
        self.head_dim = dim // n_heads

        # Note the asymmetry: q projects to full size, k/v project to the
        # smaller kv size. That's where the savings come from.
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.rope_base = 10000.0
        c, s = build_rope_cache(self.head_dim, max_seq, self.rope_base)
        self.register_buffer("rope_cos", c, persistent=False)
        self.register_buffer("rope_sin", s, persistent=False)

    def _ensure_rope(self, T: int, device) -> None:
        """Grow the cos/sin tables on demand.

        Without this, RoPE would inherit the same hard length limit as learned
        position embeddings — not because of the math, but because we only
        precomputed a fixed-size table. HF calls this 'dynamic' RoPE.
        """
        if T > self.rope_cos.size(0):
            c, s = build_rope_cache(self.head_dim, T, self.rope_base, device=device)
            self.rope_cos, self.rope_sin = c, s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        self._ensure_rope(T, x.device)
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE applies to q and k only — never to v. v carries content, not
        # position; rotating it would corrupt the information being retrieved.
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Expand kv heads to match q heads. repeat_interleave materialises the
        # tensor; PyTorch's SDPA can also do this via enable_gqa=True on newer
        # versions, which avoids the copy.
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, C))


for n_kv in [12, 4, 2, 1]:
    a = GroupedQueryAttention(768, 12, n_kv)
    n = sum(p.numel() for p in a.parameters())
    out = a(torch.randn(2, 32, 768))
    print(f"n_kv_heads={n_kv:>3}  params {n:>10,}  cache ratio {n_kv/12:.2f}x  out {tuple(out.shape)}")

# %% [markdown]
# ## Assemble: a Llama-style model

# %%
@dataclass
class LlamaConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    norm_eps: float = 1e-6
    tie_weights: bool = True


class LlamaBlock(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg.n_embd, cfg.n_head, cfg.n_kv_head, cfg.block_size)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # No position embedding table — RoPE is applied inside attention.
        self.blocks = nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


llama = LlamaModel(LlamaConfig())
logits, loss = llama(
    torch.randint(0, 50257, (2, 64)), torch.randint(0, 50257, (2, 64))
)
print(f"params: {sum(p.numel() for p in llama.parameters()):,}")
print(f"logits: {tuple(logits.shape)}   loss: {loss.item():.4f}")
print(f"expected loss at init: {math.log(50257):.4f}")

# %% [markdown]
# ## RoPE's real payoff: length extrapolation
#
# Learned position embeddings **cannot run at all** past `block_size` — it's an
# index error. RoPE just keeps rotating.
#
# Quality still degrades well past the training length (the model never learned
# to use those distances), which is why context extension needs fine-tuning. But
# "degrades" beats "crashes".

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))   # must come BEFORE the import
from llmfs.model import GPT, GPTConfig  # noqa: E402

gpt2_style = GPT(GPTConfig(vocab_size=1000, block_size=128, n_layer=2, n_head=4, n_embd=128))
llama_style = LlamaModel(LlamaConfig(vocab_size=1000, block_size=128, n_layer=2, n_head=4,
                                     n_kv_head=2, n_embd=128))

print(f"{'seq len':>9}  {'GPT-2 (learned pos)':<32}{'Llama (RoPE)':<20}")
print("-" * 64)
for T in [64, 128, 200, 512]:
    idx = torch.randint(0, 1000, (1, T))
    try:
        gpt2_style(idx)
        a = "ok"
    except Exception as e:
        a = f"{type(e).__name__}: {str(e)[:24]}"
    try:
        llama_style(idx)
        b = "ok"
    except Exception as e:
        b = f"{type(e).__name__}: {str(e)[:24]}"
    print(f"{T:>9}  {a:<32}{b:<20}")

print("\nRoPE runs at ANY length; learned positions hard-stop at block_size.")
print("Note this required `_ensure_rope` to grow the cos/sin tables — a fixed")
print("precomputed table would have failed too, for a mundane reason unrelated")
print("to the math. Worth knowing when you read other implementations.")

# %% [markdown]
# **Running is not the same as working well.** RoPE will happily process 4×
# its training length, but quality degrades — the model never learned what
# those larger relative offsets mean, and attention scores drift out of the
# range it was calibrated for.
#
# That gap is exactly what context-extension methods address: increase
# `rope_base` (or apply YaRN / position interpolation) *and* fine-tune briefly
# at the target length. The graceful-degradation property is what makes that
# cheap fine-tune sufficient, instead of needing a full retrain.

# %% [markdown]
# ## What we skipped
#
# Worth knowing the names:
#
# - **MoE (Mixture of Experts)** — replace the MLP with N experts, route each
#   token to the top-k. Far more parameters at the same FLOPs per token. Used by
#   Mixtral, DeepSeek-V3, Qwen3-MoE. Hard to train, big win at scale.
# - **MLA (Multi-head Latent Attention)** — DeepSeek's alternative to GQA;
#   compresses KV into a low-rank latent. Smaller cache than GQA at better
#   quality.
# - **Sliding-window / local attention** — each token attends to the last W
#   tokens only. O(T·W) instead of O(T²). Mistral interleaves it with full layers.
# - **QK-Norm** — normalize q and k before the dot product. Prevents attention
#   logit blowup at scale; increasingly standard.
# - **Muon optimizer** — orthogonalized momentum updates on 2-D parameters.
#   Currently holds the nanoGPT speedrun records. Notebook 06.
#
# ## Exercises
#
# 1. **Ablate.** Train four 10M models on TinyStories (notebook 04's loop):
#    baseline GPT-2, +RoPE, +RMSNorm, +SwiGLU. Which helps most per unit of time?
# 2. **RoPE base sweep.** Train at block_size 256, then evaluate perplexity at
#    512 with base ∈ {10k, 100k, 500k}. Plot the degradation curve.
# 3. **GQA quality cost.** Fix total parameters and vary `n_kv_head` ∈
#    {12, 4, 2, 1}. Measure val loss and KV cache size. Where's the knee?
#
# ## Checkpoint
#
# - [ ] You can explain why RoPE gives relative positions from absolute rotations
# - [ ] You know why RMSNorm computes in fp32
# - [ ] You know why SwiGLU shrinks its hidden dim by 2/3
# - [ ] You can compute a KV cache size from a config
#
# **Next:** `06_scaling_and_efficiency.ipynb` — make it fast, and know how big to
# go.

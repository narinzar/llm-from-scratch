# %% [markdown]
# # 03 — The Transformer, Built from Scratch
#
# **Goal:** implement a GPT from raw tensor operations — attention, causal
# masking, multi-head, the block, the full model — and understand *why* each
# piece is shaped the way it is.
#
# **Time:** 60–90 min. **Hardware:** CPU is fine; GPU is faster.
#
# ## How to use this notebook
#
# Every component is built twice: first as explicit loops and small tensors you
# can print, then as the batched implementation you'd actually use. Run the
# printouts. Look at the shapes. **Shape confusion is the number-one source of
# transformer bugs**, so we assert shapes everywhere.

# %%
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} on {device}")

# %% [markdown]
# ## Part 1 — Attention, derived
#
# ### The problem
#
# Given a sequence of token vectors, each position needs to gather information
# from other positions. "The animal didn't cross the street because **it** was
# too tired" — to represent `it`, position 9 must look back at `animal`.
#
# ### The mechanism
#
# Every token emits three vectors:
#
# | vector | role | analogy |
# |---|---|---|
# | **query** `q` | what I'm looking for | a search box |
# | **key** `k` | what I offer | a document's index terms |
# | **value** `v` | what I'll actually give you | the document's content |
#
# Relevance of position `j` to position `i` is `q_i · k_j`. Softmax over `j`
# turns those scores into weights, and the output is the weighted sum of values.
#
# Start with a single position, no batching, explicit loops.

# %%
T, d = 4, 8  # 4 tokens, 8 dims

x = torch.randn(T, d)
W_q, W_k, W_v = (torch.randn(d, d) * 0.1 for _ in range(3))

q, k, v = x @ W_q, x @ W_k, x @ W_v
print(f"x {tuple(x.shape)} -> q,k,v each {tuple(q.shape)}")

# Attention for position 2, computed the slow, obvious way.
i = 2
scores = torch.tensor([torch.dot(q[i], k[j]) for j in range(T)])
print(f"\nraw scores of token {i} against all tokens: {scores.numpy().round(3)}")

scaled = scores / math.sqrt(d)
weights = F.softmax(scaled, dim=-1)
print(f"after /sqrt(d) and softmax:               {weights.numpy().round(3)}")
print(f"weights sum to {weights.sum():.4f}")

out_i = sum(weights[j] * v[j] for j in range(T))
print(f"\noutput for token {i}: {out_i.numpy().round(3)}")

# %% [markdown]
# ### Why divide by √d?
#
# Not cosmetic. If `q` and `k` have unit-variance independent components, then
# `q · k` is a sum of `d` such products, so its **variance grows as d**. For
# d=768 the scores land in the ±30 range, softmax saturates into a one-hot
# spike, and the gradient through it goes to ~zero. Dividing by √d restores
# unit variance and keeps softmax in its useful regime.
#
# Let's confirm empirically rather than take it on faith.

# %%
print(f"{'d':>6}{'var(q·k)':>12}{'max softmax':>14}{'entropy':>10}")
print("-" * 42)
for dim in [8, 64, 512, 4096]:
    qq, kk = torch.randn(2000, dim), torch.randn(2000, dim)
    raw = (qq * kk).sum(-1)
    w_unscaled = F.softmax(raw[:16], dim=-1)
    ent = -(w_unscaled * w_unscaled.clamp_min(1e-9).log()).sum()
    print(f"{dim:>6}{raw.var().item():>12.1f}{w_unscaled.max().item():>14.4f}{ent.item():>10.3f}")

print("\nwith the 1/sqrt(d) scaling applied:")
print(f"{'d':>6}{'var(q·k)':>12}{'max softmax':>14}{'entropy':>10}")
print("-" * 42)
for dim in [8, 64, 512, 4096]:
    qq, kk = torch.randn(2000, dim), torch.randn(2000, dim)
    raw = (qq * kk).sum(-1) / math.sqrt(dim)
    w_scaled = F.softmax(raw[:16], dim=-1)
    ent = -(w_scaled * w_scaled.clamp_min(1e-9).log()).sum()
    print(f"{dim:>6}{raw.var().item():>12.1f}{w_scaled.max().item():>14.4f}{ent.item():>10.3f}")

# %% [markdown]
# Unscaled at d=4096: one weight is ~1.0 and entropy collapses to ~0 — the
# softmax has become an argmax and stopped passing gradient. Scaled: variance
# stays ~1 and entropy stays healthy at any `d`. This is the whole reason for
# the √d.

# %% [markdown]
# ### Causal masking
#
# A language model predicts the next token. If position 3 could attend to
# position 4, it would simply read the answer — training loss would collapse to
# zero and the model would generate garbage at inference, when the future
# doesn't exist yet. This bug is easy to write and produces a *suspiciously
# good* loss curve, which is the tell.
#
# Fix: set scores for `j > i` to `-inf` **before** the softmax, so those
# positions receive exactly zero weight.

# %%
scores_full = (q @ k.T) / math.sqrt(d)
mask = torch.tril(torch.ones(T, T))  # lower triangular incl. diagonal
scores_masked = scores_full.masked_fill(mask == 0, float("-inf"))

print("mask (1 = may attend):")
print(mask.int().numpy())
print("\nmasked scores (-inf shown as -inf):")
print(scores_masked.numpy().round(2))
print("\nattention weights after softmax:")
w = F.softmax(scores_masked, dim=-1)
print(w.numpy().round(3))
print("\nrow sums (must all be 1.0):", w.sum(-1).numpy().round(4))

assert torch.allclose(w.sum(-1), torch.ones(T)), "rows must be a probability dist"
assert (w.triu(diagonal=1) == 0).all(), "no weight may land on the future"
print("\ncausality verified: upper triangle is exactly zero")

# %% [markdown]
# Note row 0: token 0 can only see itself, so its weight is exactly 1.0 on
# itself — it has no context at all. That's expected and correct.

# %% [markdown]
# ## Part 2 — Multi-head attention
#
# One attention operation produces one weighted average — a single "kind" of
# relationship. Real language needs many at once: syntactic agreement,
# coreference, topical association.
#
# **Multi-head** splits `d_model` into `n_heads` chunks of size `head_dim` and
# runs attention independently in each, then concatenates. Crucially this costs
# *the same* as one big head — you're partitioning dimensions, not adding them.

# %%
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Shape convention used throughout:
        B = batch, T = time/sequence, C = channels (d_model),
        nh = n_heads, hd = head_dim (C // nh)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 use_flash: bool = True) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly into heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_flash = use_flash
        self.dropout = dropout

        # One fused projection producing q, k, v together. Three separate
        # Linears would be mathematically identical but launch 3 GEMMs
        # instead of 1 — measurably slower.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)                                  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)                       # each (B, T, C)

        # (B, T, C) -> (B, nh, T, hd). The transpose puts heads on the batch
        # side so the matmul below treats each head as an independent problem.
        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        if self.use_flash:
            # FlashAttention: same math, but never materialises the (B,nh,T,T)
            # score matrix in HBM. Memory goes O(T^2) -> O(T), and it's faster
            # because it's memory-bandwidth bound, not compute bound.
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~causal, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = F.dropout(att, p=self.dropout, training=self.training)
            y = att @ v                                     # (B, nh, T, hd)

        # Back to (B, T, C). contiguous() is required because transpose only
        # changes strides; view() needs a contiguous buffer.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


# Verify the two paths agree — this is the test that catches masking bugs.
attn_flash = CausalSelfAttention(64, 4, use_flash=True).eval()
attn_slow = CausalSelfAttention(64, 4, use_flash=False).eval()
attn_slow.load_state_dict(attn_flash.state_dict())

xb = torch.randn(2, 16, 64)
with torch.no_grad():
    a, b = attn_flash(xb), attn_slow(xb)

print(f"input  {tuple(xb.shape)}")
print(f"output {tuple(a.shape)}")
print(f"flash vs manual max abs diff: {(a - b).abs().max().item():.2e}")
assert torch.allclose(a, b, atol=1e-5), "flash and manual paths must agree"
print("the two implementations agree")

# %% [markdown]
# ### Prove causality end-to-end
#
# Shape checks don't catch a subtle masking bug. This does: perturb a **future**
# token and confirm earlier outputs are bit-identical. If they change, the model
# is leaking the future.

# %%
model_a = CausalSelfAttention(32, 4).eval()
x1 = torch.randn(1, 8, 32)
x2 = x1.clone()
x2[0, 5:] = torch.randn(3, 32)  # change only positions 5,6,7

with torch.no_grad():
    y1, y2 = model_a(x1), model_a(x2)

print(f"positions 0-4 max diff: {(y1[0, :5] - y2[0, :5]).abs().max().item():.3e}  (must be ~0)")
print(f"positions 5-7 max diff: {(y1[0, 5:] - y2[0, 5:]).abs().max().item():.3e}  (must be > 0)")
assert (y1[0, :5] - y2[0, :5]).abs().max() < 1e-6, "FUTURE IS LEAKING"
print("causality holds")

# %% [markdown]
# ## Part 3 — The MLP (where most parameters live)
#
# Attention *moves* information between positions. The MLP *processes* it,
# independently per position. Expand 4×, apply a nonlinearity, project back.
#
# Why 4×? Empirical, from the original paper, and it stuck. It's also why the
# MLP holds ~2/3 of a transformer's parameters:
# `4·d² (up) + 4·d² (down) = 8d²` vs attention's `4d²`.

# %%
class MLP(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = expansion * d_model
        self.fc = nn.Linear(d_model, hidden, bias=False)
        self.proj = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU, not ReLU. It's smooth, so it passes gradient for slightly
        # negative inputs instead of hard-zeroing them. 'tanh' is the fast
        # approximation GPT-2 used; the exact version is marginally slower.
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))


mlp = MLP(64)
n_attn = sum(p.numel() for p in CausalSelfAttention(64, 4).parameters())
n_mlp = sum(p.numel() for p in mlp.parameters())
print(f"attention params: {n_attn:,}  (4 * d^2 = {4*64*64:,})")
print(f"MLP params:       {n_mlp:,}  (8 * d^2 = {8*64*64:,})")
print(f"MLP is {n_mlp/n_attn:.1f}x the attention block")

# %% [markdown]
# ## Part 4 — Residuals and pre-norm: why deep networks train at all
#
# Two design choices do the heavy lifting for trainability.
#
# **Residual connections** (`x + f(x)`) give gradients a path that skips every
# layer. Without them, the gradient must survive multiplication through every
# layer's Jacobian, and it vanishes.
#
# **Pre-norm vs post-norm** — this is *the* change that made deep transformers
# trainable without warmup gymnastics:
#
# ```
# post-norm (2017):  x = LayerNorm(x + Attn(x))      <- norm ON the residual path
# pre-norm  (GPT-2+): x = x + Attn(LayerNorm(x))     <- residual path is CLEAN
# ```
#
# In pre-norm the residual stream is never normalized, so there's an unbroken
# identity path from input to output. Everything modern uses pre-norm.

# %%
def gradient_survival(n_layers: int, mode: str, d: int = 64) -> float:
    """Ratio of grad norm at layer 0 vs the last layer. Closer to 1 = healthier."""
    torch.manual_seed(0)
    layers = nn.ModuleList(
        [nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)) for _ in range(n_layers)]
    )
    norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])

    h = torch.randn(8, d, requires_grad=True)
    for lyr, nrm in zip(layers, norms):
        if mode == "none":
            h = lyr(h)
        elif mode == "post":
            h = nrm(h + lyr(h))
        elif mode == "pre":
            h = h + lyr(nrm(h))

    # Use a random projection, NOT h.sum(). LayerNorm forces zero mean, so the
    # sum of a post-norm output is constant and its gradient is exactly zero —
    # you'd measure 0.00 and wrongly conclude post-norm kills gradients.
    # A random-projection loss is a generic, unbiased probe.
    torch.manual_seed(1)
    (h * torch.randn_like(h)).sum().backward()

    g_first = layers[0][0].weight.grad.norm().item()
    g_last = layers[-1][0].weight.grad.norm().item()
    return g_first / max(g_last, 1e-12)


print(f"{'depth':>7}{'no residual':>14}{'post-norm':>12}{'pre-norm':>11}")
print("-" * 44)
for depth in [2, 6, 12, 24]:
    print(
        f"{depth:>7}"
        f"{gradient_survival(depth, 'none'):>14.2e}"
        f"{gradient_survival(depth, 'post'):>12.2e}"
        f"{gradient_survival(depth, 'pre'):>11.2e}"
    )
print("\n(ratio of first-layer to last-layer gradient norm; closer to 1 is healthier)")

# %% [markdown]
# **Read this result honestly.** It shows one thing dramatically and one thing
# barely:
#
# - **Residual vs no residual is enormous.** Without residuals the first layer's
#   gradient is ~10⁻¹⁷ of the last layer's at depth 24 — those layers receive
#   effectively no signal and never learn. This is the vanishing-gradient
#   problem, and it is why residuals exist.
# - **Pre-norm vs post-norm looks similar here, and that's expected.** A static
#   probe at initialization can't show the real difference. Post-norm's problem
#   is *dynamic*: during training, gradients through the norm-on-the-residual
#   path make early updates unstable, so post-norm models need careful LR warmup
#   and often diverge without it. Pre-norm trains stably at high LR from step 0.
#   You'd need a full training run to see it — which is exactly why the field
#   took a couple of years to settle on pre-norm.
#
# Don't over-claim from a measurement. Being able to say "this experiment shows
# X but not Y" is a research skill worth more than the result itself.

# %% [markdown]
# ## Part 5 — Positional embeddings
#
# Attention is **permutation-equivariant**: shuffle the input tokens and the
# outputs shuffle identically. Nothing in the math knows about order. "dog bites
# man" and "man bites dog" would be indistinguishable.
#
# GPT-2's fix is the simplest possible one: a learned embedding per position,
# added to the token embedding. (Notebook 05 replaces this with RoPE, which is
# strictly better — but learned positions are what GPT-2 did and they're easy to
# reason about.)

# %%
# Prove permutation-equivariance. We must drop the causal mask to isolate the
# property (the mask itself injects order information), so compute plain
# bidirectional attention directly.
def bare_attention(x: torch.Tensor, Wq, Wk, Wv) -> torch.Tensor:
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    att = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(x.size(-1)), dim=-1)
    return att @ v


torch.manual_seed(0)
d_demo = 16
Wq, Wk, Wv = (torch.randn(d_demo, d_demo) * 0.1 for _ in range(3))
seq = torch.randn(1, 5, d_demo)
perm = torch.tensor([4, 3, 2, 1, 0])

out_orig = bare_attention(seq, Wq, Wk, Wv)
out_perm = bare_attention(seq[:, perm], Wq, Wk, Wv)

# If attention were order-aware, permuting the input would change the outputs
# in some complicated way. Instead the outputs simply permute along with it.
print(f"attention(shuffled x)  vs  shuffle(attention(x))")
print(f"max abs difference: {(out_perm - out_orig[:, perm]).abs().max().item():.2e}")
assert torch.allclose(out_perm, out_orig[:, perm], atol=1e-5)
print("\nIDENTICAL -> attention is permutation-equivariant.")
print("It literally cannot distinguish 'dog bites man' from 'man bites dog'")
print("on content alone. Position information must be added explicitly.")

# Adding position embeddings breaks the symmetry, which is the entire point.
pos_emb = torch.randn(1, 5, d_demo) * 0.5
out_orig_p = bare_attention(seq + pos_emb, Wq, Wk, Wv)
out_perm_p = bare_attention(seq[:, perm] + pos_emb, Wq, Wk, Wv)
print(f"\nwith position embeddings added, the same comparison differs by "
      f"{(out_perm_p - out_orig_p[:, perm]).abs().max().item():.3f}  (symmetry broken)")

# %% [markdown]
# ## Part 6 — Assemble the GPT
#
# Now the whole model. Read the `forward` as a pipeline:
#
# ```
# token ids -> token emb + position emb
#           -> N x [ x + Attn(LN(x)) ; x + MLP(LN(x)) ]
#           -> final LayerNorm
#           -> linear to vocab -> logits
# ```

# %%
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024      # max context length
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    tie_weights: bool = True    # share token embedding with the output head


class Block(nn.Module):
    """One transformer block, pre-norm."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg.n_embd, cfg.n_head, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg.n_embd, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # communicate across positions
        x = x + self.mlp(self.ln2(x))    # think, per position
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)   # position embeddings
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            # Weight tying: the output head IS the token embedding matrix.
            # Saves vocab*d params (~38M of GPT-2 small's 124M!) and usually
            # improves quality — "which token is this vector" and "what vector
            # is this token" are two views of one relationship.
            self.head.weight = self.wte.weight

        self.apply(self._init_weights)
        # Scaled init for residual projections: with N blocks each adding to
        # the residual stream, variance grows like N. Scaling these by
        # 1/sqrt(2N) keeps the stream's variance stable at init. (GPT-2 paper.)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} exceeds block_size {self.cfg.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))    # (B,T,C); pos broadcasts over B

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                            # (B, T, vocab)

        loss = None
        if targets is not None:
            # Flatten to (B*T, vocab) vs (B*T,). cross_entropy expects raw
            # logits — it fuses log_softmax + NLL for numerical stability.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n


# %% [markdown]
# ### Instantiate and inspect
#
# Reproduce GPT-2 small's shape and check the parameter count lands at 124M.

# %%
cfg = GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768)
model = GPT(cfg)

total = sum(p.numel() for p in model.parameters())
print(f"total parameters:          {total:,}")
print(f"non-embedding parameters:  {model.num_params():,}")
print(f"(GPT-2 small is quoted as 124M -- we should be very close)\n")

print(f"{'component':<28}{'params':>14}{'% of total':>12}")
print("-" * 54)
groups = {
    "token embedding (wte)": model.wte.weight.numel(),
    "position embedding (wpe)": model.wpe.weight.numel(),
    "transformer blocks": sum(p.numel() for p in model.blocks.parameters()),
    "final layernorm": sum(p.numel() for p in model.ln_f.parameters()),
}
if not cfg.tie_weights:
    groups["output head"] = model.head.weight.numel()
for name, n in groups.items():
    print(f"{name:<28}{n:>14,}{100*n/total:>11.1f}%")

per_block = sum(p.numel() for p in model.blocks[0].parameters())
print(f"\nper block: {per_block:,}  x {cfg.n_layer} layers")

# %% [markdown]
# ### Forward pass and the loss you should expect at init

# %%
B, T = 4, 128
idx = torch.randint(0, cfg.vocab_size, (B, T))
targets = torch.randint(0, cfg.vocab_size, (B, T))

logits, loss = model(idx, targets)
print(f"input   {tuple(idx.shape)}")
print(f"logits  {tuple(logits.shape)}   (B, T, vocab)")
print(f"loss    {loss.item():.4f}")

expected = math.log(cfg.vocab_size)
print(f"\nexpected loss at init: ln({cfg.vocab_size}) = {expected:.4f}")
print(f"difference: {abs(loss.item() - expected):.4f}")

# %% [markdown]
# **This check is worth internalizing.** An untrained model should be uniformly
# uncertain across the vocabulary, so cross-entropy = `ln(vocab_size)` ≈ 10.82
# for 50257 tokens.
#
# - **Loss much higher** (e.g. 15+) → your initialization is broken; some logits
#   are large and confidently wrong.
# - **Loss much lower** → you have a bug. Usually label leakage (broken causal
#   mask) or your targets aren't actually shifted.
#
# Always print this before starting a long training run.

# %% [markdown]
# ## Part 7 — Generation
#
# Generation is a loop: forward the context, take the **last** position's
# logits, sample, append, repeat. Note the model computes logits for every
# position but we only need the last one — that redundancy is what KV caching
# eliminates (notebook 15).

# %%
@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    model.eval()
    for _ in range(max_new_tokens):
        # Crop to block_size — the model has no position embeddings beyond it.
        idx_cond = idx[:, -model.cfg.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)   # last position only

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
start = torch.tensor([tok.encode("The meaning of life is")], dtype=torch.long)
out = generate(model, start, max_new_tokens=20, top_k=50)
print("untrained model output:")
print(f"  {tok.decode(out[0].tolist())!r}")
print("\n(pure noise, as expected -- these weights are random. Notebook 04 fixes that.)")

# %% [markdown]
# ## Part 8 — Sanity check: can it memorize?
#
# Before a real training run, verify the model can **overfit a single batch to
# ~zero loss**. If it can't, something is fundamentally broken — the loss, the
# optimizer, the gradient flow — and no amount of data will save you.
#
# This 30-second test has saved more GPU-hours than any other debugging trick.

# %%
small_cfg = GPTConfig(vocab_size=1000, block_size=64, n_layer=2, n_head=4, n_embd=128)
tiny = GPT(small_cfg).to(device)
# lr 1e-3 is deliberately aggressive: we WANT to overfit as fast as possible.
opt = torch.optim.AdamW(tiny.parameters(), lr=1e-3)

xb = torch.randint(0, 1000, (4, 32), device=device)
yb = torch.randint(0, 1000, (4, 32), device=device)

print(f"target: drive loss from ~{math.log(1000):.2f} toward 0\n")
for step in range(401):
    _, l = tiny(xb, yb)
    opt.zero_grad(set_to_none=True)
    l.backward()
    opt.step()
    if step % 50 == 0:
        print(f"  step {step:>4}  loss {l.item():.4f}")

print(f"\nfinal loss {l.item():.4f}")
if l.item() < 0.05:
    print("PASS — the model can learn. Architecture and training loop are wired correctly.")
else:
    print("SLOW — loss is falling but hasn't collapsed yet. Run more steps.")
    print("Only worry if the loss is FLAT or increasing; that means a real bug.")

# %% [markdown]
# ### Record this run
#
# The overfit test is the architecture's smoke alarm: a correct transformer
# drives a single batch to near-zero loss, a broken one plateaus. Recording it
# means that when you untie the weights in exercise 2 — or change the init, or
# the attention mask — you can see at a glance whether the model still learns,
# and what the change cost in parameters.

# %%
import sys

sys.path.insert(0, "..")          # repo root, so `llmfs` is importable
from llmfs.bench import log_run

log_run(
    stage="03_transformer",
    metrics={
        "overfit_loss": l.item(),
        "n_params": sum(p.numel() for p in tiny.parameters()),
    },
    key="overfit_loss",
    config={"n_layer": small_cfg.n_layer, "n_embd": small_cfg.n_embd,
            "n_head": small_cfg.n_head, "steps": 400},
    notes="single-batch overfit test",
)

# %% [markdown]
# The shape of that curve matters more than the final number:
#
# | what you see | what it means |
# |---|---|
# | falls smoothly to ~0 | everything is wired correctly |
# | flat at ln(vocab) | gradients aren't reaching the weights — check `requires_grad`, check you called `opt.step()` |
# | starts far below ln(vocab) | label leakage — your targets aren't shifted, or the causal mask is broken |
# | NaN | learning rate too high, or a `-inf` leaked out of the mask into the softmax |

# %% [markdown]
# ## Part 9 — Where does the compute go?
#
# A useful mental model. Per token, forward+backward FLOPs ≈ `6 × N` where N is
# the parameter count (2 for forward matmul, 4 for backward). Attention adds a
# term that scales with `T²`, which is why long context is expensive.

# %%
def flops_per_token(cfg: GPTConfig) -> dict:
    """Approximate FLOPs per token for a forward+backward pass."""
    n = 12 * cfg.n_layer * cfg.n_embd**2          # params in the blocks
    dense = 6 * n
    attn = 6 * 2 * cfg.n_layer * cfg.block_size * cfg.n_embd  # the T-dependent part
    return {"dense": dense, "attention": attn, "total": dense + attn,
            "attn_share": attn / (dense + attn)}


print(f"{'context':>9}{'dense GFLOP':>14}{'attn GFLOP':>13}{'attn %':>9}")
print("-" * 45)
for bs in [512, 1024, 2048, 8192, 32768]:
    c = GPTConfig(block_size=bs)
    f = flops_per_token(c)
    print(f"{bs:>9}{f['dense']/1e9:>14.2f}{f['attention']/1e9:>13.2f}{100*f['attn_share']:>8.1f}%")

# %% [markdown]
# At 1k context attention is a rounding error; at 32k it dominates. That single
# fact drives most of modern LLM efficiency research — FlashAttention, sliding
# windows, GQA, linear attention, state-space models.

# %% [markdown]
# ## Exercises
#
# 1. **Break causality on purpose.** Set `is_causal=False` and train the tiny
#    model on real text. Watch the loss go implausibly low. Then generate — the
#    output will be garbage. Now you'll recognize this bug in the wild.
# 2. **Remove weight tying** (`tie_weights=False`). Compare parameter count and
#    validation loss after notebook 04's run.
# 3. **Attention maps.** Modify `CausalSelfAttention` to return `att`, run a
#    trained model on a sentence, and plot the per-head matrices with
#    `plt.imshow`. Look for the induction heads.
# 4. **Head-count ablation.** At fixed `d_model=768`, try `n_head` in
#    {1, 4, 12, 48}. What breaks at the extremes and why?
#
# ## Checkpoint
#
# - [ ] You can explain q/k/v and the √d scaling from memory
# - [ ] The causality test passed
# - [ ] Init loss ≈ ln(vocab_size)
# - [ ] The overfit-one-batch test reached ~0
#
# **Next:** `04_pretrain_your_first_llm.ipynb` — train this on real data.

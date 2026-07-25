# %% [markdown]
# # 15 — Inference, Quantization, and the Capstone
#
# **Goal:** implement KV caching from scratch, understand quantization, serve
# your model — then put the whole pipeline together.
#
# **Time:** 60 min + the capstone.
#
# ## Why inference deserves its own notebook
#
# Training happens once; inference happens forever. And generation has a
# fundamentally different bottleneck than training:
#
# | phase | bottleneck | why |
# |---|---|---|
# | **prefill** (process the prompt) | compute | all tokens in parallel, big matmuls |
# | **decode** (generate token by token) | **memory bandwidth** | one token at a time; you re-read every weight for a single token of work |
#
# Decode is bandwidth-bound. That single fact explains quantization (fewer bytes
# to read), batching (amortize the read over many sequences), and speculative
# decoding (verify several tokens per weight-read).

# %%
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print(f"device: {device}")

# %% [markdown]
# ## Part 1 — KV caching
#
# ### The waste
#
# Naive generation re-runs the whole prefix every step:
#
# ```
# step 1: forward("The cat")            -> "sat"
# step 2: forward("The cat sat")        -> "on"     <- recomputed "The cat"
# step 3: forward("The cat sat on")     -> "the"    <- recomputed again
# ```
#
# But keys and values for past tokens **never change** — they depend only on
# tokens already fixed. Cache them, and each step processes exactly one new
# token.
#
# Complexity for generating n tokens goes from **O(n³)** to **O(n²)**.

# %%
class CachedAttention(nn.Module):
    """Causal self-attention with an incremental KV cache."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cache: dict | None = None,
                use_cache: bool = False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        def heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)

        if cache is not None and cache.get("k") is not None:
            # Append the new token's k/v to what we already have.
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)

        new_cache = {"k": k, "v": v} if use_cache else None

        # Subtle but critical: when decoding with a cache, T == 1 and the single
        # query legitimately attends to ALL cached keys. is_causal=True would
        # build a 1x1 causal mask and let it see only itself — a silent
        # correctness bug that produces plausible-looking garbage.
        is_causal = q.size(2) > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), new_cache


# Prove the cached path matches the uncached path exactly.
attn = CachedAttention(64, 4).eval()
seq = torch.randn(1, 6, 64)

with torch.no_grad():
    full_out, _ = attn(seq)                     # process all 6 at once

    # Now the same thing incrementally, one token at a time.
    cache, incremental = None, []
    for t in range(6):
        out_t, cache = attn(seq[:, t : t + 1], cache=cache, use_cache=True)
        incremental.append(out_t)
    inc_out = torch.cat(incremental, dim=1)

print(f"full pass    {tuple(full_out.shape)}")
print(f"incremental  {tuple(inc_out.shape)}")
print(f"max abs diff: {(full_out - inc_out).abs().max().item():.2e}")
assert torch.allclose(full_out, inc_out, atol=1e-5), "cache must be mathematically identical"
print("\nKV cache verified: identical output, far less compute.")

# %%
# Quantify the saving in FLOPs terms.
def generation_cost(n_tokens: int, cached: bool) -> int:
    """Relative units of attention work to generate n tokens."""
    if cached:
        return sum(t for t in range(1, n_tokens + 1))        # O(n^2)
    return sum(t * t for t in range(1, n_tokens + 1))        # O(n^3)


print(f"{'tokens':>8}{'no cache':>14}{'with cache':>13}{'speedup':>10}")
print("-" * 45)
for n in [10, 100, 500, 2000]:
    a, b = generation_cost(n, False), generation_cost(n, True)
    print(f"{n:>8}{a:>14,}{b:>13,}{a/b:>9.0f}x")

# %% [markdown]
# ## Part 2 — Sampling strategies
#
# The decoding strategy changes output quality as much as a lot of fine-tuning
# does, and it's free.

# %%
def apply_sampling(logits: torch.Tensor, temperature: float = 1.0,
                   top_k: int | None = None, top_p: float | None = None,
                   repetition_penalty: float = 1.0,
                   prev_tokens: torch.Tensor | None = None,
                   min_p: float | None = None) -> torch.Tensor:
    """Return probabilities after applying the usual decoding controls."""
    logits = logits.clone()

    if repetition_penalty != 1.0 and prev_tokens is not None:
        for tid in set(prev_tokens.tolist()):
            # Divide positive logits, multiply negative ones — so the penalty
            # always pushes toward less likely, regardless of sign.
            if logits[tid] > 0:
                logits[tid] /= repetition_penalty
            else:
                logits[tid] *= repetition_penalty

    if temperature != 1.0:
        logits = logits / max(temperature, 1e-8)

    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Keep tokens up to and including the one that crosses p.
        remove = cum - F.softmax(sorted_logits, dim=-1) > top_p
        logits[sorted_idx[remove]] = float("-inf")

    if min_p is not None:
        probs = F.softmax(logits, dim=-1)
        logits = logits.masked_fill(probs < min_p * probs.max(), float("-inf"))

    return F.softmax(logits, dim=-1)


# Show how each strategy reshapes a realistic, long-tailed distribution.
torch.manual_seed(0)
vocab = 1000
raw = torch.randn(vocab) * 2
raw[:5] += torch.tensor([6.0, 5.0, 4.5, 4.0, 3.5])   # a few clear favourites

print(f"{'strategy':<34}{'eff. choices':>14}{'top prob':>11}")
print("-" * 59)
for name, kw in [
    ("greedy (temp=0.01)",       dict(temperature=0.01)),
    ("temp=0.7",                 dict(temperature=0.7)),
    ("temp=1.0",                 dict(temperature=1.0)),
    ("temp=1.5",                 dict(temperature=1.5)),
    ("top_k=40",                 dict(top_k=40)),
    ("top_p=0.9 (nucleus)",      dict(top_p=0.9)),
    ("min_p=0.05",               dict(min_p=0.05)),
    ("temp=0.8 + top_p=0.95",    dict(temperature=0.8, top_p=0.95)),
]:
    p = apply_sampling(raw, **kw)
    # Perplexity of the sampling distribution = "effective number of options".
    ent = -(p * p.clamp_min(1e-12).log()).sum()
    print(f"{name:<34}{math.exp(ent.item()):>14.1f}{p.max().item():>11.3f}")

# %% [markdown]
# **Practical defaults:**
#
# | task | settings |
# |---|---|
# | math, code, extraction | `temperature=0` (greedy) — you want the single best answer |
# | general chat | `temperature=0.7, top_p=0.9` |
# | creative writing | `temperature=0.9–1.0, top_p=0.95` |
# | reasoning models | often `temperature=0.6, top_p=0.95` — check the model card |
#
# **`min_p` is underrated.** Unlike top-p it adapts to model confidence: when the
# model is sure, it keeps almost nothing else; when it's unsure, it keeps a wide
# set. Try `min_p=0.05` with no top-k or top-p.

# %% [markdown]
# ## Part 3 — Quantization
#
# Since decode is bandwidth-bound, **halving the bytes nearly halves the
# latency.** Quantization is the highest-leverage inference optimization there is.

# %%
def quantize_absmax(w: torch.Tensor, n_bits: int = 8) -> tuple[torch.Tensor, float]:
    """Simplest possible scheme: symmetric per-tensor absmax."""
    qmax = 2 ** (n_bits - 1) - 1
    scale = w.abs().max().item() / qmax
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return q, scale


def quantize_groupwise(w: torch.Tensor, n_bits: int = 4,
                       group_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group scales — what real 4-bit methods do.

    One scale for a whole tensor is hopeless at 4 bits: a single outlier weight
    stretches the scale and everything else collapses to zero. Per-group scales
    keep outliers local.
    """
    orig_shape = w.shape
    w_flat = w.reshape(-1, group_size)
    qmax = 2 ** (n_bits - 1) - 1
    scales = w_flat.abs().amax(dim=1, keepdim=True) / qmax
    q = torch.clamp(torch.round(w_flat / scales.clamp_min(1e-8)), -qmax - 1, qmax)
    return q.reshape(orig_shape), scales


# Realistic weights: mostly Gaussian, with a few large outliers.
torch.manual_seed(0)
w = torch.randn(512, 512)
w[0, 0] = 25.0          # the outlier that ruins per-tensor scaling
w[10, 20] = -30.0

print(f"{'scheme':<32}{'rel. error':>13}{'bytes/param':>13}")
print("-" * 58)
print(f"{'fp32 (baseline)':<32}{0.0:>13.5f}{4.0:>13.1f}")
print(f"{'bf16':<32}"
      f"{((w.bfloat16().float() - w).norm() / w.norm()).item():>13.5f}{2.0:>13.1f}")

for bits in [8, 4]:
    q, s = quantize_absmax(w, bits)
    err = ((q * s - w).norm() / w.norm()).item()
    print(f"{f'int{bits} per-tensor absmax':<32}{err:>13.5f}{bits/8:>13.2f}")

for bits, gs in [(8, 64), (4, 64), (4, 32)]:
    q, s = quantize_groupwise(w, bits, gs)
    deq = (q.reshape(-1, gs) * s).reshape(w.shape)
    err = ((deq - w).norm() / w.norm()).item()
    overhead = 4 / gs      # one fp32 scale per group
    print(f"{f'int{bits} groupwise (g={gs})':<32}{err:>13.5f}{bits/8 + overhead:>13.2f}")

# %% [markdown]
# Look at the gap between per-tensor and groupwise at 4 bits. Two outlier
# weights out of 262,144 are enough to wreck per-tensor quantization — which is
# exactly why every real 4-bit method (GPTQ, AWQ, NF4, GGUF's K-quants) is
# group-wise.
#
# ### The formats you'll meet
#
# | format | bits | where |
# |---|---|---|
# | **GGUF** (Q4_K_M, Q5_K_M, Q8_0) | 2–8 | llama.cpp, Ollama, LM Studio — **CPU-friendly** |
# | **GPTQ** | 3–4 | GPU, calibration-based |
# | **AWQ** | 4 | GPU, protects salient weights |
# | **NF4** | 4 | bitsandbytes / QLoRA — **training**, not just inference |
# | **FP8** | 8 | native on Hopper/Blackwell — your 5090 has it |
#
# **Rules of thumb:** 8-bit is essentially lossless. 4-bit costs a little
# quality. Below 4 bits degrades fast. And **a 4-bit large model usually beats a
# 16-bit small model at the same memory budget** — quantize down rather than
# scaling down.

# %% [markdown]
# ## Part 4 — Serving
#
# ### vLLM — the default for GPU serving
#
# ```bash
# pip install vllm
# vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000 --max-model-len 4096
# ```
#
# Then it speaks the OpenAI API:
#
# ```python
# from openai import OpenAI
# client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
# r = client.chat.completions.create(
#     model="Qwen/Qwen2.5-1.5B-Instruct",
#     messages=[{"role": "user", "content": "Hello!"}],
# )
# ```
#
# vLLM's key idea is **PagedAttention**: manage the KV cache in fixed-size pages
# like OS virtual memory, instead of one contiguous block per sequence. That
# removes fragmentation and raises throughput several-fold under concurrency.
#
# ### llama.cpp / Ollama — for CPU, laptops, and GGUF
#
# ```bash
# ollama run qwen2.5:1.5b
#
# # convert your own model:
# python convert_hf_to_gguf.py ./my-model --outfile my-model.gguf
# ./llama-quantize my-model.gguf my-model-q4.gguf Q4_K_M
# ```
#
# ### Choosing
#
# | | vLLM | llama.cpp | TGI | plain transformers |
# |---|---|---|---|---|
# | best at | GPU throughput | CPU / portability | HF integration | prototyping |
# | concurrency | excellent | limited | good | poor |
# | quantization | GPTQ/AWQ/FP8 | GGUF (huge range) | several | bitsandbytes |
#
# **Use vLLM on your 5090. Use llama.cpp when you want it to run anywhere.**

# %% [markdown]
# ## Part 5 — Speculative decoding
#
# One more idea worth understanding, because it exploits the bandwidth bound
# directly.
#
# A small **draft** model proposes k tokens cheaply. The big model then verifies
# all k **in a single forward pass** — which costs about the same as generating
# one token, because you were bandwidth-bound anyway. Accepted tokens are free.
#
# The acceptance test is constructed so the output distribution is **exactly**
# that of the big model. It's not an approximation — it's a pure latency win.

# %%
def speculative_speedup(acceptance_rate: float, k: int,
                        draft_cost_ratio: float = 0.1) -> float:
    """Expected speedup from speculative decoding."""
    # Expected accepted tokens per round (geometric, capped at k), +1 for the
    # token the target model always contributes.
    if acceptance_rate >= 1.0:
        expected = k
    else:
        expected = (1 - acceptance_rate ** (k + 1)) / (1 - acceptance_rate) - 1
    tokens_per_round = expected + 1
    cost_per_round = 1 + k * draft_cost_ratio   # one target pass + k draft passes
    return tokens_per_round / cost_per_round


print(f"{'acceptance':>12}" + "".join(f"{f'k={k}':>9}" for k in [2, 4, 8]))
print("-" * 39)
for acc in [0.5, 0.7, 0.8, 0.9]:
    row = f"{acc:>12.0%}"
    for k in [2, 4, 8]:
        row += f"{speculative_speedup(acc, k):>9.2f}x"
    print(row)
print("\nAcceptance depends on how well the draft model mimics the target.")
print("A same-family small model (0.5B drafting for 7B) typically hits 70-85%.")

# %% [markdown]
# ---
#
# # The Capstone
#
# Build one complete model, end to end, and write it up honestly.
#
# ## The task
#
# **Take a base model and turn it into a specialist that measurably beats the
# base model at a task you choose.**
#
# Pick something verifiable so you get a clean signal:
#
# | project | data | reward/metric |
# |---|---|---|
# | **Math reasoner** | GSM8K / MATH | exact answer match |
# | **SQL generator** | Spider / WikiSQL | query executes + returns right rows |
# | **JSON extractor** | your own | schema validates + fields match |
# | **Code fixer** | HumanEval / MBPP | unit tests pass |
# | **Domain assistant** | your own docs | your own rubric |
#
# ## The pipeline
#
# ```
# 1. DATA          (nb 01)  collect, filter, dedup, inspect
# 2. BASELINE      (nb 14)  measure the base model FIRST — with CIs
# 3. SFT           (nb 08)  LoRA on task-formatted examples
# 4. EVAL          (nb 14)  did it help? is it significant?
# 5. PREFERENCE    (nb 11)  DPO on pairs, if you have them
#    or RLVR       (nb 13)  GRPO, if you have a verifier
# 6. EVAL AGAIN    (nb 14)  same protocol, same n
# 7. SERVE         (nb 15)  merge, quantize, benchmark latency
# 8. WRITE IT UP
# ```
#
# ## Requirements for a good writeup
#
# - Baseline and final numbers **with confidence intervals**
# - Exact decoding settings for every measurement
# - A **negative result** — something you tried that didn't work
# - An honest failure analysis: read 20 wrong outputs and categorize them
# - Total GPU-hours and cost
#
# That last section is what separates engineering from wishful thinking.

# %%
def capstone_report(name: str, baseline: dict, final: dict,
                    gpu_hours: float, notes: str = "") -> str:
    """Generate a writeup skeleton with the statistics done correctly."""
    n = baseline["n"]
    b, f = baseline["accuracy"], final["accuracy"]
    p_pool = (b + f) / 2
    se = math.sqrt(2 * p_pool * (1 - p_pool) / n) if n else float("inf")
    z = (f - b) / se if se else 0.0
    sig = abs(z) > 1.96

    ci = lambda p: 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n)

    return f"""
# {name}

## Results

| model | accuracy | 95% CI | n |
|---|---|---|---|
| baseline | {b:.1%} | ±{ci(b):.1%} | {n} |
| final    | {f:.1%} | ±{ci(f):.1%} | {n} |

**Delta: {100*(f-b):+.1f} pp** (z = {z:.2f}, {"significant at p<0.05"
                                             if sig else "NOT significant"})

## Cost
{gpu_hours:.1f} GPU-hours

## Notes
{notes or "(what worked, what didn't, and what you'd do differently)"}
"""


print(capstone_report(
    "GSM8K Specialist (Qwen2.5-0.5B + LoRA SFT + GRPO)",
    baseline={"accuracy": 0.31, "n": 1319},
    final={"accuracy": 0.42, "n": 1319},
    gpu_hours=6.5,
    notes="SFT alone gave +4pp. GRPO added +7pp more. Dropping the format\n"
          "reward stalled training entirely — the model never found the\n"
          "answer tags on its own.",
))

# %% [markdown]
# ## Where to go next
#
# **Repos to read now that you can read them:**
#
# - [karpathy/nanochat](https://github.com/karpathy/nanochat) — the whole
#   pipeline in one clean codebase; the natural next step from this course
# - [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — the classic
#   minimal pretraining implementation
# - [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) —
#   the speedrun; where Muon and other tricks get proven
# - [huggingface/trl](https://github.com/huggingface/trl) — read the trainers
#   you used
# - [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) —
#   book-length treatment of notebooks 02–07
# - [rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) —
#   goes deeper on the reasoning-model side
# - [volcengine/verl](https://github.com/volcengine/verl) — production RL
#   infrastructure, when you outgrow TRL
#
# **Papers, in reading order:**
#
# 1. Attention Is All You Need (2017)
# 2. GPT-2 / GPT-3 (2019, 2020)
# 3. Chinchilla — Training Compute-Optimal LLMs (2022)
# 4. LoRA (2021), QLoRA (2023)
# 5. InstructGPT (2022) — the RLHF pipeline
# 6. DPO (2023)
# 7. DeepSeekMath (GRPO, 2024), DeepSeek-R1 (2025)
# 8. FineWeb / FineWeb-Edu (2024) — the data paper
#
# ## Final checklist
#
# - [ ] Built a tokenizer, and know why prompts shouldn't end in a space
# - [ ] Built a transformer, and can prove it's causal
# - [ ] Pretrained a model to coherent English
# - [ ] Know why 7B full-finetune won't fit and QLoRA will
# - [ ] SFT'd a model and know why labels are −100 on the prompt
# - [ ] Can write the DPO loss and name its failure mode
# - [ ] Implemented GRPO and know why zero-variance groups teach nothing
# - [ ] Can compute whether a benchmark difference is significant
# - [ ] Served a quantized model
#
# If you can tick those, you understand how modern LLMs are built. The rest is
# scale and engineering.

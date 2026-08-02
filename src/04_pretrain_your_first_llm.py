# %% [markdown]
# # 04 — Pretraining: Actually Train the Thing
#
# **Goal:** train a real language model from random weights to coherent English,
# with a training loop that has every piece a production run has — LR schedule,
# gradient accumulation, clipping, mixed precision, checkpointing, evaluation.
#
# **Time:** 15 min for the TinyStories run, 3–6 h for the FineWeb-Edu run.
#
# ## In plain language
#
# ### What "training" actually means
#
# This is the notebook where a model is created. It is worth being concrete about
# what that sentence means, because "training an AI" sounds mystical and is not.
#
# **The model starts as random numbers.** Millions of them. Right now it knows
# nothing — feed it text and it outputs noise.
#
# **Training is a loop that repeats a few thousand times.** Each pass through it
# does exactly four things:
#
# 1. Grab a random chunk of text from your `.bin` file — say 512 tokens.
# 2. Hide the last token and ask the model: *what comes next?*
# 3. Compare its guess to the real answer. The gap is the **loss**.
# 4. Nudge every one of those millions of numbers a tiny amount in the direction
#    that would have made the guess better.
#
# That's it. Repeat a few thousand times and the numbers stop being random and
# start encoding grammar, facts, and style — because those are what let you
# predict text well.
#
# **Nobody tells the model any rules.** No one writes down "adjectives come
# before nouns" or "sentences end with a full stop." Those patterns emerge
# because they help step 3 go better. Everything a language model knows, it
# learned from being repeatedly asked *what comes next*.
#
# **What you actually do:** run the cell, watch a number go down for fifteen
# minutes, then read what the model writes.
#
# ### What you'll have at the end
#
# A file — `artifacts/checkpoints/smoke.pt`, a few dozen MB — holding the trained
# numbers. Load it, give it a few words, and it continues them:
#
# > **you type:** `Once upon a time there was a little`
# > **it writes:** `girl named Lily. She had a red ball and liked to play in the
# > park with her friend Tom.`
#
# **That is the whole goal of this notebook.** Grammatical, coherent English that
# the model invented — nobody wrote that sentence, it was generated one token at
# a time from patterns it found on its own.
#
# **Be clear about what it will not do**, so you are not disappointed:
#
# | you might expect | reality |
# |---|---|
# | answer a question | no — it continues text, it does not respond (notebook 07 fixes this) |
# | know facts about the world | no — it only ever read children's stories |
# | do arithmetic | no |
# | hold a conversation | no — it has no idea it is talking to anyone |
# | write fluent simple English | **yes — and that is the win** |
#
# The difference between "continues text" and "answers questions" is the entire
# subject of notebooks 07 onward. A model at this stage is like someone who has
# read a lot and can finish your sentences, but has never been told that when
# someone asks a question they are supposed to reply. That's a *separate* thing
# to teach, and it comes later.
#
# ## The plan
#
# Do this in two stages, and **do not skip stage 1**:
#
# | stage | model | data | time on a 5090 | purpose |
# |---|---|---|---|---|
# | 1. smoke test | ~10M | TinyStories 20M tok | ~15 min | prove the loop works |
# | 2. real run | 124M | FineWeb-Edu 500M tok | ~3–5 h | a model you can actually use |
#
# Stage 1 exists because debugging a 5-hour run is agony. If the loop is broken
# you want to know in 15 minutes.
#
# In restaurant terms (notebook 08): this is the chef's fifteen years in Italy.
# Expensive, done once, and it produces someone who can *cook* — not someone who
# knows your menu. Everything after this notebook is teaching them your menu.

# %%
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DATA = Path("../data")
CKPT = Path("../artifacts/checkpoints")
CKPT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
# TF32: lets fp32 matmuls use tensor cores with ~10-bit mantissa. Big speedup,
# no meaningful accuracy cost for training. Free performance on Ampere+.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"device: {device}")
if device == "cuda":
    print(f"gpu:    {torch.cuda.get_device_name()}")

# %% [markdown]
# ## The model
#
# We import the GPT from notebook 03 rather than redefining it. `llmfs/` holds
# the shared implementation so later notebooks all agree on the architecture.

# %%
import sys

sys.path.insert(0, str(Path("..").resolve()))
from llmfs.model import GPT, GPTConfig  # noqa: E402

print("model code loaded from llmfs/model.py")

# %% [markdown]
# ## Data loading: `np.memmap` and why there's no DataLoader
#
# Our corpus is a flat `uint16` array on disk. To make a batch we pick `B`
# random offsets and slice `T+1` tokens at each:
#
# ```
# tokens:  [ the | cat | sat | on | the | mat ]
# x:       [ the | cat | sat | on | the ]
# y:       [ cat | sat | on  | the| mat ]      <- x shifted by one
# ```
#
# Position `i` of `x` predicts position `i` of `y`. One sequence of length T
# gives T training signals, not one — that density is why language modeling is
# such an efficient learning objective.
#
# **Why not `torch.utils.data.DataLoader`?** For this access pattern it's pure
# overhead: no shuffling to manage (random offsets are already random), no
# collation (all samples are the same length), no decoding. `memmap` lets the OS
# page cache do the work, and after the first epoch the hot data lives in RAM.

# %%
class BinDataset:
    """Random-offset sampler over a flat uint16 token file."""

    def __init__(self, path: Path, block_size: int, device: str = "cpu") -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found — run notebook 01 to build the corpus"
            )
        self.block_size = block_size
        self.device = device
        # np.memmap does NOT load the file into RAM; it maps it into the address
        # space and the OS pages in what you touch. A 100 GB corpus works fine.
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)
        if self.n_tokens <= block_size + 1:
            raise ValueError(
                f"{self.path.name} has only {self.n_tokens:,} tokens, which is too "
                f"few for block_size={block_size}. Build a larger corpus."
            )
        # Note: np.random.default_rng() (a Generator) is not the same object as
        # the legacy np.random module — only the Generator has .integers().
        self._rng = np.random.default_rng()

    def get_batch(self, batch_size: int, generator: np.random.Generator | None = None):
        rng = generator if generator is not None else self._rng
        ix = rng.integers(0, self.n_tokens - self.block_size - 1, size=batch_size)
        # astype(int64) is required: torch embeddings need long indices, and
        # uint16 isn't a dtype torch supports.
        x = torch.from_numpy(
            np.stack([self.data[i : i + self.block_size].astype(np.int64) for i in ix])
        )
        y = torch.from_numpy(
            np.stack(
                [self.data[i + 1 : i + 1 + self.block_size].astype(np.int64) for i in ix]
            )
        )
        if self.device.startswith("cuda"):
            # pin_memory + non_blocking overlaps the host->device copy with
            # compute. Worth ~5% on a data-light workload like this.
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def __repr__(self) -> str:
        return f"BinDataset({self.path.name}, {self.n_tokens:,} tokens)"


# %% [markdown]
# ## The learning-rate schedule
#
# Two components, both non-negotiable:
#
# **Warmup.** Adam's `v` (second-moment) estimate is near-zero at step 0, so the
# effective step size is enormous and the first few updates can wreck the
# initialization. Ramp linearly from 0 over a few hundred steps.
#
# **Cosine decay.** Anneal to ~10% of peak. Large LR early explores; small LR
# late refines. Cosine consistently beats linear and step decay in practice.
#
# The shape is worth seeing rather than reading about.

# %%
def get_lr(step: int, *, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))   # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)


_steps = list(range(0, 5000, 25))
_lrs = [get_lr(s, max_lr=6e-4, min_lr=6e-5, warmup_steps=300, max_steps=5000) for s in _steps]

try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(_steps, _lrs)
    ax.axvline(300, color="crimson", ls="--", lw=1, label="warmup ends")
    ax.set(xlabel="step", ylabel="learning rate", title="warmup + cosine decay")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
except ImportError:
    for s in [0, 100, 300, 1000, 2500, 5000]:
        print(f"  step {s:>5}: lr {get_lr(s, max_lr=6e-4, min_lr=6e-5, warmup_steps=300, max_steps=5000):.2e}")

# %% [markdown]
# ## Gradient accumulation: large batches on one GPU
#
# Large batch sizes stabilize training, but a 0.5M-token batch will not fit in
# 24 GB. Accumulation splits it: run several small forward/backward passes,
# summing gradients, then step once.
#
# ```
# effective_batch = micro_batch × grad_accum_steps × block_size   (tokens)
# ```
#
# **The subtlety that bites everyone:** `cross_entropy` returns the *mean* over
# the micro-batch. Summing K micro-batch gradients gives you K× the gradient of
# the true mean. You must divide each micro-batch loss by `grad_accum_steps`.
# Forget this and your effective LR is K× too high — the run diverges and you
# blame the learning rate.

# %%
@dataclass
class TrainConfig:
    # data
    train_bin: str = "tinystories_train_split_train.bin"
    val_bin: str = "tinystories_train_split_val.bin"
    # model
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0        # pretraining on lots of data: no dropout needed
    # optimisation
    micro_batch: int = 32
    grad_accum: int = 4
    max_steps: int = 2000
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # logging / eval
    eval_every: int = 200
    eval_iters: int = 40
    log_every: int = 20
    ckpt_name: str = "tinystories_10m"
    compile_model: bool = False

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch * self.grad_accum * self.block_size


smoke = TrainConfig()
print(f"tokens per optimizer step: {smoke.tokens_per_step:,}")
print(f"total tokens seen:         {smoke.tokens_per_step * smoke.max_steps / 1e6:.1f}M")

# %% [markdown]
# ## The optimizer: which parameters get weight decay?
#
# A detail that's easy to get wrong and quietly costs you quality.
#
# **Decay** matrices (Linear and Embedding weights) — these are the parameters
# where shrinking toward zero is a meaningful regularizer.
#
# **Don't decay** biases and LayerNorm gains. A LayerNorm gain of 1.0 means
# "pass through unchanged"; decaying it toward 0 actively fights the network's
# ability to preserve scale. Same for biases — decaying a bias just shifts the
# function for no regularization benefit.
#
# The rule of thumb: **decay tensors with ≥2 dimensions, don't decay 1-D ones.**

# %%
def configure_optimizer(model: nn.Module, weight_decay: float, lr: float, device: str):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    decay = [p for p in params.values() if p.dim() >= 2]
    no_decay = [p for p in params.values() if p.dim() < 2]

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    print(f"  decayed:     {len(decay):>3} tensors, {sum(p.numel() for p in decay):>12,} params")
    print(f"  not decayed: {len(no_decay):>3} tensors, {sum(p.numel() for p in no_decay):>12,} params")

    # fused AdamW does the whole update in one CUDA kernel instead of many
    # small elementwise ops. Meaningfully faster; CUDA only.
    use_fused = device.startswith("cuda")
    return torch.optim.AdamW(
        groups,
        lr=lr,
        betas=(0.9, 0.95),   # beta2=0.95 (not the 0.999 default) is standard for
                             # LLMs: adapts faster to the changing loss landscape
        eps=1e-8,
        fused=use_fused,
    )


# %% [markdown]
# ## Evaluation during training
#
# Estimate loss over several batches (a single batch is far too noisy) with the
# model in `eval()` mode and gradients off.
#
# **Perplexity = exp(loss)** is the interpretable version: "the model is as
# confused as if it were choosing uniformly among PPL options." A perplexity of
# 20 is a decent small model; 1000 means it has learned almost nothing.

# %%
@torch.no_grad()
def estimate_loss(model, datasets: dict, cfg: TrainConfig, autocast_ctx) -> dict:
    out = {}
    model.eval()
    rng = np.random.default_rng(0)   # fixed seed => comparable across steps
    for split, ds in datasets.items():
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            xb, yb = ds.get_batch(cfg.micro_batch, generator=rng)
            with autocast_ctx:
                _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# %% [markdown]
# ## The training loop
#
# Everything assembled. Read it once before running — this is the shape of every
# LLM training loop you will ever see.

# %%
def train(cfg: TrainConfig, resume: bool = False) -> dict:
    train_ds = BinDataset(DATA / cfg.train_bin, cfg.block_size, device)
    val_ds = BinDataset(DATA / cfg.val_bin, cfg.block_size, device)
    print(f"train: {train_ds}\nval:   {val_ds}\n")

    model_cfg = GPTConfig(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = GPT(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M parameters")

    print("optimizer groups:")
    optimizer = configure_optimizer(model, cfg.weight_decay, cfg.max_lr, device)

    # bf16 autocast: matmuls run in bf16, reductions stay fp32. No GradScaler
    # needed (that's an fp16 requirement), which keeps the loop simple.
    if device.startswith("cuda"):
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)

    start_step = 0
    ckpt_path = CKPT / f"{cfg.ckpt_name}.pt"
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_step = ck["step"] + 1
        print(f"resumed from step {start_step}")

    if cfg.compile_model:
        # torch.compile traces and fuses the graph. 1.3-2x speedup, but the
        # first step takes ~1-2 minutes to compile. Worth it for long runs only.
        print("compiling (first step will be slow)...")
        model = torch.compile(model)

    history = {"step": [], "train_loss": [], "val_loss": [], "lr": []}
    running = None
    t0 = time.time()
    tokens_done = 0

    print(f"\n{'step':>6} {'loss':>8} {'lr':>9} {'tok/s':>10} {'elapsed':>9}")
    print("-" * 48)

    model.train()
    for step in range(start_step, cfg.max_steps):
        lr = get_lr(
            step,
            max_lr=cfg.max_lr,
            min_lr=cfg.min_lr,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(cfg.grad_accum):
            xb, yb = train_ds.get_batch(cfg.micro_batch)
            with autocast_ctx:
                _, loss = model(xb, yb)
                # THE division that everyone forgets. Without it the gradient
                # is grad_accum times too large.
                loss = loss / cfg.grad_accum
            loss.backward()
            accum_loss += loss.item()

        # Clip by GLOBAL norm across all parameters. This is the single most
        # effective defence against loss spikes: one bad batch (a weird
        # document, a numerical edge case) produces a huge gradient that would
        # otherwise blow up the weights permanently.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        tokens_done += cfg.tokens_per_step
        running = accum_loss if running is None else 0.9 * running + 0.1 * accum_loss

        if step % cfg.log_every == 0:
            if device.startswith("cuda"):
                torch.cuda.synchronize()   # timings are meaningless without this
            el = time.time() - t0
            print(f"{step:>6} {running:>8.4f} {lr:>9.2e} {tokens_done/el:>10,.0f} {el:>8.0f}s")

        if step > 0 and step % cfg.eval_every == 0 or step == cfg.max_steps - 1:
            losses = estimate_loss(model, {"train": train_ds, "val": val_ds}, cfg, autocast_ctx)
            print(
                f"  >> eval  train {losses['train']:.4f}  val {losses['val']:.4f}  "
                f"(val ppl {math.exp(min(losses['val'], 20)):.1f})  |grad| {grad_norm:.2f}"
            )
            history["step"].append(step)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])
            history["lr"].append(lr)

            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(
                {
                    "model": raw.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": asdict(model_cfg),
                    "train_config": asdict(cfg),
                    "step": step,
                    "val_loss": losses["val"],
                },
                ckpt_path,
            )

    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {ckpt_path}")
    return history


# %% [markdown]
# ## What the loss number actually means
#
# You are about to stare at one number for fifteen minutes, so it is worth
# knowing what it measures.
#
# The model outputs a probability for every token in the vocabulary at every
# position. The loss is **cross-entropy**: the negative log of the probability
# it assigned to the token that actually came next.
#
# ```
# loss = -ln( p(correct next token) )
# ```
#
# That is all. Which gives you a conversion you can do in your head:
#
# | loss | `p(correct token)` | what it means |
# |---|---|---|
# | 10.82 | 1/50257 | uniform guessing over the vocabulary |
# | 6.9 | 1/1000 | narrowed to ~1000 plausible tokens |
# | 4.6 | 1/100 | narrowed to ~100 |
# | 2.3 | 1/10 | narrowed to ~10 — text starts looking like English |
# | 1.6 | ~1/5 | fluent on simple text |
# | 0.0 | 1.0 | perfect prediction (means you are overfitting) |
#
# **Why training starts at ln(vocab_size).** At initialization the model knows
# nothing, so it spreads probability uniformly: `p = 1/50257` for every token,
# and `-ln(1/50257) = 10.82`. This is the single most useful debugging check in
# the whole course. If your first loss is not ≈10.82, stop — you have a bug
# *before* you have a training problem:
#
# | first loss | almost certainly |
# |---|---|
# | ≈ 10.82 | correct |
# | ≈ 0 | labels leaked into the input; the model can see the answer |
# | 15–20+ | bad init (weights too large), or logits not scaled |
# | `nan` | `-inf` in the input, or fp16 overflow |
#
# **Perplexity** is just `exp(loss)`, and it has a nicer reading: "the model is
# as confused as if it were choosing uniformly among this many options." Loss
# 2.3 → perplexity 10 → "about 10 plausible next tokens." Papers report
# perplexity; training loops print loss; they are the same fact.
#
# One caution: **perplexity is only comparable within the same tokenizer.** A
# model with a 32k vocab and one with a 128k vocab cannot be compared by
# perplexity, because they are not predicting the same units. This trips up a
# lot of model comparisons on leaderboards.
#
# ## Run 1 — the smoke test (~15 min)
#
# A 10M-parameter model on TinyStories. Watch for:
#
# - loss starts near **ln(50257) ≈ 10.82**
# - drops below 4 within a few hundred steps
# - ends somewhere around **1.5–2.5**
#
# TinyStories has a small vocabulary and simple grammar, so low loss is
# achievable. If loss plateaus above 5, something is wrong.
#
# **What you should see.** Roughly this shape — the exact numbers will drift,
# the shape should not:
#
# ```
#   step     train      val       lr
#      0   10.8241  10.8198  6.00e-05
#    100    5.1032   5.0876  3.00e-04
#    500    2.8814   2.9001  2.87e-04
#   1000    2.1077   2.1355  2.41e-04
#   2000    1.7215   1.7684  1.24e-04
# ```
#
# Three things to notice, because each one teaches something:
#
# **The first drop is enormous, then it slows.** 10.8 → 5.0 in 100 steps is the
# model learning token *frequency* — that "the" is common and "zygote" is not.
# That is cheap. Everything after is learning *context*, which is the hard part.
# A loss curve that looks like it stalled after the first plunge has not stalled;
# that is what learning looks like from step 200 onward.
#
# **Val tracks train closely here.** On 20M tokens with a 10M-parameter model you
# are nowhere near enough capacity to memorize, so the two curves sit on top of
# each other. When you scale to the 124M model on FineWeb-Edu, watch for them to
# separate — that gap *is* overfitting, measured.
#
# **The learning rate rises, then falls.** It climbs during warmup and decays
# after. If your loss spikes exactly when the LR peaks, your peak LR is too high.
#
# Timing on an RTX 5090: expect **12–20 minutes** for the smoke run. If it is
# projecting hours, your throughput is wrong — check `tokens/sec` against
# notebook 06 before letting it run.

# %%
history_smoke = train(smoke)

# %% [markdown]
# ## Look at the curves
#
# **Read your loss curve** — it tells you what to change:
#
# | pattern | diagnosis | fix |
# |---|---|---|
# | val tracks train, both falling | healthy | train longer / go bigger |
# | val flattens while train falls | overfitting | more data, or add dropout |
# | sudden spike | bad batch or LR too high | lower LR, tighter clipping |
# | NaN | numerical blowup | check for `-inf`, lower LR |
# | flat from step 0 | not learning at all | LR far too low, or a wiring bug |

# %%
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(history_smoke["step"], history_smoke["train_loss"], "o-", label="train")
    axes[0].plot(history_smoke["step"], history_smoke["val_loss"], "s-", label="val")
    axes[0].axhline(math.log(50257), color="gray", ls=":", label="random baseline")
    axes[0].set(xlabel="step", ylabel="cross-entropy loss", title="loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    ppl = [math.exp(min(v, 20)) for v in history_smoke["val_loss"]]
    axes[1].plot(history_smoke["step"], ppl, "s-", color="darkgreen")
    axes[1].set(xlabel="step", ylabel="perplexity", title="validation perplexity", yscale="log")
    axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()
except ImportError:
    for s, tr, va in zip(history_smoke["step"], history_smoke["train_loss"], history_smoke["val_loss"]):
        print(f"  step {s:>5}  train {tr:.4f}  val {va:.4f}  ppl {math.exp(min(va,20)):.1f}")

# %% [markdown]
# ## Generate — the moment of truth

# %%
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")


def load_model(name: str):
    ck = torch.load(CKPT / f"{name}.pt", map_location=device, weights_only=False)
    m = GPT(GPTConfig(**ck["model_config"])).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    print(f"loaded {name} @ step {ck['step']}, val loss {ck['val_loss']:.4f}")
    return m


@torch.no_grad()
def sample(model, prompt: str, max_new_tokens: int = 100,
           temperature: float = 0.8, top_k: int = 50) -> str:
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        ids_cond = ids[:, -model.cfg.block_size :]
        logits, _ = model(ids_cond)
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        ids = torch.cat([ids, nxt], dim=1)
    return tok.decode(ids[0].tolist())


m = load_model(smoke.ckpt_name)
for prompt in ["Once upon a time", "The little girl", "Tom and Sara went to"]:
    print(f"\n--- {prompt!r} ---")
    print(sample(m, prompt))

# %% [markdown]
# You should see grammatical, coherent (if simple) stories. **A 10M-parameter
# model wrote that** — it has learned English syntax, subject–verb agreement,
# narrative structure, and basic pronoun consistency, purely from next-token
# prediction. Nothing else was supervised.
#
# It knows no facts about the world, because TinyStories contains none. That's
# what the FineWeb-Edu run is for.

# %% [markdown]
# ## Temperature and top-k, made concrete
#
# Two knobs on the same distribution:
#
# - **temperature** divides the logits. `<1` sharpens (safer, repetitive), `>1`
#   flattens (more diverse, more incoherent). `t→0` is greedy argmax.
# - **top-k** truncates to the k most likely tokens before sampling, cutting the
#   long tail of nonsense that would otherwise occasionally get picked.

# %%
for t in [0.1, 0.7, 1.0, 1.5]:
    print(f"\n--- temperature {t} ---")
    print(sample(m, "Once upon a time", max_new_tokens=60, temperature=t, top_k=200))

# %% [markdown]
# Low temperature loops and repeats; high temperature drifts into incoherence.
# **0.7–0.9 is the usual sweet spot** for creative text; use 0.0–0.3 for tasks
# with a single correct answer (math, code, extraction).

# %% [markdown]
# ## Run 2 — the real model (3–6 hours)
#
# 124M parameters on FineWeb-Edu. Same loop, bigger everything.
#
# **Before you start a long run, sanity-check the throughput.** Let it log a few
# steps, note the tok/s, and compute total time. If the estimate is 40 hours,
# stop and reduce `max_steps` — don't discover that at hour 8.
#
# Tuning for 24 GB: if you OOM, **halve `micro_batch` and double `grad_accum`.**
# The effective batch is unchanged, so the run is mathematically the same.

# %%
real = TrainConfig(
    train_bin="fineweb_edu_train_split_train.bin",
    val_bin="fineweb_edu_train_split_val.bin",
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    micro_batch=12,          # ~14-18 GB on a 24 GB card at bf16
    grad_accum=40,           # -> ~491k tokens per optimizer step
    max_steps=1000,
    max_lr=6e-4,
    min_lr=6e-5,
    warmup_steps=100,
    eval_every=100,
    ckpt_name="fineweb_124m",
    compile_model=True,      # worth the compile cost over 1000 steps
)

print(f"tokens/step:  {real.tokens_per_step:,}")
print(f"total tokens: {real.tokens_per_step * real.max_steps / 1e9:.2f}B")
print(f"\nA 5090 should do roughly 60-110k tok/s for a 124M model at bf16.")
print(f"At 85k tok/s that's ~{real.tokens_per_step*real.max_steps/85_000/3600:.1f} hours.")

# %%
# Uncomment to launch the real run. Consider running it as a script instead
# (`python train.py`) so a browser crash doesn't kill 5 hours of work.
#
# history_real = train(real)

# %% [markdown]
# ### Expected results for the 124M run
#
# | tokens seen | val loss | perplexity | what it can do |
# |---|---|---|---|
# | 100M | ~4.5 | ~90 | grammatical fragments |
# | 500M | ~3.8 | ~45 | coherent paragraphs, some facts |
# | 2B | ~3.3 | ~27 | decent short-form text |
# | 10B | ~3.0 | ~20 | approaching real GPT-2 small (~3.0) |
#
# For reference, GPT-2 small was trained on ~10B tokens. Matching it takes real
# time; getting most of the way there takes an afternoon.

# %% [markdown]
# ## Debugging guide
#
# **Loss is NaN.** Almost always LR too high, or an `-inf` from the attention
# mask surviving into the softmax. Lower `max_lr` 3×, verify `grad_clip` is on.
# If it appears at a specific step, print that batch — you may have a corrupt
# region in the `.bin`.
#
# **Loss plateaus high (>6) and won't move.** Check `get_lr` is actually being
# applied to `param_groups` (a classic no-op bug). Check your targets are
# shifted by exactly one. Re-run the overfit-one-batch test from notebook 03.
#
# **OOM.** Halve `micro_batch`, double `grad_accum`. Then enable gradient
# checkpointing (notebook 06). Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
#
# **Throughput much lower than expected.** Is `compile_model` on? Is autocast
# actually active? Is the `.bin` on `/mnt/c/`? (Move it to the Linux filesystem.)
# Is another process using the GPU — check `nvidia-smi`.
#
# **Loss spikes then recovers.** Normal, if occasional. Persistent spikes mean
# LR too high or clipping too loose.
#
# ## Exercises
#
# 1. **Batch size sweep.** At fixed total tokens, compare `micro_batch × accum`
#    of 8×8, 16×4, 32×2. Same effective batch — is wall-clock the same?
# 2. **No warmup.** Set `warmup_steps=0` with `max_lr=1e-3`. Watch the first 50
#    steps. This is why warmup exists.
# 3. **Depth vs width.** At ~matched parameter count, compare 12 layers × 768
#    against 6 layers × 1088. Which reaches lower val loss?
# 4. **Resume.** Kill training mid-run and restart with `resume=True`. Verify
#    the loss picks up where it left off rather than spiking.
#
# ## Checkpoint
#
# - [ ] `artifacts/checkpoints/tinystories_10m.pt` exists
# - [ ] Generated text is coherent English
# - [ ] You can explain why the loss is divided by `grad_accum`
# - [ ] You know what to change first when you OOM
#
# **Next:** `05_modern_architecture.ipynb` — upgrade GPT-2 (2019) to Llama
# (2024+): RoPE, RMSNorm, SwiGLU, GQA.

# %% [markdown]
# # 00 — Setup & Hardware Check (RTX 5090 / WSL2)
#
# **Goal:** get a working GPU stack and learn to read your own VRAM budget, so
# every later notebook can tell you "this will fit" or "this will OOM" *before*
# you spend an hour on it.
#
# **Time:** 20–40 min (mostly downloads). **Run once.**
#
# ## In plain language
#
# **What you're doing:** checking that your graphics card works for this kind of
# maths, and learning to predict how much of it will fit in memory.
#
# **The everyday version.** You've bought an oven and you're about to cook in it
# for weeks. Before you start, you check two things: does it actually heat up,
# and how big a dish fits inside? Getting halfway through a five-hour roast and
# discovering the tray doesn't fit is the thing we're avoiding.
#
# **Why this isn't optional.** GPUs have a fixed amount of memory — yours has
# 24 GB. If your model needs 30 GB, training doesn't slow down, it *stops*, with
# an error, usually twenty minutes in. There's no warning. The skill you learn
# here is doing the arithmetic beforehand, on paper, in about ten seconds.
#
# **What you'll have at the end:**
#
# - a confirmed-working PyTorch that actually computes on your GPU
# - the ability to look at any model and say "that fits" or "that won't"
# - a Hugging Face login so later notebooks can download models
#
# **What could go wrong:** almost all of it is one problem — the wrong PyTorch
# build. Your RTX 5090 is new enough that ordinary `pip install torch` gives you
# a version that *claims* to work and then fails on the first real calculation.
# The next section is about that specifically. Don't skip it.
#
# ## The one gotcha that will waste your evening
#
# The RTX 5090 is **Blackwell**, CUDA compute capability **sm_120**. PyTorch
# wheels built before CUDA 12.8 contain no compiled kernels for sm_120. The
# failure mode is nasty: `torch.cuda.is_available()` returns `True`, memory
# allocates fine, and then the first real matmul dies with:
#
# ```
# CUDA error: no kernel image is available for execution on the device
# ```
#
# or
#
# ```
# NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible
# with the current PyTorch installation.
# ```
#
# You need a **cu128 (or newer) build**. Everything else in this course depends
# on getting this right, so we verify it properly below rather than trusting
# `is_available()`.

# %% [markdown]
# ## Step 1 — Create the environment (run in your WSL2 terminal, not here)
#
# ```bash
# # from the repo root
# cd llm-from-scratch
#
# python3 -m venv .venv
# source .venv/bin/activate
# python -m pip install -U pip wheel
#
# # PyTorch for Blackwell — the --index-url is NOT optional.
# # Without it pip silently gives you a CPU build or a pre-sm_120 CUDA build.
# pip install torch --index-url https://download.pytorch.org/whl/cu128
#
# # Everything else comes from normal PyPI
# pip install -r requirements.txt
#
# # register the venv as a Jupyter kernel, then pick it in the notebook UI
# python -m ipykernel install --user --name llm-fs --display-name "llm-from-scratch"
# ```
#
# **WSL2 notes**
# - Install the NVIDIA driver on **Windows**, not inside WSL. WSL sees the GPU
#   through `/usr/lib/wsl/lib`. Installing a Linux driver inside WSL breaks it.
# - `nvidia-smi` should work inside WSL with no extra install. If it doesn't,
#   update your Windows driver first.
# - Put this repo on the **Linux filesystem** (`~/code/...`), not `/mnt/c/...`.
#   Dataset loading across the 9p mount is roughly 10× slower.
# - Give WSL enough RAM. In `C:\Users\<you>\.wslconfig`:
#   ```ini
#   [wsl2]
#   memory=48GB
#   swap=16GB
#   ```
#   (You have 64 GB; leaving ~16 GB to Windows is a sane split.)

# %% [markdown]
# ## Step 2 — Verify the install *actually* computes
#
# This is the important cell. It does not just ask "is CUDA available" — it
# forces a real matmul on the GPU, which is what catches the sm_120 mismatch.

# %%
import platform
import sys

import torch

print(f"python   {sys.version.split()[0]}  ({platform.system()} {platform.release()})")
print(f"torch    {torch.__version__}")
print(f"built for CUDA {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\nNo CUDA. On WSL2 check: `nvidia-smi` works, and you installed the cu128 wheel.")
else:
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    cap = torch.cuda.get_device_capability(idx)
    print(f"\ndevice   {props.name}")
    print(f"capability sm_{cap[0]}{cap[1]}")
    print(f"VRAM     {props.total_memory / 1024**3:.1f} GiB")
    print(f"SMs      {props.multi_processor_count}")

    supported = torch.cuda.get_arch_list()
    print(f"\ntorch was compiled for: {supported}")
    if f"sm_{cap[0]}{cap[1]}" not in supported:
        print(
            f"\n!! MISMATCH: your GPU is sm_{cap[0]}{cap[1]} but this torch build "
            f"has no kernels for it.\n"
            f"   Reinstall:  pip install --force-reinstall torch "
            f"--index-url https://download.pytorch.org/whl/cu128"
        )

    # The real test: allocate and compute. This is what actually fails on a bad build.
    try:
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        c = (a @ b).float().sum().item()
        torch.cuda.synchronize()
        print(f"\nOK — real bf16 matmul on GPU succeeded (checksum {c:.1f})")
    except RuntimeError as exc:
        print(f"\n!! GPU compute FAILED: {exc}")
        print("   This is the sm_120 problem. Reinstall with the cu128 index URL.")

# %% [markdown]
# ### Why bf16 and not fp16?
#
# You will see `bfloat16` everywhere in this course. Both are 16-bit, but they
# spend their bits differently:
#
# | dtype | exponent bits | mantissa bits | dynamic range | needs loss scaling? |
# |---|---|---|---|---|
# | fp32 | 8 | 23 | huge | no |
# | fp16 | 5 | 10 | **narrow** — overflows ~65504 | **yes** |
# | bf16 | 8 | 7 | same as fp32 | **no** |
#
# bf16 keeps fp32's exponent, so activations and gradients don't overflow, and
# you can skip the `GradScaler` dance entirely. It has fewer mantissa bits (less
# precision) but for training that trades away almost nothing. Every Ampere or
# newer card supports it, including yours. **Use bf16 unless you have a specific
# reason not to.**

# %% [markdown]
# ## Step 3 — Learn to budget VRAM before you train
#
# The single most useful skill for a solo GPU. Training memory is roughly:
#
# ```
# total ≈ parameters + gradients + optimizer state + activations + fragmentation
# ```
#
# For **full fine-tuning / pretraining with AdamW in mixed precision**, per parameter:
#
# | what | bytes/param | why |
# |---|---|---|
# | weights (bf16) | 2 | the model itself |
# | master weights (fp32) | 4 | AdamW keeps an fp32 copy for stable updates |
# | gradients | 2–4 | one per parameter |
# | Adam `m` (momentum) | 4 | fp32 |
# | Adam `v` (variance) | 4 | fp32 |
# | **≈ total** | **~16–18** | before any activations |
#
# So a 1B-parameter model needs **~16 GB just to hold state**, before a single
# token of activation. That is why full-finetuning a 7B model on 24 GB is
# impossible, and why LoRA exists (notebook 08).
#
# **Why 16 bytes and not 2.** This surprises everyone the first time, so it is
# worth walking through. You might reasonably expect a bf16 model to cost 2 bytes
# per parameter. The other 14 are the *optimizer*, and they are not optional:
#
# - AdamW keeps an **fp32 master copy** of every weight (4 bytes). Updates are
#   tiny — often 1e-7 relative to the weight — and bf16 has only ~3 decimal
#   digits of precision, so adding an update directly to a bf16 weight rounds to
#   *no change at all*. The fp32 copy is what makes small updates accumulate.
# - It keeps **two running averages per parameter**, `m` and `v` (4 bytes each),
#   which is the entire content of "adaptive" in Adam.
#
# 2 + 4 + 2 + 4 + 4 = 16. The model is one-eighth of its own training footprint.
#
# This immediately explains three things you will meet later:
#
# | technique | what it drops | notebook |
# |---|---|---|
# | **SGD instead of Adam** | both `m` and `v` — 8 bytes/param | — (worse convergence, rarely worth it) |
# | **LoRA** | trains ~1% of parameters, so optimizer state shrinks ~100× | 08 |
# | **QLoRA** | base weights to 4-bit *and* LoRA on top | 08 |
# | **8-bit Adam** | `m` and `v` to 1 byte each — 6 bytes/param saved | 08 |
#
# **Do the arithmetic before you launch, not after.** A quick worked example for
# your 24 GB card, full fine-tuning a 1.5B model:
#
# ```
# state:       1.5B × 16 bytes  = 24.0 GB   <- already over budget
# activations: (whatever they are) > 0
# ```
#
# Over before activations. With LoRA at rank 16, the same model:
#
# ```
# frozen base (bf16):  1.5B × 2      =  3.0 GB
# LoRA params + Adam:  ~10M × 16     =  0.16 GB
# activations:                       ~  2-4 GB
# total:                             ~  6 GB   <- comfortable
# ```
#
# Same model, same GPU, 4× headroom. That is the entire argument for notebook 08,
# and you can now derive it yourself rather than taking it on faith.
#
# Activations scale with `batch × seq_len × layers × d_model`, and are the part
# you control at runtime via batch size and gradient checkpointing.

# %%
def estimate_training_memory(
    n_params: float,
    *,
    mode: str = "full",
    bytes_per_param_override: float | None = None,
    batch: int = 8,
    seq_len: int = 1024,
    d_model: int = 768,
    n_layers: int = 12,
    grad_checkpointing: bool = False,
) -> dict:
    """Rough VRAM estimate in GiB. Optimistic by ~10-20%; leave headroom."""
    GB = 1024**3

    if bytes_per_param_override is not None:
        bpp = bytes_per_param_override
    elif mode == "full":
        bpp = 18.0        # bf16 weights + fp32 master + grads + Adam m/v
    elif mode == "lora":
        bpp = 2.0         # frozen bf16 base; adapter state is a rounding error
    elif mode == "qlora":
        bpp = 0.65        # 4-bit NF4 base + small fp16 overhead
    elif mode == "inference":
        bpp = 2.0         # bf16 weights only
    else:
        raise ValueError(f"unknown mode {mode!r}")

    state = n_params * bpp / GB

    # Activation estimate: the dominant term is the per-layer residual stream
    # plus attention/MLP intermediates. ~20 tensors of (batch, seq, d_model) per
    # layer in bf16 is a decent empirical rule for a non-fused implementation.
    act_per_layer = batch * seq_len * d_model * 2 * 20 / GB
    acts = act_per_layer * (1 if grad_checkpointing else n_layers)

    overhead = 0.8  # CUDA context, cuBLAS workspaces, allocator fragmentation
    return {
        "model_state_GiB": round(state, 2),
        "activations_GiB": round(acts, 2),
        "overhead_GiB": overhead,
        "total_GiB": round(state + acts + overhead, 2),
    }


VRAM = 24.0  # your RTX 5090 budget, in GiB

print(f"{'scenario':<46} {'state':>7} {'acts':>7} {'total':>7}   fits in 24 GiB?")
print("-" * 88)

scenarios = [
    ("124M pretrain (GPT-2 small), bs8 x 1024",
     dict(n_params=124e6, mode="full", batch=8, seq_len=1024, d_model=768, n_layers=12)),
    ("350M pretrain, bs8 x 1024",
     dict(n_params=350e6, mode="full", batch=8, seq_len=1024, d_model=1024, n_layers=24)),
    ("1.5B full finetune, bs2 x 1024",
     dict(n_params=1.5e9, mode="full", batch=2, seq_len=1024, d_model=1536, n_layers=28)),
    ("1.5B LoRA, bs4 x 1024",
     dict(n_params=1.5e9, mode="lora", batch=4, seq_len=1024, d_model=1536, n_layers=28)),
    ("7B QLoRA, bs2 x 2048, +ckpt",
     dict(n_params=7e9, mode="qlora", batch=2, seq_len=2048, d_model=4096,
          n_layers=32, grad_checkpointing=True)),
    ("7B full finetune, bs1 x 1024  (hopeless)",
     dict(n_params=7e9, mode="full", batch=1, seq_len=1024, d_model=4096, n_layers=32)),
]

for name, kw in scenarios:
    est = estimate_training_memory(**kw)
    verdict = "yes" if est["total_GiB"] < VRAM * 0.9 else "NO"
    print(
        f"{name:<46} {est['model_state_GiB']:>7.1f} {est['activations_GiB']:>7.1f} "
        f"{est['total_GiB']:>7.1f}   {verdict}"
    )

# %% [markdown]
# Read that table carefully — it is the map of what you can and cannot do on this
# machine, and it explains the shape of the whole course:
#
# - **Pretraining from scratch:** stick to **124M–350M**. That is genuinely a
#   real LLM, just a small one, and it trains in hours not weeks.
# - **Full finetuning:** fine up to ~1.5B.
# - **7B+:** QLoRA only. Never full finetune.
# - Activations, not weights, are what you tune with batch size. When you OOM,
#   halve the batch and double `grad_accum_steps` — the math is identical.

# %% [markdown]
# ## Step 4 — Make OOM debugging easier
#
# Two settings worth knowing about now:

# %%
import os

# Reduces allocator fragmentation, which is a common cause of "OOM with plenty
# of free memory". Must be set BEFORE torch initialises CUDA, so in real scripts
# put it at the very top of the file or export it in your shell.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Slower but gives you the real line number on a CUDA error instead of a
# stack trace pointing at some unrelated later op. Turn on only while debugging.
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def vram_report(tag: str = "") -> None:
    """Print current / peak VRAM. Call it around training steps to find the peak."""
    if not torch.cuda.is_available():
        print("(no cuda)")
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"{tag:<24} allocated {alloc:6.2f} GiB | reserved {reserved:6.2f} | peak {peak:6.2f}")


def reset_vram_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


if torch.cuda.is_available():
    reset_vram_peak()
    vram_report("baseline")
    big = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)  # 128 MiB
    vram_report("after 128 MiB alloc")
    del big
    torch.cuda.empty_cache()
    vram_report("after free")

# %% [markdown]
# ## Step 5 — Hugging Face account & cache
#
# You need a free HF account for datasets and models. Some (Llama, Gemma)
# require accepting a licence on the model page first.
#
# ```bash
# pip install -U "huggingface_hub[cli]"
# hf auth login          # paste a token from https://huggingface.co/settings/tokens
# ```
#
# **Move the cache off your C: drive if space is tight.** Models and datasets add
# up to hundreds of GB fast. In `~/.bashrc`:
#
# ```bash
# export HF_HOME=~/hf-cache
# ```
#
# Check what you're using with `hf cache scan`, and prune with `hf cache delete`.

# %%
try:
    from huggingface_hub import HfFolder, whoami

    token = HfFolder.get_token()
    if token:
        print(f"logged in to HF as: {whoami(token)['name']}")
    else:
        print("no HF token found — run `hf auth login` in your terminal")
except ImportError:
    print("huggingface_hub not installed yet — `pip install -r requirements.txt`")
except Exception as exc:
    print(f"HF check failed ({exc}). Not fatal; public datasets still work.")

print(f"\nHF_HOME = {os.environ.get('HF_HOME', '~/.cache/huggingface (default)')}")

# %% [markdown]
# ## Checkpoint — you're ready when
#
# - [ ] The bf16 matmul cell printed **OK**, not a `no kernel image` error.
# - [ ] `torch.cuda.get_arch_list()` contains `sm_120`.
# - [ ] You understand why 7B full-finetune is off the table and QLoRA isn't.
# - [ ] `hf auth login` succeeded.
#
# **Next:** `01_data_from_huggingface.ipynb` — where the data actually comes from,
# and why data quality dominates everything else you'll do.

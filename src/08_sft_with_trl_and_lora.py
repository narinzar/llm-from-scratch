# %% [markdown]
# # 08 — SFT the Production Way: TRL + LoRA/QLoRA
#
# **Goal:** fine-tune a genuinely capable model (0.5B–7B) on your 24 GB card
# using LoRA and QLoRA, with HuggingFace TRL — and understand the mechanism well
# enough to debug it.
#
# **Time:** 60–90 min including a real training run.
#
# ## Why we switch models here
#
# Notebook 07's 124M model can learn *format* but has no knowledge to draw on.
# To build something actually useful you fine-tune a strong base model. But full
# fine-tuning even a 7B model needs ~112 GB (recall notebook 00's table).
#
# **LoRA** and **QLoRA** are how that becomes possible on one consumer GPU.

# %%
import gc
import os

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
device = "cuda" if torch.cuda.is_available() else "cpu"


def free_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def vram(tag: str = "") -> None:
    if torch.cuda.is_available():
        print(f"{tag:<28} {torch.cuda.memory_allocated()/1024**3:6.2f} GiB allocated, "
              f"peak {torch.cuda.max_memory_allocated()/1024**3:6.2f}")


print(f"device: {device}")
if device == "cuda":
    print(f"gpu:    {torch.cuda.get_device_name()}")

# %% [markdown]
# ## Part 1 — LoRA, derived from scratch
#
# ### The observation
#
# Fine-tuning updates a weight matrix `W` by some `ΔW`. The LoRA paper's insight
# is that this `ΔW` has **low intrinsic rank** — the adaptation genuinely needed
# for a downstream task lives in a small subspace, even though `ΔW` is full-size.
#
# So don't store `ΔW` (d×d). Store two thin matrices whose product approximates
# it:
#
# ```
# W' = W + ΔW ≈ W + B·A
#
# where A is (r × d) and B is (d × r), with r << d
# ```
#
# Parameters drop from `d²` to `2·r·d`. At d=4096 and r=16 that's 16.7M → 131k,
# a **128× reduction**.
#
# `W` stays **frozen**. You only ever compute gradients for A and B — which is
# why the optimizer state (the dominant memory cost) nearly vanishes.

# %%
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """Wrap a frozen Linear with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False        # this is what saves the memory

        self.r = r
        # alpha/r scaling decouples the LR from the rank: change r without
        # having to re-tune the learning rate. Convention is alpha = 2r.
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        self.dropout = nn.Dropout(dropout)

        # A is randomly initialised, B is ZERO. So B@A = 0 at the start and the
        # model is EXACTLY the original at step 0. If both were random you'd
        # inject noise into a converged model and lose quality immediately.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + lora * self.scaling

    def merge(self) -> nn.Linear:
        """Fold B@A into the base weight — zero inference overhead afterwards."""
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None)
        merged.weight.data = self.base.weight.data + \
            (self.lora_B @ self.lora_A) * self.scaling
        if self.base.bias is not None:
            merged.bias.data = self.base.bias.data
        return merged


# Verify the zero-init property and the parameter savings.
base = nn.Linear(512, 512, bias=False)
lora = LoRALinear(base, r=8, alpha=16)

x = torch.randn(4, 512)
print(f"output identical at init: "
      f"{torch.allclose(base(x), lora(x), atol=1e-6)}   <- required")

n_full = base.weight.numel()
n_lora = lora.lora_A.numel() + lora.lora_B.numel()
print(f"\nfull fine-tune params: {n_full:>10,}")
print(f"LoRA params (r=8):     {n_lora:>10,}   ({n_full/n_lora:.0f}x fewer)")

trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in lora.parameters() if not p.requires_grad)
print(f"\ntrainable {trainable:,} | frozen {frozen:,} "
      f"({100*trainable/(trainable+frozen):.2f}% trainable)")

# %%
# Confirm merging is exact — this is what lets you ship a LoRA with no
# inference-time cost.
with torch.no_grad():
    lora.lora_B.copy_(torch.randn_like(lora.lora_B) * 0.02)   # simulate training

merged = lora.merge()
with torch.no_grad():
    print(f"merged vs unmerged max diff: {(lora(x) - merged(x)).abs().max().item():.2e}")
    print("(exact -> a merged LoRA is literally the same model, just one matrix)")

# %% [markdown]
# ### Choosing rank and target modules
#
# | rank | when |
# |---|---|
# | 4–8 | style/format adaptation, small datasets |
# | 16–32 | **the usual default** — most instruction tuning |
# | 64–128 | teaching genuinely new capability or a new domain |
# | 256+ | you probably want full fine-tuning instead |
#
# **Which modules?** The original paper attached LoRA to `q_proj` and `v_proj`
# only. Later work (and QLoRA's ablations) found that targeting **all linear
# layers** — attention *and* MLP — is consistently better and costs little.
# That's the modern default:
#
# ```python
# target_modules = ["q_proj","k_proj","v_proj","o_proj",
#                   "gate_proj","up_proj","down_proj"]
# ```

# %% [markdown]
# ## Part 2 — QLoRA: 4-bit base weights
#
# LoRA freezes the base but still stores it in bf16 — 14 GB for a 7B model.
# **QLoRA** quantizes the frozen base to **4 bits** (3.5 GB) and keeps the LoRA
# adapters in bf16. Three ingredients:
#
# 1. **NF4 (4-bit NormalFloat)** — a data type whose 16 levels are placed at the
#    quantiles of a normal distribution. Since neural network weights are
#    roughly Gaussian, this is information-theoretically better than uniform
#    int4 for the same bit count.
# 2. **Double quantization** — the quantization constants themselves get
#    quantized. Saves ~0.4 bits/param.
# 3. **Paged optimizers** — spill optimizer state to CPU RAM on memory spikes
#    instead of OOMing.
#
# Gradients still flow through the frozen 4-bit weights (dequantized on the fly
# to bf16 for each matmul) into the adapters.
#
# **The cost:** ~30% slower than LoRA, due to dequantization on every forward.
# **The benefit:** 7B models fit in ~6 GB instead of ~16 GB.

# %%
def memory_comparison(n_params_b: float) -> None:
    GB = 1
    print(f"\n{n_params_b}B parameter model:\n")
    print(f"{'method':<22}{'base':>9}{'grads':>9}{'optim':>9}{'total':>9}")
    print("-" * 58)
    n = n_params_b * 1e9
    rows = [
        ("full FT (bf16+Adam)", n * 2, n * 4, n * 12),
        ("LoRA r=32",           n * 2, 0.003 * n * 4, 0.003 * n * 12),
        ("QLoRA r=32 (NF4)",    n * 0.55, 0.003 * n * 4, 0.003 * n * 12),
    ]
    for name, b, g, o in rows:
        tot = (b + g + o) / 1024**3
        print(f"{name:<22}{b/1024**3:>8.1f}G{g/1024**3:>8.2f}G{o/1024**3:>8.2f}G"
              f"{tot:>8.1f}G")


for size in [1.5, 7, 14]:
    memory_comparison(size)
print("\n(activations add a few GB on top; 24 GiB is the budget)")

# %% [markdown]
# Read the 7B row: full fine-tuning wants ~117 GB, LoRA ~13 GB, QLoRA ~3.9 GB.
# That is the entire reason you can do this at home. Note also that LoRA does
# *not* reduce the base-weight memory at all — it only kills gradients and
# optimizer state. That's why 14B needs QLoRA even with LoRA adapters.

# %% [markdown]
# ## Part 3 — The real thing, with TRL
#
# Install (in your venv):
#
# ```bash
# pip install -U trl peft bitsandbytes accelerate datasets
# ```
#
# **A Blackwell note:** `bitsandbytes` needs a build with sm_120 kernels. If
# 4-bit loading throws a CUDA error on your 5090, either upgrade
# (`pip install -U bitsandbytes`) or fall back to plain LoRA in bf16 — a
# 1.5B–3B model in bf16 fits fine on 24 GB without any quantization.

# %%
MODEL_ID = "Qwen/Qwen2.5-0.5B"     # start here; it trains fast and it's genuinely decent
# MODEL_ID = "Qwen/Qwen2.5-1.5B"   # better quality, still comfortable
# MODEL_ID = "Qwen/Qwen2.5-7B"     # needs QLoRA on 24 GB

USE_QLORA = False   # True -> 4-bit base. Needed for 7B; unnecessary for 0.5B.

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"model: {MODEL_ID}")
print(f"tokenizer vocab: {len(tokenizer):,}")
print(f"\nthis model's built-in chat template:\n")
print(tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 2+2?"},
     {"role": "assistant", "content": "4"}],
    tokenize=False,
))

# %% [markdown]
# **Use the model's own chat template.** Every model family has its own format
# (Qwen uses ChatML, Llama 3 uses `<|start_header_id|>`, Gemma uses `<start_of_turn>`).
# Using the wrong one is a silent quality killer — the model was pretrained to
# recognize *its* markers, and yours will look like noise. `apply_chat_template`
# reads the correct format from the tokenizer config. Never hand-roll it.

# %%
free_vram()

model_kwargs = dict(dtype=torch.bfloat16, device_map={"": 0} if device == "cuda" else "cpu")

if USE_QLORA:
    from transformers import BitsAndBytesConfig

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",              # NOT "fp4" — nf4 is better
        bnb_4bit_compute_dtype=torch.bfloat16,  # matmuls run in bf16
        bnb_4bit_use_double_quant=True,
    )

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
model.config.use_cache = False   # incompatible with gradient checkpointing
vram("base model loaded")
print(f"parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

# %%
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

if USE_QLORA:
    # Casts norms to fp32 and enables gradient checkpointing — required for
    # stable 4-bit training.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,          # = 2r, the usual convention
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
vram("after LoRA attach")

# %% [markdown]
# That printout is the whole point: typically **<1% trainable**. The 99% is
# frozen, so no gradients and no Adam state for it.

# %%
from datasets import load_dataset

# smol-smoltalk is filtered for small models: shorter turns, no advanced math.
train_ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train[:8000]")
eval_ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="test[:400]")

print(f"train {len(train_ds)}  eval {len(eval_ds)}")
print(f"\nexample:")
for m in train_ds[0]["messages"][:4]:
    print(f"  {m['role']:>10}: {m['content'][:150]}")

# %% [markdown]
# ### TRL's SFTTrainer
#
# It handles what you implemented by hand in notebook 07: applying the chat
# template, masking the prompt (`assistant_only_loss`), packing, collation.
#
# The settings below are tuned for 24 GB. The key relationship, as always:
# `effective_batch = per_device_batch × grad_accum`.

# %%
from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    output_dir="../artifacts/qwen-sft-lora",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,        # effective batch 32
    learning_rate=2e-4,                   # LoRA tolerates ~10x higher LR than full FT
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_length=1024,
    packing=True,                         # ~100% token efficiency
    assistant_only_loss=True,             # the -100 masking from notebook 07
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    report_to="none",                     # set "wandb" if you want dashboards
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)

print(f"steps: {len(trainer.get_train_dataloader())}")

# %% [markdown]
# **Why LoRA's LR is ~10× higher than full fine-tuning's:** the adapters start
# at zero and must travel a real distance to matter, and there are far fewer of
# them so each carries more responsibility. `1e-4` to `3e-4` is the standard
# range; `2e-5` (a full-FT rate) would barely move them.

# %%
# Uncomment to train. On a 5090, 0.5B + LoRA on 8k examples is roughly 20-40 min.
#
# result = trainer.train()
# trainer.save_model("../artifacts/qwen-sft-lora/final")
# print(result.metrics)
# vram("after training")

# %% [markdown]
# ## Part 4 — Inference and merging

# %%
def build_chat_pipeline(adapter_path: str | None = None, merge: bool = False):
    """Load the base model, optionally attach (or merge) a LoRA adapter."""
    from peft import PeftModel

    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16,
        device_map={"": 0} if device == "cuda" else "cpu",
    )
    if adapter_path:
        m = PeftModel.from_pretrained(m, adapter_path)
        if merge:
            # Folds B@A into the base weights. The result is a normal
            # transformer with no PEFT dependency and no inference overhead.
            m = m.merge_and_unload()
            print("adapter merged into base weights")
    m.eval()
    return m


@torch.no_grad()
def ask(m, question: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(m.device)
    out = m.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Slice off the prompt so you only see the completion.
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# %% [markdown]
# ### Compare base vs fine-tuned — always do this
#
# The honest test of whether SFT helped. Run the same prompts through both.

# %%
QUESTIONS = [
    "What is the capital of France?",
    "Explain photosynthesis in two sentences.",
    "Write a haiku about debugging.",
    "What is 17 * 23?",
]

# base = build_chat_pipeline()
# for q in QUESTIONS:
#     print(f"\n=== {q} ===")
#     print(f"[base]      {ask(base, q)}")
# del base; free_vram()
#
# tuned = build_chat_pipeline("../artifacts/qwen-sft-lora/final")
# for q in QUESTIONS:
#     print(f"\n=== {q} ===")
#     print(f"[finetuned] {ask(tuned, q)}")

# %% [markdown]
# **Note that Qwen2.5-0.5B *base* will already look somewhat instruction-capable.**
# Modern "base" models see instruction-formatted data during pretraining, so the
# line is blurry. Your SFT should still improve consistency, formatting, and
# adherence to the chat template — but don't expect night-and-day, and don't
# fool yourself that it's bigger than it is.

# %% [markdown]
# ## Part 5 — LoRA variants worth knowing
#
# | variant | idea | when to use |
# |---|---|---|
# | **DoRA** | decompose into magnitude + direction, adapt separately | small quality gain over LoRA at low rank; `use_dora=True` |
# | **rsLoRA** | scale by `alpha/sqrt(r)` instead of `alpha/r` | makes high ranks (64+) actually work; `use_rslora=True` |
# | **LoRA+** | different LR for A and B (B higher) | faster convergence |
# | **PiSSA** | init A,B from the SVD of W instead of random/zero | faster early progress |
#
# All are one flag in `LoraConfig`. Start plain; reach for these if you plateau.
#
# ## When does LoRA lose to full fine-tuning?
#
# Be honest about this — LoRA is not free:
#
# - **Learning genuinely new knowledge** (a new language, a new domain
#   vocabulary). LoRA adapts behaviour well and absorbs facts poorly.
# - **Very large SFT datasets** (>100k examples), where the low-rank constraint
#   becomes a real bottleneck.
# - **When you have the VRAM anyway.** For a 0.5B–1.5B model on 24 GB, just full
#   fine-tune; it's simpler and slightly better.
#
# LoRA's sweet spot is exactly the case you're in: **a big model you couldn't
# otherwise touch, and a task about behaviour rather than knowledge.**
#
# ## Troubleshooting
#
# | symptom | fix |
# |---|---|
# | OOM immediately | lower `per_device_train_batch_size`, raise `grad_accum` |
# | OOM mid-run | a long sequence — lower `max_length` |
# | loss ~0 from step 1 | template mismatch; check `assistant_only_loss` masked correctly |
# | loss doesn't move | LR too low for LoRA — go to 2e-4 |
# | model outputs gibberish after FT | LR far too high, or the wrong chat template |
# | bitsandbytes CUDA error on 5090 | upgrade bnb, or set `USE_QLORA=False` |
#
# ## Exercises
#
# 1. **Rank sweep.** r ∈ {4, 16, 64} at fixed steps. Plot eval loss vs trainable
#    params. Where do the returns stop?
# 2. **Target modules.** Attention-only vs all-linear at the same rank.
# 3. **Merge and measure.** Time generation with an attached adapter vs merged.
# 4. **QLoRA a 7B.** Set `MODEL_ID` to Qwen2.5-7B, `USE_QLORA=True`, batch 1,
#    accum 16. Watch it fit in 24 GB.
#
# **Next:** `09_reward_modeling.ipynb` — SFT teaches format. Now teach quality.

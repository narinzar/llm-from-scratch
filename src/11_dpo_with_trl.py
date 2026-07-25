# %% [markdown]
# # 11 — Preference Tuning with TRL
#
# **Goal:** run DPO on a real model with TRL, read the diagnostics correctly,
# and know when to reach for a variant.
#
# **Time:** 30 min setup + 30–60 min training.
#
# This notebook is short by design. You built DPO in notebook 10; here you learn
# the production controls and — more importantly — **how to tell whether it
# worked.**

# %%
import gc
import os

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
device = "cuda" if torch.cuda.is_available() else "cpu"


def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print(f"device: {device}")

# %% [markdown]
# ## The pipeline position
#
# DPO fine-tunes an **already SFT'd model**. Running DPO on a base model does
# not work well — preference pairs assume the model can already produce
# coherent responses in the right format.
#
# Ideally you use *your own* SFT model from notebook 08. If you skipped it, an
# off-the-shelf instruct model works fine for learning the mechanics.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

# Option A: your own SFT'd model from notebook 08
# MODEL_ID = "../artifacts/qwen-sft-lora/final"

# Option B: an off-the-shelf instruct model
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16,
    device_map={"": 0} if device == "cuda" else "cpu",
)
model.config.use_cache = False
print(f"loaded {MODEL_ID}: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

# %% [markdown]
# ## The reference model, and the LoRA trick
#
# DPO needs `π_ref`. Three options, in increasing order of cleverness:
#
# | approach | memory | notes |
# |---|---|---|
# | pass `ref_model=<second copy>` | 2× weights | simple, wasteful |
# | `precompute_ref_log_probs=True` | ~1× | ref never changes, so cache its outputs |
# | **`peft_config=...` and no `ref_model`** | **1×** | **disable adapters to recover the reference** |
#
# The third is the one to use. With LoRA, the base weights *are* the reference —
# TRL just turns the adapters off for the reference forward pass. Free.

# %%
from datasets import load_dataset

# TRL expects columns: prompt, chosen, rejected
train_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:4000]")
eval_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs[:400]")

print(f"columns: {train_ds.column_names}")
print(f"train {len(train_ds)}  eval {len(eval_ds)}")

ex = train_ds[0]
print(f"\nPROMPT:   {ex['prompt'][:150]}")
print(f"CHOSEN:   {ex['chosen'][-1]['content'][:200]}")
print(f"REJECTED: {ex['rejected'][-1]['content'][:200]}")

# %% [markdown]
# ### Filter by length before you train
#
# Truncation in DPO is worse than in SFT. If a long chosen response gets cut but
# the short rejected one doesn't, you're comparing a *truncated* answer against a
# complete one — and teaching the model something that isn't in your data.

# %%
def total_len(row) -> int:
    text = row["prompt"] + row["chosen"][-1]["content"] + row["rejected"][-1]["content"]
    return len(tokenizer(text)["input_ids"])


MAX_LEN = 1024
before = len(train_ds)
train_ds = train_ds.filter(lambda r: total_len(r) < MAX_LEN)
print(f"kept {len(train_ds)}/{before} pairs under {MAX_LEN} tokens "
      f"({100*len(train_ds)/before:.0f}%)")

# %%
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

dpo_config = DPOConfig(
    output_dir="../artifacts/qwen-dpo",
    beta=0.1,                     # the KL strength from notebook 10
    learning_rate=5e-6,           # higher than 5e-7 because LoRA adapters need it
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_length=MAX_LEN,
    max_prompt_length=512,        # prompt truncated from the LEFT, keeping the end
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    report_to="none",
    seed=42,
    # loss_type="sigmoid",        # "ipo" | "kto_pair" | "robust" are alternatives
    # rpo_alpha=1.0,              # adds an SFT term on chosen — anti-degeneration
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,               # None + peft_config => adapter-disable trick
    args=dpo_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    peft_config=peft_config,
)

print(f"steps: {len(trainer.get_train_dataloader())}")

# %%
# Uncomment to train. ~30-60 min for 4k pairs on a 5090.
#
# result = trainer.train()
# trainer.save_model("../artifacts/qwen-dpo/final")

# %% [markdown]
# ## Reading the metrics — the actual skill
#
# TRL logs these. **Most people only watch `loss`, which is the least useful.**
#
# | metric | healthy | what it means |
# |---|---|---|
# | `rewards/accuracies` | rises to 0.65–0.8 | fraction of pairs ordered correctly |
# | `rewards/margins` | grows steadily | mean implicit-reward gap |
# | **`rewards/chosen`** | **slightly negative, stable** | **the degeneration alarm** |
# | `rewards/rejected` | clearly negative | expected — you're suppressing these |
# | `logps/chosen` | roughly flat | absolute log-prob of good responses |
#
# ### The one diagnostic that matters
#
# From notebook 10: if `rewards/chosen` trends **strongly negative** while
# margins grow, you are degenerating. The model is suppressing everything and
# only relatively preferring the chosen response.
#
# ```
# healthy:      chosen  -0.2   rejected  -1.8   margin 1.6
# degenerating: chosen  -4.5   rejected  -9.0   margin 4.5   <- margin looks GREAT
# ```
#
# Both have a fine-looking margin. The second model is worse than when you
# started. Fixes, in order: **lower the LR**, raise `beta`, or set `rpo_alpha`
# to add an SFT anchor on the chosen response.

# %%
def diagnose(chosen: float, rejected: float, accuracy: float) -> str:
    """Interpret a DPO metric snapshot."""
    margin = chosen - rejected
    if chosen < -2.0:
        return f"DEGENERATING (margin {margin:.1f} but chosen={chosen:.1f}) — lower LR"
    if accuracy < 0.55:
        return f"NOT LEARNING (acc {accuracy:.2f}) — raise LR, or check data"
    if margin < 0.1:
        return f"BARELY MOVING (margin {margin:.2f}) — raise LR or train longer"
    return f"HEALTHY (margin {margin:.2f}, acc {accuracy:.2f}, chosen {chosen:.2f})"


print("interpreting metric snapshots:\n")
for c, r, a in [(-0.2, -1.8, 0.72), (-4.5, -9.0, 0.85), (-0.01, -0.02, 0.51), (-0.05, -0.9, 0.68)]:
    print(f"  chosen={c:>6.2f} rejected={r:>6.2f} acc={a:.2f}  ->  {diagnose(c, r, a)}")

# %% [markdown]
# ## Evaluating: did it actually get better?
#
# Preference metrics measure whether you fit the preference data. They do **not**
# tell you the model improved. Use two independent checks:
#
# 1. **Win rate against the pre-DPO model**, judged by a stronger model
#    (notebook 14 covers LLM-as-judge and its biases).
# 2. **A capability benchmark** (MMLU, GSM8K) to confirm you didn't degrade
#    general ability while chasing style.
#
# **Always control for length.** DPO reliably makes responses longer, and both
# human and LLM judges prefer longer answers. An uncontrolled win rate mostly
# measures verbosity.

# %%
@torch.no_grad()
def generate(m, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inp = tokenizer(text, return_tensors="pt").to(m.device)
    out = m.generate(**inp, max_new_tokens=max_new_tokens, temperature=temperature,
                     do_sample=temperature > 0, top_p=0.9,
                     pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def compare_lengths(model_a, model_b, prompts: list[str]) -> None:
    """Length is the confound. Measure it explicitly."""
    la, lb = [], []
    for p in prompts:
        la.append(len(generate(model_a, p).split()))
        lb.append(len(generate(model_b, p).split()))
    print(f"pre-DPO  mean length: {sum(la)/len(la):.0f} words")
    print(f"post-DPO mean length: {sum(lb)/len(lb):.0f} words")
    print(f"ratio: {(sum(lb)/len(lb))/(sum(la)/len(la)):.2f}x")
    print("\nIf that ratio is >1.3, any win-rate improvement is suspect —")
    print("re-measure with length-controlled comparisons.")


EVAL_PROMPTS = [
    "Explain the difference between TCP and UDP.",
    "Write a Python function to reverse a linked list.",
    "What causes the seasons on Earth?",
    "Summarize the plot of Hamlet in three sentences.",
]

# Uncomment after training:
# from peft import PeftModel
# base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"":0})
# tuned = PeftModel.from_pretrained(
#     AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"":0}),
#     "../artifacts/qwen-dpo/final").merge_and_unload()
# compare_lengths(base, tuned, EVAL_PROMPTS)

# %% [markdown]
# ## Choosing a method
#
# ```
# Do you have preference PAIRS?
#  ├── no, just good/bad labels          -> KTO
#  └── yes
#       ├── want to skip SFT entirely?   -> ORPO (one stage, no reference model)
#       ├── preferences are noisy?       -> cDPO (label_smoothing=0.1) or "robust"
#       ├── responses vary a lot in length? -> SimPO (length-normalized)
#       ├── chosen logprobs collapsing?  -> DPO + rpo_alpha
#       └── otherwise                    -> plain DPO, beta=0.1
# ```
#
# In TRL most of these are a `loss_type` string on `DPOConfig`, or a sibling
# trainer (`KTOTrainer`, `ORPOTrainer`). Start with plain DPO.
#
# ## Troubleshooting
#
# | symptom | fix |
# |---|---|
# | `rewards/accuracies` stuck at 0.5 | LR too low, or prompt/chosen/rejected columns mismatched |
# | `rewards/chosen` diving | LR too high — this is degeneration |
# | OOM | DPO holds 2 forward passes; halve batch, raise accum |
# | eval loss up, train loss down | overfitting — 1 epoch is usually enough |
# | outputs got much longer | expected; control for it when evaluating |
#
# ## Exercises
#
# 1. **Beta sweep.** β ∈ {0.05, 0.1, 0.5}. Plot `rewards/chosen` for each and
#    find where degeneration begins.
# 2. **`rpo_alpha`.** Rerun the LR that degenerated, with `rpo_alpha=1.0`. Does
#    the SFT anchor hold the chosen log-probs up?
# 3. **DPO vs ORPO.** Run ORPO directly on the base model and compare against
#    SFT→DPO for the same total compute.
# 4. **Length-controlled win rate.** Compare only response pairs within 10% of
#    each other in length. How much of your win rate survives?
#
# **Next:** `12_grpo_rlvr_from_scratch.ipynb` — online RL with verifiable
# rewards, the technique behind reasoning models.

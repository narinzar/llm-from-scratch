# %% [markdown]
# # 13 — GRPO on GSM8K with TRL
#
# **Goal:** run a real RLVR training loop — teach a small model to do grade-school
# math better than it could before — and measure the improvement honestly.
#
# **Time:** 30 min setup, 2–5 h training. This is the most expensive notebook.
#
# ## What to expect
#
# On a 5090 with Qwen2.5-0.5B-Instruct and a few hours of GRPO on GSM8K, a
# realistic outcome is **+5 to +15 percentage points** of accuracy. You will not
# reproduce DeepSeek-R1. What you *will* see is the mechanism working: reward
# rising, completion length growing, and reasoning structure emerging from
# nothing but a correctness signal.
#
# **Start from an instruct model, not a base model.** GRPO needs the policy to
# occasionally produce correct answers — that's the only signal it gets. A base
# model that's correct 0% of the time yields all-zero advantages forever
# (notebook 12).

# %%
import os
import re

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# %%
from datasets import load_dataset

gsm8k = load_dataset("openai/gsm8k", "main")
print(gsm8k)

ex = gsm8k["train"][0]
print(f"\nQUESTION:\n{ex['question']}")
print(f"\nANSWER:\n{ex['answer']}")

# %% [markdown]
# Note GSM8K's answer format: worked reasoning, then `#### <number>`. That
# `####` marker is what makes it machine-verifiable, and it's why GSM8K became
# the standard RLVR testbed.

# %%
SYSTEM_PROMPT = """Respond in the following format:
<think>
Work through the problem step by step here.
</think>
<answer>
The final numeric answer only.
</answer>"""


def extract_gsm8k_answer(text: str) -> str | None:
    """Ground truth: everything after ####."""
    if "####" not in text:
        return None
    return text.split("####")[-1].strip().replace(",", "")


def extract_model_answer(text: str) -> str | None:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        nums = re.findall(r"-?\d+\.?\d*", candidate.replace(",", ""))
        return nums[-1] if nums else None
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


def to_prompt(row: dict) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ],
        "ground_truth": extract_gsm8k_answer(row["answer"]),
    }


train_ds = gsm8k["train"].map(to_prompt, remove_columns=gsm8k["train"].column_names)
test_ds = gsm8k["test"].map(to_prompt, remove_columns=gsm8k["test"].column_names)

print(f"train {len(train_ds)}  test {len(test_ds)}")
print(f"\nexample ground_truth: {train_ds[0]['ground_truth']}")

# %% [markdown]
# ## Reward functions, TRL-style
#
# TRL passes `completions` (a list) plus any extra dataset columns as kwargs.
# Each function returns a list of floats, one per completion. Multiple reward
# functions are **summed**.

# %%
def correctness_reward_func(completions, ground_truth, **kwargs) -> list[float]:
    """The verifiable signal. Worth the most."""
    responses = [c[0]["content"] for c in completions]
    out = []
    for r, gt in zip(responses, ground_truth):
        pred = extract_model_answer(r)
        if pred is None or gt is None:
            out.append(0.0)
            continue
        try:
            out.append(2.0 if abs(float(pred) - float(gt)) < 1e-4 else 0.0)
        except ValueError:
            out.append(0.0)
    return out


def format_reward_func(completions, **kwargs) -> list[float]:
    """Small, so it can't dominate correctness."""
    responses = [c[0]["content"] for c in completions]
    scores = []
    for r in responses:
        s = 0.0
        if re.search(r"<think>.*?</think>", r, re.DOTALL):
            s += 0.25
        if re.search(r"<answer>.*?</answer>", r, re.DOTALL):
            s += 0.25
        scores.append(s)
    return scores


def integer_reward_func(completions, **kwargs) -> list[float]:
    """GSM8K answers are always integers. Nudge toward well-formed output."""
    responses = [c[0]["content"] for c in completions]
    out = []
    for r in responses:
        a = extract_model_answer(r)
        out.append(0.25 if a is not None and a.lstrip("-").replace(".", "").isdigit() else 0.0)
    return out


# Verify the reward functions on synthetic completions before spending GPU hours.
# TRL hands each completion over as a list of message dicts, hence the nesting.
fake = [
    [{"content": "<think>2+2</think><answer>4</answer>"}],   # correct + formatted
    [{"content": "<answer>5</answer>"}],                     # wrong, half format
    [{"content": "the answer is 4"}],                        # correct, no format
    [{"content": "I am not sure"}],                          # nothing
]
gts = ["4", "4", "4", "4"]

print(f"{'completion':<44}{'correct':>9}{'format':>8}{'int':>6}{'total':>8}")
print("-" * 75)
c = correctness_reward_func(fake, gts)
f = format_reward_func(fake)
i = integer_reward_func(fake)
for k, comp in enumerate(fake):
    print(f"{comp[0]['content'][:42]:<44}{c[k]:>9.2f}{f[k]:>8.2f}{i[k]:>6.2f}"
          f"{c[k]+f[k]+i[k]:>8.2f}")

# %% [markdown]
# **Test your reward functions like this before every run.** A reward bug means
# hours of GPU time optimizing the wrong thing, and it is not obvious from the
# loss curve.

# %% [markdown]
# ## Measure the baseline FIRST
#
# You cannot claim improvement without a before number. Do this before training.
#
# This sounds obvious and is skipped constantly. The reason it gets skipped is
# psychological: after training you have a number, it looks good, and finding a
# baseline feels like homework. So people compare against a *remembered* figure,
# or a published one measured with a different prompt, and conclude they gained
# ten points they never gained.
#
# **The baseline must be measured by this code, on this split, with this prompt
# template, on this machine.** Every one of those changes the number:
#
# | change | effect on GSM8K accuracy |
# |---|---|
# | different prompt template | several points, easily |
# | `temperature=0` vs `0.7` | several points, and adds variance |
# | 200 examples vs the full 1,319 | ±4 points of noise either way |
# | different answer-extraction regex | several points — a correct answer scored wrong |
#
# That last one is the sneakiest. Half of apparent "gains" on maths benchmarks
# are the extractor getting better at *parsing* the model, not the model getting
# better at *maths*. Since we use the same extractor before and after, that
# cancels out — but only because we measured both ends the same way.
#
# **What to expect for Qwen2.5-0.5B-Instruct on GSM8K:** roughly **30–40%**
# greedy pass@1. If your baseline comes back near 0%, do not start training —
# something is wrong with the extractor or the chat template, and you would be
# "improving" a broken measurement. If it comes back above 60%, be suspicious of
# contamination or an over-permissive extractor.
#
# Note also that we evaluate at `temperature=0.0`. Training samples at high
# temperature (that is where exploration comes from), but *evaluation* must be
# deterministic, or you cannot separate a real gain from a lucky sample.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


@torch.no_grad()
def evaluate_gsm8k(model, n: int = 200, max_new_tokens: int = 400,
                   temperature: float = 0.0, batch_size: int = 16) -> dict:
    """Greedy pass@1 accuracy on the GSM8K test split."""
    model.eval()
    correct = lengths = 0
    subset = test_ds.select(range(min(n, len(test_ds))))

    for start in range(0, len(subset), batch_size):
        rows = subset.select(range(start, min(start + batch_size, len(subset))))
        texts = [tokenizer.apply_chat_template(r, tokenize=False, add_generation_prompt=True)
                 for r in rows["prompt"]]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        padding_side="left").to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=temperature > 0,
                             temperature=temperature if temperature > 0 else None,
                             pad_token_id=tokenizer.pad_token_id)
        for k in range(len(rows)):
            gen = out[k][enc["input_ids"].shape[1]:]
            resp = tokenizer.decode(gen, skip_special_tokens=True)
            lengths += len(gen)
            pred = extract_model_answer(resp)
            gt = rows["ground_truth"][k]
            try:
                if pred is not None and gt is not None and abs(float(pred) - float(gt)) < 1e-4:
                    correct += 1
            except ValueError:
                pass

    return {"accuracy": correct / len(subset),
            "n": len(subset),
            "mean_tokens": lengths / len(subset)}


# %%
# baseline_model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID, dtype=torch.bfloat16, device_map={"": 0})
# baseline = evaluate_gsm8k(baseline_model, n=200)
# print(f"BASELINE: {baseline['accuracy']:.1%} on {baseline['n']} problems, "
#       f"{baseline['mean_tokens']:.0f} tokens/response")
# del baseline_model
# torch.cuda.empty_cache()

# %% [markdown]
# ## Training with GRPOTrainer
#
# ### vLLM: the difference between 5 hours and 20
#
# GRPO spends most of its time generating. HF `generate()` is slow; vLLM is
# 5–10× faster thanks to paged attention and continuous batching.
#
# ```bash
# pip install vllm
# # in a SECOND terminal, keep this running:
# trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
# ```
#
# Then set `use_vllm=True`. It's the single highest-leverage setting here. If
# vLLM won't install for Blackwell yet, the run still works — just slower.

# %%
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

grpo_config = GRPOConfig(
    output_dir="../artifacts/qwen-grpo-gsm8k",

    # --- the GRPO-specific knobs ---
    num_generations=8,            # G — the group size. Must divide the batch.
    max_completion_length=400,
    max_prompt_length=350,
    temperature=1.0,              # MUST be > 0 or every advantage is zero
    top_p=1.0,                    # don't truncate; it biases importance ratios
    beta=0.04,                    # KL to the reference
    epsilon=0.2,                  # clip range
    loss_type="bnpo",             # "grpo" | "bnpo" | "dr_grpo" (no std division)
    scale_rewards=True,

    # --- ordinary training knobs ---
    learning_rate=1e-6,           # RL is far more fragile than SFT — keep it tiny
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=10,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    max_steps=500,
    bf16=True,
    gradient_checkpointing=True,

    # --- generation backend ---
    use_vllm=False,               # True once `trl vllm-serve` is running
    # vllm_mode="server",
    # vllm_server_port=8000,

    logging_steps=1,              # RL is noisy — log every step
    save_steps=100,
    report_to="none",
    seed=42,
)

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# %%
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID, dtype=torch.bfloat16, device_map={"": 0})
# model.config.use_cache = False
#
# trainer = GRPOTrainer(
#     model=model,
#     reward_funcs=[correctness_reward_func, format_reward_func, integer_reward_func],
#     args=grpo_config,
#     train_dataset=train_ds,
#     processing_class=tokenizer,
#     peft_config=peft_config,
# )
# trainer.train()
# trainer.save_model("../artifacts/qwen-grpo-gsm8k/final")

# %% [markdown]
# ## Reading the run
#
# TRL logs these. The interesting ones are not the loss.
#
# | metric | healthy | what it tells you |
# |---|---|---|
# | `reward` | rises slowly, noisily | the actual objective |
# | `rewards/correctness_reward_func` | **the one that matters** | real capability |
# | `rewards/format_reward_func` | saturates early | format is learned in ~50 steps |
# | `reward_std` | **> 0** | zero means all groups degenerate — no learning |
# | `completions/mean_length` | grows | reasoning emerging |
# | `kl` | small, stable | policy staying near the reference |
# | `clip_ratio` | 0.05–0.2 | >0.4 means steps too large |
#
# ### The failure you must be able to spot
#
# **`reward` rises but `rewards/correctness_reward_func` is flat.** The model
# learned to collect the format and integer rewards while getting no more
# answers right. It looks like progress and isn't.
#
# This is exactly why you separate reward components in your logging rather than
# only tracking the sum.

# %%
def diagnose_grpo(reward: float, correctness: float, fmt: float,
                  reward_std: float, kl: float) -> str:
    if reward_std < 1e-3:
        return "DEAD: all groups uniform. Check temperature>0 and problem difficulty."
    if kl > 0.5:
        return "DIVERGING: KL too high. Lower LR or raise beta."
    if correctness < 0.05 and fmt > 0.4:
        return "FORMAT-ONLY HACK: reward rising from format, not correctness."
    if correctness > 0:
        return f"HEALTHY: correctness {correctness:.2f}, format {fmt:.2f}, KL {kl:.3f}"
    return "NO SIGNAL YET: keep going, or problems may be too hard."


print("interpreting GRPO metric snapshots:\n")
for r, c, f, s, k in [
    (0.55, 0.10, 0.45, 0.30, 0.01),
    (0.50, 0.00, 0.50, 0.25, 0.02),
    (0.50, 0.00, 0.50, 0.00, 0.00),
    (1.20, 0.70, 0.50, 0.40, 0.80),
]:
    print(f"  reward={r:.2f} correct={c:.2f} fmt={f:.2f} std={s:.2f} kl={k:.2f}")
    print(f"    -> {diagnose_grpo(r, c, f, s, k)}\n")

# %% [markdown]
# ## Measure the improvement honestly
#
# ```python
# from peft import PeftModel
# tuned = PeftModel.from_pretrained(
#     AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"":0}),
#     "../artifacts/qwen-grpo-gsm8k/final").merge_and_unload()
#
# after = evaluate_gsm8k(tuned, n=200)
# print(f"before {baseline['accuracy']:.1%} -> after {after['accuracy']:.1%}")
# print(f"length {baseline['mean_tokens']:.0f} -> {after['mean_tokens']:.0f} tokens")
# ```
#
# **Three honesty checks before you believe the number:**
#
# 1. **Same decoding settings** for both. Greedy vs sampled is not a fair fight.
# 2. **Test set only.** GRPO trained on `train`; evaluating there measures
#    memorization.
# 3. **Enough samples.** At n=200, the standard error is ~3.5 points. A "+2%"
#    improvement is noise. Use n≥500 for anything you'd report, or run multiple
#    seeds.

# %%
def improvement_is_significant(acc_before: float, acc_after: float, n: int) -> str:
    """Two-proportion z-test — is this difference real or noise?"""
    p_pool = (acc_before + acc_after) / 2
    se = (2 * p_pool * (1 - p_pool) / n) ** 0.5
    if se == 0:
        return "degenerate"
    z = (acc_after - acc_before) / se
    verdict = "SIGNIFICANT (p<0.05)" if abs(z) > 1.96 else "NOT significant — could be noise"
    return f"delta {100*(acc_after-acc_before):+.1f}pp, z={z:.2f} -> {verdict}"


print(f"{'n':>6}{'before':>9}{'after':>8}   verdict")
print("-" * 66)
for n, b, a in [(200, 0.30, 0.32), (200, 0.30, 0.42), (1000, 0.30, 0.34), (1319, 0.30, 0.38)]:
    print(f"{n:>6}{b:>9.0%}{a:>8.0%}   {improvement_is_significant(b, a, n)}")

# %% [markdown]
# Note row 1: a +2pp gain at n=200 is **not** distinguishable from noise. That's
# the kind of result that gets reported as a win all the time. Don't do it.
#
# ## Troubleshooting
#
# | symptom | cause | fix |
# |---|---|---|
# | `reward_std` = 0 | greedy sampling, or all-too-hard problems | temperature ≥ 0.7; easier subset |
# | reward flat at 0 | model never gets one right | use an instruct model; easier problems |
# | reward up, accuracy flat | format-reward hacking | lower format weight |
# | OOM during generation | G × batch × length is large | lower `num_generations` or `max_completion_length` |
# | extremely slow | HF generate | install vLLM and `use_vllm=True` |
# | KL explodes | LR too high | drop LR 10×; raise beta |
# | outputs become gibberish | policy escaped the reference | raise beta, restart from checkpoint |
#
# ## Exercises
#
# 1. **Ablate the format reward.** Train with correctness only. Does reasoning
#    structure still emerge?
# 2. **Difficulty curriculum.** Filter to problems the baseline solves 20–80% of
#    the time (8 samples each). Compare learning speed against the full set.
# 3. **Group size.** G ∈ {4, 8, 16} at matched wall-clock. Which wins?
# 4. **Dr. GRPO.** Set `loss_type="dr_grpo"` and compare — it removes the std
#    normalization that biases toward low-variance prompts.
# 5. **Cross-task transfer.** Evaluate your GSM8K-trained model on MATH. Did it
#    learn math, or GSM8K?
#
# **Next:** `14_evaluation.ipynb` — how to know whether any of this worked.

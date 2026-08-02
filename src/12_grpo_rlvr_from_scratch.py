# %% [markdown]
# # 12 — GRPO & RLVR from Scratch
#
# **Goal:** implement Group Relative Policy Optimization end to end and
# understand why "RL with verifiable rewards" produced the reasoning-model wave.
#
# **Time:** 60–75 min.
#
# ## In plain language
#
# **What you're doing:** letting the model teach *itself*, by trying a problem
# several times and learning from which attempts turned out to be right.
#
# **The everyday version.** A child learning arithmetic. You don't hand them the
# perfect method — you give them a problem, let them try, and tell them whether
# the answer was right. They try again. Over time they work out an approach that
# gets right answers, and *nobody ever described that approach to them*.
#
# The essential ingredient: **you can check the answer.** 7 × 8 is 56 or it
# isn't. No opinion, no judge, no taste.
#
# **Why this is different from everything before it.** Notebooks 09–11 learned
# from data humans produced. Here the model generates its **own** attempts and
# learns from which ones worked. It's no longer limited by what's in the dataset
# — it can discover approaches nobody wrote down.
#
# **That's not an overstatement.** This is the technique behind reasoning models
# — o1, R1, and the rest. Nobody taught those models to "think step by step and
# check their work." They discovered it, because longer careful reasoning
# produced more right answers, and right answers scored higher.
#
# **How it works in one paragraph.** Give the model a question. Have it answer
# **eight times** (it'll produce different attempts — that's the point). Check
# which are correct. Now the clever bit: score each attempt *relative to the
# others in its group*. Better than its siblings → make it more likely. Worse →
# less likely. No separate judge model needed, because the group is the
# yardstick. That's the "Group Relative" in GRPO.
#
# **What you'll have at the end:** the whole algorithm implemented yourself —
# maybe 60 lines — plus a real understanding of when it silently does nothing.
#
# **What to expect — the honest part.** This notebook **doesn't train anything**.
# Real GRPO needs a GPU and hours; that's notebook 13. Here you build the
# machinery and test it on small deterministic examples where you can verify the
# maths by hand. Less satisfying, far more instructive.
#
# **The one insight to keep.** If all eight attempts are right, or all eight are
# wrong, you learn **nothing** — there's no "better than its siblings" when
# they're all the same. Your problems must be ones the model gets right *some* of
# the time. Too easy and too hard both teach nothing, which is why difficulty
# selection matters more than any hyperparameter here. You'll measure this
# yourself.
#
# ## The idea that changed post-training
#
# **The restaurant, one last time.** In notebook 09 you trained a food critic,
# and the chef learned to game them. In notebook 10 you skipped the critic and
# learned directly from diners' preferences — better, but still bounded by
# opinion, and still only about dishes someone already tasted.
#
# Now imagine a different kind of feedback entirely: **the health inspection.**
# The kitchen either passes or it fails. There is no taste, no preference, no
# critic to charm — just a check that either succeeds or doesn't. You cannot
# sweet-talk a thermometer.
#
# That is RLVR. And notice what changes: the chef can now *practise*. Cook a
# dish, check it, adjust, cook again — thousands of times, with no diner and no
# critic in the loop, because the verification is free and objective. That is why
# this technique produced reasoning models: for maths and code, correctness is
# checkable, so the model can generate its own attempts and learn from which
# ones actually worked.
#
# The catch, which you will see in the code: it only works where a verifier
# exists. There is no health inspection for "write me a moving poem."
#
# Notebook 09 showed the core problem with learned reward models: they get
# hacked. The policy finds inputs where the proxy is high and quality isn't.
#
# **RLVR (RL with Verifiable Rewards) sidesteps this entirely.** For tasks with
# a checkable answer, don't learn a reward — *compute* it:
#
# | domain | verifier |
# |---|---|
# | math | does the final answer equal ground truth? |
# | code | do the unit tests pass? |
# | format | does the output match the required schema? |
# | logic | does a solver confirm it? |
#
# An exact verifier **cannot be hacked** in the reward-model sense. There's no
# proxy to exploit — the reward *is* the objective. (You can still get
# degenerate strategies, like guessing common answers; more on that below.)
#
# This is how DeepSeek-R1 was trained, and it's the biggest post-training idea
# of the last few years.

# %% [markdown]
# ## Why GRPO instead of PPO
#
# PPO needs a **value network** — a second model, roughly the same size,
# predicting expected return per token, to compute the advantage
# `A = R − V(s)`. That's a lot of memory and another thing to tune.
#
# GRPO's trick: **sample a group of G responses to the same prompt, and use the
# group's own mean reward as the baseline.**
#
# ```
# PPO:   A_i = R_i − V(state)          <- learned value network
# GRPO:  A_i = (R_i − mean(R)) / std(R)  <- the group IS the baseline
# ```
#
# The value network disappears. You trade it for G× more sampling, which is a
# good deal because sampling parallelizes and a value network doesn't.
#
# The intuition is exactly the "grading on a curve" idea: within a group of
# attempts at the same problem, responses better than the group average get
# reinforced; worse-than-average get suppressed.

# %%
import math
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print(f"device: {device}")

# %% [markdown]
# ## Step 1 — Reward functions (the verifiers)
#
# In RLVR, reward design *is* the task design. Typically you combine a
# correctness reward with a format reward.

# %%
def extract_answer(text: str) -> str | None:
    """Pull the final answer out of a model response.

    Order matters: check the strict format first, then fall back to looser
    patterns. A too-loose extractor gives credit for the right digit appearing
    anywhere, which the policy will absolutely learn to exploit.
    """
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"####\s*([\-0-9\.\,]+)", text)     # GSM8K's own format
    if m:
        return m.group(1).strip().replace(",", "")
    m = re.findall(r"-?\d+\.?\d*", text)              # last number in the text
    return m[-1] if m else None


def normalize_number(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def correctness_reward(completion: str, ground_truth: str) -> float:
    """1.0 if the final answer matches, else 0.0. The unhackable part."""
    pred = normalize_number(extract_answer(completion))
    true = normalize_number(extract_answer(ground_truth) or ground_truth)
    if pred is None or true is None:
        return 0.0
    return 1.0 if abs(pred - true) < 1e-4 else 0.0


def format_reward(completion: str) -> float:
    """Partial credit for following the required structure."""
    score = 0.0
    if re.search(r"<think>.*?</think>", completion, re.DOTALL):
        score += 0.5
    if re.search(r"<answer>.*?</answer>", completion, re.DOTALL):
        score += 0.5
    return score


def total_reward(completion: str, ground_truth: str,
                 w_correct: float = 1.0, w_format: float = 0.2) -> float:
    return (w_correct * correctness_reward(completion, ground_truth)
            + w_format * format_reward(completion))


GT = "The answer is #### 18"
tests = [
    ("<think>3 x 6 = 18</think><answer>18</answer>", "perfect: correct + formatted"),
    ("<think>Let me see...</think><answer>21</answer>", "wrong answer, good format"),
    ("The answer is 18",                              "correct, no format"),
    ("<answer>18</answer>",                           "correct, half format"),
    ("I don't know",                                  "nothing"),
]
print(f"{'completion':<48}{'correct':>9}{'format':>8}{'total':>8}")
print("-" * 73)
for c, _ in tests:
    print(f"{c[:46]:<48}{correctness_reward(c, GT):>9.1f}"
          f"{format_reward(c):>8.1f}{total_reward(c, GT):>8.2f}")

# %% [markdown]
# ### Reward design is where you'll spend your time
#
# Three rules learned the hard way:
#
# 1. **Make correctness dominate.** If format is worth too much, the policy
#    learns to emit beautiful empty templates. Keep the format weight small.
# 2. **Be strict about extraction.** A loose regex rewards accidental
#    correctness, and the policy will find that.
# 3. **Avoid dense shaping unless you must.** Rewarding intermediate steps
#    invites hacking them. Sparse, verifiable, terminal rewards are safer.

# %% [markdown]
# ## Step 2 — Group-relative advantages
#
# The heart of GRPO.

# %%
def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-4,
                             scale_by_std: bool = True) -> torch.Tensor:
    """rewards: (n_prompts, group_size) -> advantages of the same shape."""
    mean = rewards.mean(dim=1, keepdim=True)
    advantages = rewards - mean
    if scale_by_std:
        # Dividing by std normalizes the update size across prompts of
        # differing difficulty. It's also the step "Dr. GRPO" argues introduces
        # a bias toward low-variance (easy or uniformly-failed) prompts.
        advantages = advantages / (rewards.std(dim=1, keepdim=True) + eps)
    return advantages


examples = {
    "mixed (the useful case)":     [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "all correct (no signal)":     [1.0] * 8,
    "all wrong (no signal)":       [0.0] * 8,
    "one lucky success":           [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "one failure":                 [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
}

for name, r in examples.items():
    rt = torch.tensor([r])
    adv = compute_group_advantages(rt)[0]
    print(f"\n{name}")
    print(f"  rewards:    {[f'{x:.1f}' for x in r]}")
    print(f"  advantages: {[f'{x:+.2f}' for x in adv.tolist()]}")

# %% [markdown]
# **Look at the degenerate cases.** When all 8 responses are correct — or all
# wrong — every advantage is **exactly zero**. No gradient. The group learns
# nothing from that prompt.
#
# This is not a bug; it's informative:
#
# - **All correct** → the problem is too easy. No information left.
# - **All wrong** → the problem is too hard. The policy never stumbles on a
#   success, so there's nothing to reinforce.
#
# The practical consequence: **GRPO only learns from problems at the edge of the
# model's ability.** Curriculum matters enormously. If your dataset is all too
# hard, training does nothing and you'll stare at a flat curve wondering why.
#
# Measure this — the fraction of groups with non-zero advantage is your real
# effective batch size.

# %%
def useful_fraction(all_rewards: torch.Tensor) -> float:
    """Fraction of prompt-groups that produce any gradient at all."""
    return (all_rewards.std(dim=1) > 1e-6).float().mean().item()


torch.manual_seed(0)
print(f"{'model accuracy':>16}{'useful groups':>16}")
print("-" * 32)
useful_curve = {}
for p in [0.05, 0.2, 0.5, 0.8, 0.95]:
    rewards = (torch.rand(500, 8) < p).float()
    useful_curve[p] = useful_fraction(rewards)
    print(f"{p:>16.0%}{useful_curve[p]:>16.1%}")

# %% [markdown]
# The signal peaks when the model is right about half the time, and collapses at
# both extremes. **Pick problems the model gets right 20–80% of the time.** This
# is why RLVR runs often start with a difficulty-filtered subset.

# %% [markdown]
# ## Step 3 — The GRPO loss
#
# Per-token, with PPO-style clipping and an explicit KL penalty:
#
# ```
# ratio_t   = π(a_t|s_t) / π_old(a_t|s_t)
# L_t       = min( ratio_t · A , clip(ratio_t, 1−ε, 1+ε) · A )
# L         = −mean_t( L_t ) + β · KL(π ‖ π_ref)
# ```
#
# **Why clip?** The advantage was computed under `π_old`. If the policy moves too
# far in one update, that estimate is no longer valid. Clipping caps how much any
# single token's probability can change per step.
#
# Note the asymmetry: clipping only bites when the update would be *large in the
# direction the advantage favours*. It's a trust region, not a general bound.

# %%
def grpo_loss(
    policy_logprobs: torch.Tensor,      # (B, T) current policy
    old_logprobs: torch.Tensor,         # (B, T) policy that generated the data
    ref_logprobs: torch.Tensor,         # (B, T) frozen reference (SFT model)
    advantages: torch.Tensor,           # (B,) one scalar per response
    completion_mask: torch.Tensor,      # (B, T) 1 for generated tokens
    beta: float = 0.04,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    adv = advantages.unsqueeze(1)       # broadcast the scalar over the sequence

    ratio = torch.exp(policy_logprobs - old_logprobs)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv
    # min() takes the PESSIMISTIC option — this is what makes it a lower bound
    # on the true objective, and why it's safe to take multiple steps on one
    # batch of samples.
    policy_loss = -torch.min(unclipped, clipped)

    # k3 estimator of KL: always non-negative and lower variance than the naive
    # (logp - logp_ref) difference. From Schulman's "approximating KL" note.
    log_diff = ref_logprobs - policy_logprobs
    kl = torch.exp(log_diff) - log_diff - 1.0

    per_token = policy_loss + beta * kl
    # Normalize by the number of REAL tokens, not the padded length.
    n_tokens = completion_mask.sum().clamp(min=1)
    loss = (per_token * completion_mask).sum() / n_tokens

    with torch.no_grad():
        clip_frac = (((ratio < 1 - eps_low) | (ratio > 1 + eps_high))
                     * completion_mask).sum() / n_tokens
        metrics = {
            "loss": loss.item(),
            "kl": ((kl * completion_mask).sum() / n_tokens).item(),
            "clip_frac": clip_frac.item(),
            "ratio_mean": ((ratio * completion_mask).sum() / n_tokens).item(),
        }
    return loss, metrics


# Behaviour probes.
B, T = 4, 10
mask = torch.ones(B, T)
old = torch.full((B, T), -2.0)

print(f"{'scenario':<44}{'loss':>9}{'kl':>8}{'clipfrac':>10}")
print("-" * 71)
probes = {}
for name, pol_delta, adv in [
    ("no change, positive advantage",   0.0,  torch.tensor([1.0, 1.0, 1.0, 1.0])),
    ("small increase, positive adv",    0.1,  torch.tensor([1.0, 1.0, 1.0, 1.0])),
    ("LARGE increase, positive adv",    1.0,  torch.tensor([1.0, 1.0, 1.0, 1.0])),
    ("small increase, negative adv",    0.1,  torch.tensor([-1.0, -1.0, -1.0, -1.0])),
    ("mixed advantages",                0.1,  torch.tensor([1.5, -0.5, 0.5, -1.5])),
]:
    pol = old + pol_delta
    _, m = grpo_loss(pol, old, old, adv, mask)
    probes[name] = m
    print(f"{name:<44}{m['loss']:>9.4f}{m['kl']:>8.4f}{m['clip_frac']:>10.2f}")

# %% [markdown]
# Note the third row: a large policy move with positive advantage gets **100%
# clipped**. That's the trust region doing its job — without it, one confident
# batch could destroy the policy.

# %% [markdown]
# ## Step 4 — The full GRPO loop
#
# ```
# repeat:
#   1. sample a batch of prompts
#   2. generate G completions per prompt  (the expensive part)
#   3. score each with the verifier
#   4. compute group-relative advantages
#   5. take a few gradient steps on the GRPO loss
# ```
#
# Step 2 dominates wall-clock — often 70–90% of the time. That's why vLLM
# integration matters so much for real runs.

# %%
@dataclass
class GRPOConfig_:
    group_size: int = 8          # G — completions per prompt
    n_prompts: int = 4           # prompts per iteration
    max_new_tokens: int = 200
    temperature: float = 1.0     # must be > 0! see the warning below
    beta: float = 0.04           # KL coefficient
    eps: float = 0.2             # clip range
    lr: float = 1e-6
    inner_epochs: int = 1        # gradient steps per batch of samples


@torch.no_grad()
def generate_group(model, tokenizer, prompt_ids: torch.Tensor,
                   cfg: GRPOConfig_) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate G completions for one prompt. Returns (sequences, completion_mask)."""
    # Repeat the prompt G times and sample — the group must be DIVERSE, or every
    # advantage is zero and you learn nothing.
    batch = prompt_ids.repeat(cfg.group_size, 1)
    out = model.generate(
        batch,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,                  # NEVER greedy here
        temperature=cfg.temperature,
        top_p=1.0,                       # don't truncate the distribution:
                                         # it biases the importance ratios
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = prompt_ids.shape[1]
    mask = torch.zeros_like(out, dtype=torch.float)
    mask[:, prompt_len:] = 1.0
    mask[out == tokenizer.pad_token_id] = 0.0
    return out, mask


# %% [markdown]
# ### The single most common GRPO bug
#
# **Sampling with `temperature=0` or `do_sample=False`.** All G completions come
# out identical, every reward is identical, every advantage is exactly zero, and
# the loss sits at a constant while nothing learns. It looks like a broken
# learning rate; it's a broken sampler.
#
# Verify diversity explicitly — this check costs nothing and saves hours.

# %%
def check_group_diversity(completions: list[str]) -> dict:
    unique = len(set(completions))
    return {
        "n": len(completions),
        "unique": unique,
        "diversity": unique / max(len(completions), 1),
    }


print("group diversity check:")
for name, group in [
    ("healthy (varied samples)", ["answer A", "answer B", "answer C", "answer A"]),
    ("BROKEN (greedy decoding)", ["answer A"] * 4),
]:
    d = check_group_diversity(group)
    flag = "" if d["diversity"] > 0.3 else "   <-- temperature=0? do_sample=False?"
    print(f"  {name:<28} {d['unique']}/{d['n']} unique{flag}")

# %%
def grpo_train_step(policy, ref_model, tokenizer, prompts: list[str],
                    ground_truths: list[str], optimizer, cfg: GRPOConfig_) -> dict:
    """One full GRPO iteration over a batch of prompts."""
    all_seqs, all_masks, all_rewards = [], [], []

    # --- rollout phase ---
    policy.eval()
    for prompt, gt in zip(prompts, ground_truths):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(policy.device)
        seqs, mask = generate_group(policy, tokenizer, ids, cfg)

        texts = [tokenizer.decode(s[ids.shape[1]:], skip_special_tokens=True) for s in seqs]
        rewards = torch.tensor([total_reward(t, gt) for t in texts], device=policy.device)

        all_seqs.append(seqs)
        all_masks.append(mask)
        all_rewards.append(rewards)

    rewards_matrix = torch.stack(all_rewards)                     # (n_prompts, G)
    advantages = compute_group_advantages(rewards_matrix)          # same shape

    # --- learning phase ---
    policy.train()
    metrics_acc = []
    for i, (seqs, mask) in enumerate(zip(all_seqs, all_masks)):
        adv = advantages[i]

        with torch.no_grad():
            # old_logprobs come from the policy AS IT WAS when sampling. With
            # inner_epochs=1 they equal the current policy, so the ratio is
            # exactly 1 and clipping never fires. With >1 they diverge — which
            # is the whole point of the clipped objective.
            old_lp = token_logprobs(policy, seqs)
            ref_lp = token_logprobs(ref_model, seqs)

        for _ in range(cfg.inner_epochs):
            pol_lp = token_logprobs(policy, seqs)
            loss, m = grpo_loss(pol_lp, old_lp, ref_lp, adv, mask[:, 1:],
                                beta=cfg.beta, eps_low=cfg.eps, eps_high=cfg.eps)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            metrics_acc.append(m)

    return {
        "reward_mean": rewards_matrix.mean().item(),
        "reward_std": rewards_matrix.std().item(),
        "accuracy": (rewards_matrix >= 1.0).float().mean().item(),
        "useful_groups": useful_fraction(rewards_matrix),
        "loss": sum(m["loss"] for m in metrics_acc) / len(metrics_acc),
        "kl": sum(m["kl"] for m in metrics_acc) / len(metrics_acc),
        "clip_frac": sum(m["clip_frac"] for m in metrics_acc) / len(metrics_acc),
    }


def token_logprobs(model, sequences: torch.Tensor) -> torch.Tensor:
    """Per-token log p under `model`. Returns (B, T-1)."""
    logits = model(sequences).logits[:, :-1, :]
    targets = sequences[:, 1:]
    logps = torch.log_softmax(logits, dim=-1)
    return torch.gather(logps, 2, targets.unsqueeze(2)).squeeze(2)


# %% [markdown]
# ## Step 5 — The prompt matters as much as the algorithm
#
# DeepSeek-R1's system prompt is famously simple. It does not teach reasoning —
# it *creates room* for reasoning, and RL fills it.

# %%
R1_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a \
question, and the Assistant solves it. The Assistant first thinks about the \
reasoning process in the mind and then provides the user with the answer. The \
reasoning process and answer are enclosed within <think> </think> and <answer> \
</answer> tags, respectively."""

print(R1_SYSTEM_PROMPT)
print("\nThat's it. No worked examples, no chain-of-thought demonstrations.")
print("The format reward makes the tags appear; the correctness reward makes")
print("the content inside them useful. Longer reasoning EMERGES because it")
print("raises the chance of a correct final answer — nobody asked for it.")

# %% [markdown]
# ## What to watch during a GRPO run
#
# | metric | healthy | trouble |
# |---|---|---|
# | `reward_mean` | rises slowly | flat = no learning; spikes = hacking |
# | `accuracy` | rises | flat at 0 = problems too hard |
# | `useful_groups` | 0.3–0.8 | ~0 = all groups degenerate; fix difficulty |
# | `kl` | small, stable | growing fast = policy running away |
# | `clip_frac` | 0.05–0.2 | >0.4 = steps too large, lower LR |
# | completion length | often grows | sudden collapse to 1 token = reward hack |
#
# ### Reward hacking still happens with verifiers
#
# The reward function itself can't be gamed, but the *setup* can:
#
# - Model emits `<answer>42</answer>` immediately with no reasoning, because a
#   common answer sometimes hits.
# - Model learns your regex, not the math — e.g. printing every number 1–100 so
#   "the last number" is sometimes right.
# - Model produces enormously long reasoning because length correlates with
#   accuracy in your data, blowing up your compute.
#
# Fix these in the **reward function and extractor**, not the algorithm.
#
# ## The variants
#
# | variant | change |
# |---|---|
# | **Dr. GRPO** | drop the `/std` normalization (removes a difficulty bias) |
# | **DAPO** | asymmetric clipping (`eps_high` > `eps_low`), dynamic sampling |
# | **GSPO** | importance ratios at the sequence level, not per token |
# | **RLOO** | leave-one-out baseline instead of the group mean |
#
# All are small edits to what you've written above — which is the point of
# building it yourself.

# %% [markdown]
# ## Record this run
#
# Nothing here is trained — `grpo_train_step` needs a GPU and lands in notebook
# 13. What this notebook produces instead is a set of **seeded correctness
# probes**, and those are worth tracking precisely because they are
# deterministic: if you implement Dr. GRPO, swap in DAPO's asymmetric clipping,
# or try the RLOO baseline, these numbers move, and the delta in `BENCHMARK.md`
# tells you *how* your variant differs from vanilla GRPO.
#
# Two invariants to watch. `clip_frac_unchanged` must stay at 0 — if a policy
# identical to the sampler is getting clipped, the ratio is wrong. And
# `useful_groups` peaks near 50% accuracy and collapses at both extremes, which
# is the whole argument for curriculum difficulty.

# %%
import sys

sys.path.insert(0, "..")          # repo root, so `llmfs` is importable
from llmfs.bench import log_run

log_run(
    stage="12_grpo_rlvr_from_scratch",
    metrics={
        "useful_groups_at_50pct": useful_curve[0.5],
        "useful_groups_at_95pct": useful_curve[0.95],
        "clip_frac_unchanged": probes["no change, positive advantage"]["clip_frac"],
        "clip_frac_large_update": probes["LARGE increase, positive adv"]["clip_frac"],
        "kl_large_update": probes["LARGE increase, positive adv"]["kl"],
    },
    key="useful_groups_at_50pct",
    config={"group_size": 8, "eps": 0.2, "n_groups": 500, "seed": 0},
    notes="seeded probes, vanilla GRPO",
)

# %% [markdown]
# ## Exercises
#
# 1. **Break the sampler.** Set `temperature=0.0` and confirm all advantages go
#    to zero. Now you'll recognize this instantly in the wild.
# 2. **Group size.** G ∈ {2, 4, 8, 16}. Plot `useful_groups` and wall-clock.
#    Where's the knee?
# 3. **Drop the KL.** Set `beta=0`. Watch the policy drift — track output length
#    and readability, not just reward.
# 4. **Hack your own reward.** Write a deliberately loose extractor (any number
#    anywhere) and see what the policy learns to emit.
#
# ## Checkpoint
#
# - [ ] You can explain why GRPO needs no value network
# - [ ] You know why all-correct and all-wrong groups give zero gradient
# - [ ] You know the temperature=0 failure by heart
# - [ ] You know what `clip_frac` tells you
#
# **Next:** `13_grpo_with_trl.ipynb` — run this on GSM8K for real.

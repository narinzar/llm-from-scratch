# %% [markdown]
# # 10 — DPO from Scratch: Alignment Without a Reward Model
#
# **Goal:** derive the DPO loss, implement it, train with it, and understand its
# failure modes.
#
# **Time:** 45–60 min.
#
# ## The problem DPO solves
#
# Classic RLHF is a three-stage pipeline, and stage 3 is genuinely painful:
#
# ```
# 1. SFT                          (easy)
# 2. train a reward model         (medium — notebook 09)
# 3. PPO against the reward model (hard)
# ```
#
# PPO needs **four models in memory** (policy, reference, reward, value), is
# notoriously sensitive to hyperparameters, and can collapse in ways that are
# hard to diagnose.
#
# DPO's insight: **the optimal policy for the RLHF objective has a closed form,
# so you can solve for the reward in terms of the policy and eliminate it.**
# What's left is a supervised loss on preference pairs. No reward model, no
# sampling loop, no value network.

# %% [markdown]
# ## The derivation (worth following once)
#
# **Step 1 — the RLHF objective.** Maximize reward while staying close to the
# reference (SFT) policy:
#
# ```
# max_π  E[r(x,y)]  −  β · KL(π ‖ π_ref)
# ```
#
# The KL term is essential: without it the policy runs off to whatever
# degenerate text maximizes the reward model.
#
# **Step 2 — the optimum has a closed form.** This is a standard result:
#
# ```
# π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)
# ```
#
# **Step 3 — invert it.** Solve for the reward:
#
# ```
# r(x,y) = β · log( π*(y|x) / π_ref(y|x) ) + β·log Z(x)
# ```
#
# **Step 4 — the partition function cancels.** Substitute into Bradley–Terry.
# `P(y_w > y_l) = sigmoid(r(x,y_w) − r(x,y_l))` — and since `β log Z(x)` depends
# only on `x`, it appears in both terms and **cancels**. That intractable `Z(x)`
# was the entire reason you needed RL, and it evaporates.
#
# **The DPO loss:**
#
# ```
# L = −log sigmoid( β · [ log(π(y_w|x)/π_ref(y_w|x))
#                       − log(π(y_l|x)/π_ref(y_l|x)) ] )
# ```
#
# **The model is its own reward model.** That's the whole trick.

# %%
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path("..").resolve()))
from llmfs.model import GPT, GPTConfig  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print(f"device: {device}")

# %% [markdown]
# ## Step 1 — Sequence log-probabilities
#
# DPO needs `log π(y|x)`: the summed log-probability of the response tokens
# given the prompt. Two details that are easy to get wrong:
#
# 1. **Only response tokens count.** Prompt tokens are given, not chosen.
# 2. **Sum, don't average** (by default). The theory calls for the sum. Averaging
#    gives you length normalization, which changes the objective — that's the
#    difference between DPO and IPO-style variants.

# %%
def sequence_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    average: bool = False,
) -> torch.Tensor:
    """Sum of log p(token) over positions where labels != -100. Returns (B,)."""
    logits, _ = model(input_ids)

    # Same shift as SFT: logits[:, :-1] predicts labels[:, 1:]
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    mask = labels != -100
    # gather() would fail on -100, so substitute a valid index first; the mask
    # removes those positions afterwards anyway.
    safe_labels = labels.masked_fill(~mask, 0)

    logps = torch.log_softmax(logits, dim=-1)
    token_logps = torch.gather(logps, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
    token_logps = token_logps * mask

    if average:
        return token_logps.sum(-1) / mask.sum(-1).clamp(min=1)
    return token_logps.sum(-1)


# Sanity-check against a hand computation.
m = GPT(GPTConfig(vocab_size=50, block_size=16, n_layer=1, n_head=2, n_embd=32))
ids = torch.randint(0, 50, (2, 8))
lbl = ids.clone()
lbl[:, :3] = -100          # first 3 positions are "prompt"

got = sequence_logprobs(m, ids, lbl)

with torch.no_grad():
    logits, _ = m(ids)
    lp = torch.log_softmax(logits[:, :-1], dim=-1)
    manual = torch.stack([
        sum(lp[b, t, ids[b, t + 1]] for t in range(2, 7))   # positions 3..7 of labels
        for b in range(2)
    ])

print(f"sequence_logprobs: {got.detach().numpy().round(4)}")
print(f"manual loop:       {manual.numpy().round(4)}")
print(f"match: {torch.allclose(got, manual, atol=1e-4)}")

# %% [markdown]
# ## Step 2 — The DPO loss

# %%
def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Returns (loss, metrics)."""
    # The implicit reward: how much more likely the policy makes this response
    # than the reference does. This IS r(x,y) up to the constant that cancelled.
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    logits = chosen_rewards - rejected_rewards

    if label_smoothing > 0:
        # cDPO: assume a fraction of preference labels are simply wrong.
        # Prevents driving rejected probability to zero on mislabelled pairs.
        losses = (
            -F.logsigmoid(logits) * (1 - label_smoothing)
            - F.logsigmoid(-logits) * label_smoothing
        )
    else:
        losses = -F.logsigmoid(logits)

    metrics = {
        "rewards/chosen": chosen_rewards.mean().item(),
        "rewards/rejected": rejected_rewards.mean().item(),
        "rewards/margin": (chosen_rewards - rejected_rewards).mean().item(),
        "rewards/accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
    }
    return losses.mean(), metrics


# Behaviour check across scenarios.
def probe(pc, pr, rc, rr, beta=0.1):
    t = lambda v: torch.tensor([float(v)])
    loss, mt = dpo_loss(t(pc), t(pr), t(rc), t(rr), beta=beta)
    return loss.item(), mt


print(f"{'scenario':<44}{'loss':>8}{'margin':>9}")
print("-" * 61)
cases = [
    ("policy == reference (start of training)", (-10, -12, -10, -12)),
    ("policy prefers chosen MORE than ref",     (-9, -13, -10, -12)),
    ("policy prefers rejected more (wrong)",    (-11, -11, -10, -12)),
    ("policy strongly favours chosen",          (-5, -20, -10, -12)),
]
for name, args in cases:
    l, mt = probe(*args)
    print(f"{name:<44}{l:>8.4f}{mt['rewards/margin']:>9.3f}")

print(f"\nAt initialization policy == reference, so the margin is exactly 0 and")
print(f"the loss is ln(2) = {math.log(2):.4f}. Same random baseline as notebook 09.")

# %% [markdown]
# ## Step 3 — What beta actually controls
#
# `beta` is the KL penalty strength — how hard the policy is held to the
# reference.
#
# | beta | behaviour |
# |---|---|
# | 0.01–0.05 | weak constraint; large changes, risk of degeneration |
# | **0.1** | **the standard default** |
# | 0.3–0.5 | strong constraint; safe, small changes |
#
# Mechanically, beta scales the logits going into the sigmoid, which changes how
# saturated the loss is — and therefore how large the gradients are.

# %%
diffs = torch.linspace(-3, 3, 13)
print(f"{'logratio diff':>14}" + "".join(f"{f'β={b}':>10}" for b in [0.05, 0.1, 0.5]))
print("-" * 46)
for d in diffs[::3]:
    row = f"{d.item():>14.2f}"
    for b in [0.05, 0.1, 0.5]:
        row += f"{-F.logsigmoid(torch.tensor(b * d.item())).item():>10.4f}"
    print(row)
print("\nHigher beta => more loss curvature => stronger push per unit of drift.")

# %% [markdown]
# ## Step 4 — The training loop
#
# **The reference model is frozen and never updated.** Two ways to get its
# log-probs:
#
# 1. Keep a second frozen copy (simple, costs ~2× memory for the weights).
# 2. **Precompute** ref log-probs once before training (memory-efficient — the
#    reference never changes, so why keep recomputing it?). TRL does this when
#    `precompute_ref_log_probs=True`.
#
# With LoRA there's a third, elegant option: **disable the adapters** to recover
# the reference. Same weights, zero extra memory. TRL does this automatically
# when you pass a `peft_config`.

# %%
@dataclass
class DPOConfig_:
    beta: float = 0.1
    lr: float = 5e-7           # VERY low — see the warning below
    batch_size: int = 4
    epochs: int = 1
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    log_every: int = 10


def train_dpo(policy: nn.Module, reference: nn.Module, pairs: list[dict],
              cfg: DPOConfig_):
    # Freeze the reference. If you forget this, the reference drifts toward the
    # policy, the log-ratio collapses toward 0, and the loss goes nowhere while
    # looking superficially fine.
    reference.eval()
    for p in reference.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    print(f"{'step':>6}{'loss':>9}{'acc':>7}{'margin':>9}{'r_chosen':>10}{'r_reject':>10}")
    print("-" * 51)

    policy.train()
    step = 0
    for _ in range(cfg.epochs):
        perm = torch.randperm(len(pairs))
        for i in range(0, len(pairs) - cfg.batch_size, cfg.batch_size):
            batch = [pairs[j] for j in perm[i : i + cfg.batch_size].tolist()]

            c_ids = torch.stack([b["chosen_ids"] for b in batch]).to(device)
            c_lbl = torch.stack([b["chosen_labels"] for b in batch]).to(device)
            r_ids = torch.stack([b["rejected_ids"] for b in batch]).to(device)
            r_lbl = torch.stack([b["rejected_labels"] for b in batch]).to(device)

            with ctx:
                pol_c = sequence_logprobs(policy, c_ids, c_lbl)
                pol_r = sequence_logprobs(policy, r_ids, r_lbl)
                with torch.no_grad():
                    ref_c = sequence_logprobs(reference, c_ids, c_lbl)
                    ref_r = sequence_logprobs(reference, r_ids, r_lbl)

                loss, mt = dpo_loss(pol_c, pol_r, ref_c, ref_r,
                                    beta=cfg.beta,
                                    label_smoothing=cfg.label_smoothing)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt.step()

            if step % cfg.log_every == 0:
                print(f"{step:>6}{loss.item():>9.4f}{mt['rewards/accuracy']:>7.2f}"
                      f"{mt['rewards/margin']:>9.3f}{mt['rewards/chosen']:>10.3f}"
                      f"{mt['rewards/rejected']:>10.3f}")
            step += 1
    return policy


# %% [markdown]
# ### Why the learning rate is 5e-7
#
# That is **40× lower than SFT** and it looks like a typo. It isn't.
#
# DPO's gradient pushes the policy *away* from `π_ref` on the rejected response.
# There's nothing anchoring absolute probabilities — only the *ratio* appears in
# the loss. So the easiest way to increase the margin is to **crush the
# probability of the rejected response**, and with it, everything nearby.
#
# At a normal LR this happens fast, and the model degenerates: it stops
# producing the rejected text and also stops producing anything good. The tell
# is in the metrics — **`rewards/chosen` going strongly negative**. The margin
# looks great while the model gets worse. Watch that number, not the loss.

# %% [markdown]
# ## Step 5 — Watch degeneration happen
#
# A synthetic demo, so you can see the failure without a long run.

# %%
def degeneration_demo(lr: float, beta: float = 0.1, steps: int = 400):
    """Train a tiny model with DPO on fixed pairs; track ABSOLUTE logprobs."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=2, n_embd=64)
    policy = GPT(cfg)
    reference = GPT(cfg)
    reference.load_state_dict(policy.state_dict())
    for p in reference.parameters():
        p.requires_grad = False

    # Realistic preference pairs: SAME prompt, and two responses that overlap
    # heavily (differing only near the end). This matters — with two totally
    # unrelated responses the model can push one down without touching the
    # other, and you never see the failure. Real preference pairs share a
    # prompt and most of their vocabulary, which is exactly what couples them.
    prompt = torch.randint(0, 100, (4, 4))
    resp_c = torch.randint(0, 100, (4, 12))
    resp_r = resp_c.clone()
    resp_r[:, -3:] = torch.randint(0, 100, (4, 3))   # differ only in the tail

    ids_c = torch.cat([prompt, resp_c], dim=1)
    ids_r = torch.cat([prompt, resp_r], dim=1)
    lbl_c, lbl_r = ids_c.clone(), ids_r.clone()
    lbl_c[:, :4] = -100          # mask the prompt
    lbl_r[:, :4] = -100

    with torch.no_grad():
        ref_c = sequence_logprobs(reference, ids_c, lbl_c)
        ref_r = sequence_logprobs(reference, ids_r, lbl_r)

    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    trace = []
    for s in range(steps + 1):
        pc = sequence_logprobs(policy, ids_c, lbl_c)
        pr = sequence_logprobs(policy, ids_r, lbl_r)
        loss, mt = dpo_loss(pc, pr, ref_c, ref_r, beta=beta)
        if s % (steps // 5) == 0:
            trace.append((s, loss.item(), mt["rewards/margin"],
                          pc.mean().item(), pr.mean().item()))
        opt.zero_grad(); loss.backward(); opt.step()
    return trace


for lr in [1e-5, 1e-4]:
    print(f"\n=== lr = {lr:.0e} ===")
    print(f"{'step':>6}{'loss':>9}{'margin':>9}{'logp(chosen)':>15}{'logp(rejected)':>16}")
    print("-" * 55)
    for s, l, mg, pc, pr in degeneration_demo(lr):
        print(f"{s:>6}{l:>9.4f}{mg:>9.3f}{pc:>15.2f}{pr:>16.2f}")

# %% [markdown]
# **Read the two tables by column, not by loss.**
#
# At `lr=1e-5` everything is healthy: the margin grows *and* `logp(chosen)`
# rises. The model genuinely learns to prefer the chosen response.
#
# At `lr=1e-4` watch `logp(chosen)` specifically. It rises, **peaks around step
# 240, and then starts falling** — while the margin keeps climbing and the loss
# keeps dropping. Past that turning point the model is no longer learning to
# like the chosen response; it's suppressing the rejected one so aggressively
# that it drags the chosen response (which shares most of its tokens) down too.
#
# Loss and margin both look *better* the whole time. Only the absolute log-prob
# reveals the problem.
#
# **This is why you log `rewards/chosen` and stop when it turns over.** In a real
# run the symptom is a model that wins on your preference metric and produces
# noticeably worse text. Note also how mild the settings are — this is not an
# exotic failure, it's the default outcome of a slightly-too-high LR.

# %% [markdown]
# ## The DPO variant zoo
#
# Each fixes a specific flaw:
#
# | method | change | fixes |
# |---|---|---|
# | **IPO** | squared loss instead of log-sigmoid | overfitting to deterministic preferences |
# | **cDPO** | label smoothing (implemented above) | noisy/mislabelled preference pairs |
# | **KTO** | needs only good/bad labels, not pairs | pair collection is expensive |
# | **ORPO** | odds-ratio penalty added to the SFT loss | **removes the reference model entirely**; one stage |
# | **SimPO** | length-normalized reward, no reference | length bias + memory |
# | **RPO / RSO** | adds an SFT term on chosen | the degeneration above |
#
# **Practical advice:** start with plain DPO at `beta=0.1`. If chosen log-probs
# collapse, either lower the LR, or add an SFT loss term on the chosen response
# (`rpo_alpha` in TRL), which directly anchors absolute probabilities.
#
# ORPO is worth a look if you want to skip SFT and DPO as separate stages.

# %% [markdown]
# ## DPO vs online RL — the honest comparison
#
# | | DPO | PPO / GRPO |
# |---|---|---|
# | models in memory | 2 (policy + ref) | 3–4 |
# | needs generation during training | **no** | yes (slow) |
# | hyperparameter sensitivity | low | high |
# | data | fixed, offline pairs | fresh on-policy samples |
# | ceiling | **limited by the dataset** | can exceed it |
#
# The last row matters. DPO learns from preferences collected on *someone else's*
# responses. Online RL samples from the *current* policy, so it keeps getting
# signal about its own actual failure modes. That's why frontier labs still run
# online RL despite the cost — and why reasoning models are trained with GRPO,
# not DPO.
#
# ## Exercises
#
# 1. **Beta sweep.** β ∈ {0.01, 0.1, 0.5} on real data. Track margin *and*
#    absolute chosen log-prob. Find where degeneration starts.
# 2. **Forget the freeze.** Delete `requires_grad = False` on the reference.
#    Watch the loss look fine while nothing improves.
# 3. **Implement IPO.** Replace `-logsigmoid(β·h)` with `(h − 1/(2β))²`. Compare
#    robustness on deliberately noisy pairs.
# 4. **Add the RPO term.** `loss + alpha * sft_loss(chosen)`. Does it stop the
#    degeneration at high LR?
#
# ## Checkpoint
#
# - [ ] You can sketch why `Z(x)` cancels
# - [ ] You know what beta controls
# - [ ] You know why DPO's LR is ~5e-7
# - [ ] You know to watch `rewards/chosen`, not just the margin
#
# **Next:** `11_dpo_with_trl.ipynb` — the same thing on a real model.

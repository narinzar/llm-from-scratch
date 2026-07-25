# %% [markdown]
# # 09 — Reward Modeling: Learning What "Better" Means
#
# **Goal:** train a reward model from human preference pairs, implementing the
# Bradley–Terry loss from scratch, and see for yourself why reward models get
# hacked.
#
# **Time:** 45 min.
#
# ## Why preferences instead of labels
#
# SFT teaches the model to imitate good answers. But "good" is hard to write
# down. Ask a human to *write* the ideal response to "explain quantum computing"
# and you'll get something slow, expensive, and inconsistent.
#
# Ask them **"which of these two is better?"** and you get a fast, cheap, far
# more reliable signal. People are much better at comparing than at generating.
#
# That's the whole premise of preference learning: collect `(prompt, chosen,
# rejected)` triples and learn a scalar **reward** function consistent with them.
#
# ## Where this sits
#
# ```
#                 ┌─ reward model ─> PPO/GRPO         (classic RLHF)
# SFT model ──────┤
#                 └─ DPO / direct methods             (skip the reward model)
# ```
#
# DPO (notebook 10) skips the explicit reward model. So why learn this?
#
# 1. **DPO's derivation *is* Bradley–Terry.** You can't understand DPO's loss
#    without it.
# 2. Reward models are still needed for **online RL** (PPO, GRPO with a learned
#    reward), best-of-n sampling, and automated evaluation.
# 3. **Reward hacking** is the central failure mode of all of alignment, and
#    this is where you can watch it happen.

# %%
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print(f"device: {device}")

# %% [markdown]
# ## The Bradley–Terry model
#
# From 1952, originally for ranking chess players. Each item has a latent
# "strength" `r`. The probability that A beats B is:
#
# ```
# P(A > B) = exp(r_A) / (exp(r_A) + exp(r_B)) = sigmoid(r_A - r_B)
# ```
#
# Only **differences** matter — adding a constant to every reward changes
# nothing. (Remember that: it's why raw reward values are meaningless and only
# margins are interpretable.)
#
# Our reward model is a transformer with a scalar head. Train it by maximum
# likelihood on the observed preferences:
#
# ```
# L = -log sigmoid(r(prompt, chosen) - r(prompt, rejected))
# ```
#
# That's it. **One line.**

# %%
def bradley_terry_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor,
                       margin: float = 0.0) -> torch.Tensor:
    """-log sigmoid(r_chosen - r_rejected). Uses logsigmoid for stability."""
    # F.logsigmoid is numerically stable; log(sigmoid(x)) overflows for very
    # negative x. Always use the fused version.
    return -F.logsigmoid(r_chosen - r_rejected - margin).mean()


print(f"{'r_chosen':>10}{'r_reject':>10}{'margin':>9}{'P(correct)':>12}{'loss':>9}")
print("-" * 50)
for rc, rr in [(2.0, 1.0), (1.0, 2.0), (5.0, -5.0), (0.0, 0.0), (0.1, 0.0)]:
    c, r = torch.tensor([rc]), torch.tensor([rr])
    p = torch.sigmoid(c - r).item()
    print(f"{rc:>10.1f}{rr:>10.1f}{rc-rr:>9.1f}{p:>12.3f}"
          f"{bradley_terry_loss(c, r).item():>9.4f}")

# %% [markdown]
# Note the asymptotics: at margin 0 the loss is `ln 2 ≈ 0.693` (pure chance),
# and a wrong ordering is penalized hard. **A reward model at 0.693 loss has
# learned nothing** — that's your random baseline, exactly like `ln(vocab_size)`
# was for the language model.

# %% [markdown]
# ## The reward model architecture
#
# Take a pretrained transformer, throw away the LM head, bolt on a scalar head.
# Read the reward off the **last non-padding token**, because only that position
# has attended to the entire sequence.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))
from llmfs.model import GPT, GPTConfig  # noqa: E402


class RewardModel(nn.Module):
    """A transformer trunk with a scalar value head."""

    def __init__(self, backbone: GPT) -> None:
        super().__init__()
        self.backbone = backbone
        d = backbone.cfg.n_embd
        self.v_head = nn.Linear(d, 1, bias=False)
        # Small init: a large random head produces huge initial rewards and a
        # saturated sigmoid, which kills the gradient before training starts.
        nn.init.normal_(self.v_head.weight, std=1 / math.sqrt(d))

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Run the trunk but stop before the LM head — we want hidden states.
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.backbone.wte(input_ids) + self.backbone.wpe(pos)
        for block in self.backbone.blocks:
            x = block(x)
        x = self.backbone.ln_f(x)                 # (B, T, d)

        rewards = self.v_head(x).squeeze(-1)      # (B, T)

        if attention_mask is None:
            return rewards[:, -1]
        # Index of the last real token per sequence. Getting this wrong (e.g.
        # always taking [:, -1] on a padded batch) means you read the reward off
        # a PAD token and the model learns nothing useful.
        last_idx = attention_mask.sum(dim=1) - 1
        return rewards[torch.arange(B, device=rewards.device), last_idx]


rm = RewardModel(GPT(GPTConfig(vocab_size=1000, block_size=64, n_layer=2,
                               n_head=4, n_embd=128))).to(device)
ids = torch.randint(0, 1000, (4, 20), device=device)
mask = torch.ones_like(ids)
mask[2, 15:] = 0     # sequence 2 is padded from position 15
print(f"rewards: {rm(ids, mask).detach().cpu().numpy().round(3)}")
print(f"(shape {tuple(rm(ids, mask).shape)} — one scalar per sequence)")

# %% [markdown]
# ## Preference datasets
#
# | dataset | size | source of preference |
# |---|---|---|
# | `HuggingFaceH4/ultrafeedback_binarized` | 64k | GPT-4 rated, **the standard default** |
# | `Anthropic/hh-rlhf` | 170k | Human, helpfulness + harmlessness |
# | `argilla/distilabel-intel-orca-dpo-pairs` | 13k | AI feedback, cleaned |
# | `nvidia/HelpSteer2` | 21k | Human, multi-attribute ratings |
# | `openbmb/UltraFeedback` | 64k | The unbinarized original with scores |
#
# Note how many say "GPT-4 rated" rather than "human". That's **RLAIF** — RL
# from AI Feedback — and it's now more common than human labelling because it's
# ~1000× cheaper and, for many attributes, about as good. It does mean your
# model inherits the labeller model's biases.

# %%
from datasets import load_dataset

prefs = load_dataset("HuggingFaceH4/ultrafeedback_binarized",
                     split="train_prefs", streaming=True)

sample = next(iter(prefs.take(1)))
print("fields:", list(sample.keys()))
print(f"\nPROMPT: {sample['prompt'][:220]}")
print(f"\nCHOSEN   ({sample.get('score_chosen', '?')}): "
      f"{sample['chosen'][-1]['content'][:280]}")
print(f"\nREJECTED ({sample.get('score_rejected', '?')}): "
      f"{sample['rejected'][-1]['content'][:280]}")

# %% [markdown]
# **Read a dozen of these before you train.** Preference data is noisy —
# annotators disagree with each other ~25–40% of the time on subtle cases, which
# puts a hard ceiling on achievable accuracy. A reward model at 65–75% pairwise
# accuracy is doing *well*; if you see 95%, be suspicious that something in the
# data is leaking (like length).

# %% [markdown]
# ## Training the reward model
#
# The key trick: put chosen and rejected in **the same forward pass** by
# concatenating along the batch dimension. Same batch norm statistics, same
# dropout mask, half the launches.

# %%
@dataclass
class RMConfig:
    lr: float = 1e-5          # low — you're adapting a pretrained trunk
    batch_size: int = 8
    max_len: int = 512
    epochs: int = 1
    grad_clip: float = 1.0
    log_every: int = 20


def train_reward_model(rm: RewardModel, pairs: list[dict], cfg: RMConfig):
    """pairs: [{'chosen_ids': [...], 'rejected_ids': [...]}, ...]"""
    opt = torch.optim.AdamW(rm.parameters(), lr=cfg.lr, weight_decay=0.0)
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    n_steps = (len(pairs) // cfg.batch_size) * cfg.epochs
    print(f"{n_steps} steps\n")
    print(f"{'step':>6}{'loss':>9}{'acc':>8}{'margin':>9}")
    print("-" * 32)

    rm.train()
    step, run_loss, run_acc = 0, None, None
    for _ in range(cfg.epochs):
        perm = torch.randperm(len(pairs))
        for i in range(0, len(pairs) - cfg.batch_size, cfg.batch_size):
            batch = [pairs[j] for j in perm[i : i + cfg.batch_size].tolist()]

            maxlen = min(max(max(len(b["chosen_ids"]), len(b["rejected_ids"]))
                             for b in batch), cfg.max_len)

            def pad(seqs):
                ids = torch.zeros(len(seqs), maxlen, dtype=torch.long)
                msk = torch.zeros(len(seqs), maxlen, dtype=torch.long)
                for k, s in enumerate(seqs):
                    s = s[:maxlen]
                    ids[k, : len(s)] = torch.tensor(s)
                    msk[k, : len(s)] = 1
                return ids.to(device), msk.to(device)

            c_ids, c_mask = pad([b["chosen_ids"] for b in batch])
            r_ids, r_mask = pad([b["rejected_ids"] for b in batch])

            with ctx:
                # One forward for both halves.
                all_ids = torch.cat([c_ids, r_ids], dim=0)
                all_mask = torch.cat([c_mask, r_mask], dim=0)
                all_r = rm(all_ids, all_mask)
                r_c, r_r = all_r.chunk(2, dim=0)
                loss = bradley_terry_loss(r_c, r_r)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), cfg.grad_clip)
            opt.step()

            acc = (r_c > r_r).float().mean().item()
            margin = (r_c - r_r).mean().item()
            run_loss = loss.item() if run_loss is None else 0.9 * run_loss + 0.1 * loss.item()
            run_acc = acc if run_acc is None else 0.9 * run_acc + 0.1 * acc

            if step % cfg.log_every == 0:
                print(f"{step:>6}{run_loss:>9.4f}{run_acc:>8.3f}{margin:>9.3f}")
            step += 1
    return rm


# %% [markdown]
# ### The metrics to watch
#
# | metric | meaning | healthy value |
# |---|---|---|
# | loss | BT negative log-likelihood | starts 0.693, falls to 0.4–0.6 |
# | **accuracy** | fraction where `r_chosen > r_rejected` | **0.65–0.75** |
# | margin | mean `r_chosen − r_rejected` | grows steadily; a blowup means overfitting |
#
# **Accuracy is the metric that matters.** 0.5 = random. Above ~0.8 on held-out
# data usually means a shortcut, not understanding — read on.

# %% [markdown]
# ## Reward hacking: watch it happen
#
# This is the most important part of the notebook.
#
# **Goodhart's law:** when a measure becomes a target, it ceases to be a good
# measure. Your reward model is a *proxy* for human preference. Optimize against
# it hard enough and the policy will find inputs where the proxy is high and
# actual quality is not.
#
# The classic, ubiquitous example is **length bias**. Human annotators mildly
# prefer thorough answers. The reward model learns "longer = better" because
# it's the easiest available correlation. Then the policy learns to pad.

# %%
# Measure the length bias present in a real preference dataset.
n_longer = n_total = 0
len_diffs = []
for row in load_dataset("HuggingFaceH4/ultrafeedback_binarized",
                        split="train_prefs", streaming=True).take(1500):
    lc = len(row["chosen"][-1]["content"])
    lr = len(row["rejected"][-1]["content"])
    n_longer += lc > lr
    n_total += 1
    len_diffs.append(lc - lr)

import statistics

print(f"chosen is longer than rejected in {100*n_longer/n_total:.1f}% of pairs")
print(f"(50% would mean no length signal at all)")
print(f"mean length difference: {statistics.mean(len_diffs):+.0f} chars")
print(f"median:                 {statistics.median(len_diffs):+.0f} chars")

# %% [markdown]
# A "guess longer" classifier would score well above 50% on this dataset **with
# zero understanding of quality.** Your reward model absolutely will find that
# shortcut, because it's the cheapest way to reduce the loss.
#
# Let's prove the shortcut is learnable by fitting a model that sees *only*
# length.

# %%
class LengthOnlyRewardModel(nn.Module):
    """A deliberately stupid 'reward model' with a single feature: length."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, lengths: torch.Tensor) -> torch.Tensor:
        return self.w * (lengths / 1000.0) + self.b


rows = list(load_dataset("HuggingFaceH4/ultrafeedback_binarized",
                         split="train_prefs", streaming=True).take(3000))
len_c = torch.tensor([float(len(r["chosen"][-1]["content"])) for r in rows])
len_r = torch.tensor([float(len(r["rejected"][-1]["content"])) for r in rows])

toy = LengthOnlyRewardModel()
opt = torch.optim.Adam(toy.parameters(), lr=0.05)
for it in range(400):
    loss = bradley_terry_loss(toy(len_c), toy(len_r))
    opt.zero_grad(); loss.backward(); opt.step()

acc = (toy(len_c) > toy(len_r)).float().mean().item()
print(f"length-only 'reward model':")
print(f"  learned weight: {toy.w.item():+.3f}  (positive => longer is rewarded)")
print(f"  final BT loss:  {loss.item():.4f}   (0.693 = chance)")
print(f"  accuracy:       {acc:.3f}")
print(f"\nA model with ZERO understanding of content gets {100*acc:.0f}%.")
print("That is the floor your real reward model must beat to be worth anything.")

# %% [markdown]
# ### Defences against reward hacking
#
# | defence | how |
# |---|---|
# | **KL penalty** | penalize drift from the SFT policy — the main tool, see notebook 12 |
# | **length normalization** | subtract a length term from the reward, or use length-controlled win rates |
# | **reward model ensembles** | average several RMs; hacks rarely transfer across all |
# | **early stopping** | monitor the *true* objective, stop before the proxy diverges from it |
# | **verifiable rewards (RLVR)** | replace the learned RM with an exact checker — notebook 12 |
#
# That last row is why RLVR has taken over for math and code. If you can
# *verify* correctness, you don't need a hackable learned proxy at all. It's the
# single biggest idea in recent post-training.

# %% [markdown]
# ## Using TRL's RewardTrainer
#
# ```python
# from trl import RewardTrainer, RewardConfig
# from transformers import AutoModelForSequenceClassification
#
# model = AutoModelForSequenceClassification.from_pretrained(
#     "Qwen/Qwen2.5-0.5B-Instruct",
#     num_labels=1,              # <- this makes it a reward model
#     dtype=torch.bfloat16,
# )
#
# trainer = RewardTrainer(
#     model=model,
#     args=RewardConfig(
#         output_dir="../artifacts/reward-model",
#         per_device_train_batch_size=4,
#         gradient_accumulation_steps=4,
#         learning_rate=1e-5,
#         num_train_epochs=1,
#         max_length=1024,
#         center_rewards_coefficient=0.01,   # keeps rewards near 0; helps PPO stability
#         bf16=True,
#         report_to="none",
#     ),
#     train_dataset=load_dataset("HuggingFaceH4/ultrafeedback_binarized",
#                                split="train_prefs[:8000]"),
#     processing_class=tokenizer,
#     peft_config=LoraConfig(r=16, lora_alpha=32, task_type="SEQ_CLS"),
# )
# trainer.train()
# ```
#
# TRL expects columns named `chosen` and `rejected`. `num_labels=1` is what
# turns a classifier head into a scalar reward head.
#
# ## Exercises
#
# 1. **Beat the length baseline.** Train a real RM and compare its held-out
#    accuracy to the length-only number above. If it doesn't clearly win, it
#    hasn't learned anything content-related.
# 2. **Length-debias.** Subtract `alpha * len(response)` from the reward and
#    re-measure accuracy. How much accuracy is *pure* length?
# 3. **Reward distribution.** Plot histograms of `r_chosen` and `r_rejected`.
#    Overlapping distributions with a positive mean gap is what you want; a huge
#    gap means overfitting.
# 4. **Best-of-n.** Generate 8 samples from your SFT model, rank with the RM,
#    keep the best. This is the cheapest possible use of a reward model and it
#    works surprisingly well.
#
# ## Checkpoint
#
# - [ ] You can write the Bradley–Terry loss from memory
# - [ ] You know 0.693 is the random-baseline loss
# - [ ] You know why the reward is read off the last non-pad token
# - [ ] You measured the length bias yourself
#
# **Next:** `10_dpo_from_scratch.ipynb` — the trick that removes the reward model
# entirely.

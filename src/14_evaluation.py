# %% [markdown]
# # 14 — Evaluation: Knowing Whether Any of It Worked
#
# **Goal:** measure your models properly — perplexity, benchmarks, LLM-as-judge —
# and learn to recognize the many ways evaluation lies to you.
#
# **Time:** 45–60 min.
#
# ## In plain language
#
# **What you're doing:** learning to find out whether your changes actually
# helped — and, mostly, learning the many ways you'll accidentally lie to
# yourself.
#
# **The everyday version.** You changed a recipe. Is it better?
#
# The tempting answer is to taste it yourself and decide. But you *made* it — you
# want it to be better, you know which one is the new one, and you tasted the old
# one last week. Every one of those biases your answer.
#
# What you'd actually need: serve both to people who don't know which is which,
# enough people that it isn't luck, and check they didn't just prefer the one
# served warmer.
#
# **Everyone skips this**, and it's why most claimed improvements aren't real.
#
# **Why this is the most important notebook in the course.** Every earlier
# notebook has knobs — learning rate, LoRA rank, beta, temperature. Without
# trustworthy measurement you're turning knobs in the dark and shipping
# regressions confidently. Evaluation is what converts fiddling into progress.
#
# **What you'll have at the end:** four ways to measure a model, and — more
# useful — a clear sense of when each one is lying.
#
# **The single most important idea, if you read nothing else.** Suppose you test
# on 200 questions and score 71%. Run the *same* model again with a different
# random seed and you might get 68% or 74%. That spread isn't the model changing;
# it's noise.
#
# So if your shiny new method scores 73% against the old one's 71%, **you have
# learned nothing.** The gap is smaller than the noise. To claim that 2 points
# you'd need thousands of examples, or several runs, or both.
#
# There's a formula for how much noise to expect, and it takes ten seconds to
# apply. Most published LLM improvements do not survive it.
#
# **What to expect emotionally:** this notebook will make you less confident
# about results you were pleased with, including your own from notebooks 11 and
# 13. That is the notebook working correctly. Being able to tell a real gain from
# a lucky one is the difference between doing engineering and doing astrology.
#
# ## Why this notebook is the most important one
#
# Every technique in notebooks 04–13 has a knob you can turn. Without
# trustworthy evaluation you are turning knobs in the dark, and you will
# confidently ship regressions.
#
# Most published LLM improvements are smaller than their error bars. By the end
# of this notebook you should be unable to fool yourself in the usual ways.
#
# ### The restaurant, closing the loop
#
# Notebook 08 opened with a chef. Everything since has been about improving
# them — your menu, a critic, diner preferences, health inspections. This
# notebook asks the only question that ever mattered: **is the food actually
# better?**
#
# And it is where the analogy gets uncomfortable, because the ways a restaurant
# fools itself map exactly onto the ways teams fool themselves about models:
#
# | in the restaurant | in ML | notebook section |
# |---|---|---|
# | asking the chef whether the food improved | reading your own training loss | Level 1 |
# | serving three friends and calling it a survey | evaluating on 50 examples | statistical significance |
# | the critic you trained also grades the final | LLM-as-judge with the same model family | Level 3 |
# | the exam questions were on the practice sheet | benchmark contamination in pretraining | contamination |
# | perfecting carbonara while the pizza got worse | task metric up, everything else degraded | Level 4 |
#
# Each of those has a section below. None is hypothetical — all five are routine
# in published work.
#
# ### The four levels, and what each is actually for
#
# Evaluation is not one thing. These are four different instruments, and using
# the wrong one is most of how people go wrong:
#
# | level | measures | trust it for | do not trust it for |
# |---|---|---|---|
# | **1. Perplexity** | how well the model predicts held-out text | pretraining progress; catching regressions | anything after SFT — a chatty model has *worse* perplexity and is more useful |
# | **2. Benchmarks** | accuracy on fixed question sets | comparing against published numbers | your specific task; anything contaminated |
# | **3. LLM-as-judge** | a strong model's preference | open-ended quality at scale | small gaps — judges have position and verbosity biases |
# | **4. Your own task eval** | does it do *your* job | the only thing that decides shipping | comparing to anyone else's model |
#
# The progression is deliberate: cheap and general at the top, expensive and
# specific at the bottom. **Level 4 is the one that decides whether you ship.**
# The first three exist to catch problems early and cheaply, not to make the
# decision.
#
# If you take one habit from this notebook, take this: **always evaluate the
# thing you did not train on.** Notebook 08's retention axis, measured here.

# %%
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# %% [markdown]
# ## Level 1 — Perplexity
#
# `PPL = exp(mean cross-entropy)`. The model's average uncertainty per token.
#
# **What it's good for:** comparing checkpoints *of the same model* on the *same
# data* during pretraining. It's cheap, dense, and reliable for that.
#
# **What it cannot do:** compare models with different tokenizers. A model with
# a bigger vocabulary has fewer, harder-to-predict tokens, so its perplexity is
# not comparable. Perplexity also correlates only loosely with usefulness — a
# model can have great perplexity and be a poor assistant.

# %%
@torch.no_grad()
def compute_perplexity(model, token_ids: np.ndarray, block_size: int,
                       stride: int | None = None, batch_size: int = 8) -> dict:
    """Sliding-window perplexity over a flat token array."""
    stride = stride or block_size
    model.eval()

    nlls, n_tokens = [], 0
    windows = []
    for start in range(0, len(token_ids) - block_size - 1, stride):
        windows.append(start)

    for i in range(0, len(windows), batch_size):
        chunk = windows[i : i + batch_size]
        x = torch.tensor(
            np.stack([token_ids[s : s + block_size].astype(np.int64) for s in chunk])
        ).to(device)
        y = torch.tensor(
            np.stack([token_ids[s + 1 : s + 1 + block_size].astype(np.int64) for s in chunk])
        ).to(device)

        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        )
        nlls.append(loss.item())
        n_tokens += y.numel()

    mean_nll = sum(nlls) / n_tokens
    return {"loss": mean_nll, "perplexity": math.exp(min(mean_nll, 20)), "n_tokens": n_tokens}


# %% [markdown]
# ### The stride trick
#
# With `stride == block_size` the windows don't overlap, so **the first token of
# every window is predicted with zero context**. That inflates perplexity.
#
# Using a smaller stride gives every token more context at the cost of more
# compute. When you compare numbers to a paper, check which they used — it
# changes the result by a meaningful margin.

# %%
# Illustration of the effect (needs a trained checkpoint from notebook 04):
#
# import sys; sys.path.insert(0, "..")
# from llmfs.model import GPT, GPTConfig
# ck = torch.load("../artifacts/checkpoints/fineweb_124m.pt", weights_only=False)
# model = GPT(GPTConfig(**ck["model_config"])).to(device); model.load_state_dict(ck["model"])
# val = np.memmap("../data/fineweb_edu_train_split_val.bin", dtype=np.uint16, mode="r")
#
# for stride in [1024, 512, 256]:
#     r = compute_perplexity(model, np.array(val[:200_000]), 1024, stride=stride)
#     print(f"stride {stride:>5}: ppl {r['perplexity']:7.2f}  (loss {r['loss']:.4f})")

# %% [markdown]
# ## Level 2 — Benchmarks
#
# The standard suite, and what each actually measures:
#
# | benchmark | measures | format | notes |
# |---|---|---|---|
# | **MMLU** | broad knowledge, 57 subjects | 4-way multiple choice | heavily contaminated by now |
# | **HellaSwag** | commonsense continuation | 4-way MC | good for small models |
# | **ARC-Easy/Challenge** | science questions | 4-way MC | |
# | **GSM8K** | grade-school math | free-form + `####` | **verifiable** |
# | **MATH** | competition math | free-form | much harder |
# | **HumanEval / MBPP** | code | execute unit tests | **verifiable** |
# | **IFEval** | instruction following | programmatic checks | **verifiable**, underrated |
# | **TruthfulQA** | resists common falsehoods | MC or generative | |
# | **MT-Bench / Arena-Hard** | open-ended chat | LLM judge | see the caveats below |
#
# **For a 124M model, only HellaSwag and ARC-Easy will show signal.** MMLU will
# sit at chance (25%) and that tells you nothing. Pick benchmarks matched to the
# capability level you're actually testing.

# %% [markdown]
# ### How multiple-choice evaluation actually works
#
# Not by asking the model to output "A". You score each option by its
# log-probability under the model and take the argmax. Three normalizations are
# in common use, and they give **different answers**:

# %%
def score_choices(model, tokenizer, context: str, choices: list[str]) -> dict:
    """Score each choice three ways. Returns per-method argmax."""
    ctx_ids = tokenizer.encode(context)
    raw, per_token, per_char = [], [], []

    for choice in choices:
        full = tokenizer.encode(context + choice)
        cont = full[len(ctx_ids):]
        if not cont:
            raw.append(-1e9); per_token.append(-1e9); per_char.append(-1e9)
            continue

        x = torch.tensor([full], device=device)
        with torch.no_grad():
            logits, _ = model(x)
        logps = torch.log_softmax(logits[0, :-1], dim=-1)
        tgt = torch.tensor(full[1:], device=device)
        tok_lp = logps[torch.arange(len(tgt)), tgt]

        # Only the continuation counts — the context is identical across choices.
        cont_lp = tok_lp[len(ctx_ids) - 1:].sum().item()
        raw.append(cont_lp)
        per_token.append(cont_lp / len(cont))
        per_char.append(cont_lp / max(len(choice), 1))

    return {
        "raw": int(np.argmax(raw)),
        "per_token": int(np.argmax(per_token)),
        "per_char": int(np.argmax(per_char)),
        "scores_raw": raw,
    }


# %% [markdown]
# **Why normalization matters:** raw summed log-prob favours *short* answers,
# because every extra token adds negative log-probability. If one option is "Yes"
# and another is a 30-word sentence, raw scoring picks "Yes" almost regardless of
# content.
#
# Length-normalized (per-token or per-char) scoring corrects this. HellaSwag's
# standard metric is `acc_norm` (byte-length normalized) for exactly this reason
# — and **`acc` vs `acc_norm` can differ by 10+ points**, which is a favourite
# way to accidentally (or deliberately) inflate a result.
#
# Demonstrate the bias with synthetic scores:

# %%
choices_demo = ["Yes.", "The answer depends on several interacting factors."]
# Suppose the model assigns roughly -1.0 nats per token to both.
for name, lp_per_tok in [("equally likely per token", [-1.0, -1.0])]:
    n_tok = [2, 9]
    raw = [lp * n for lp, n in zip(lp_per_tok, n_tok)]
    norm = lp_per_tok
    print(f"{name}:")
    print(f"  raw totals:      {[f'{r:.1f}' for r in raw]}  -> picks '{choices_demo[int(np.argmax(raw))][:20]}'")
    print(f"  per-token:       {[f'{r:.1f}' for r in norm]}  -> tie (correct)")
print("\nRaw scoring systematically prefers the shorter option. Always report")
print("which normalization you used.")

# %% [markdown]
# ## The standard harnesses — use these, don't hand-roll
#
# Hand-rolled benchmark code is where fake numbers come from. Two maintained
# options:
#
# ```bash
# # EleutherAI lm-evaluation-harness — the de-facto standard
# pip install lm-eval
#
# lm_eval --model hf \
#   --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,dtype=bfloat16 \
#   --tasks hellaswag,arc_easy,gsm8k \
#   --device cuda:0 --batch_size 8 \
#   --output_path results/
#
# # HuggingFace lighteval — powers the Open LLM Leaderboard
# pip install lighteval
# lighteval accelerate \
#   --model_args "pretrained=Qwen/Qwen2.5-0.5B-Instruct" \
#   --tasks "leaderboard|hellaswag|0|0"
# ```
#
# **Always report the harness, version, and n-shot setting.** The same model
# scores differently across harnesses; comparisons across sources are usually
# invalid.

# %% [markdown]
# ## Level 3 — LLM-as-judge
#
# For open-ended quality there's no ground truth, so a stronger model judges.
# This is standard practice (MT-Bench, AlpacaEval, Arena-Hard) and it is **full
# of biases you must control for**:
#
# | bias | effect | mitigation |
# |---|---|---|
# | **position** | prefers whichever answer is shown first | evaluate both orders, average |
# | **length** | prefers longer answers | length-controlled win rate |
# | **self-preference** | prefers its own family's outputs | use a different judge family |
# | **style over substance** | prefers confident, formatted, listy text | rubric with explicit criteria |
# | **sycophancy** | agrees with assertions in the prompt | neutral prompt wording |

# %%
JUDGE_PROMPT = """You are comparing two AI assistant responses to the same question.

Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Judge on: correctness first, then helpfulness, then clarity.
Ignore length and formatting differences. A longer answer is NOT automatically
better. Verify any factual or numeric claims.

Reply with exactly one of: A, B, or TIE. Then one sentence of justification."""


def judged_win_rate(pairs: list[tuple[str, str, str]], judge_fn) -> dict:
    """pairs: [(question, response_a, response_b)].

    Each pair is judged TWICE with the responses swapped. This is the single
    most important control — position bias alone can swing results by 10-20%.
    """
    wins_a = wins_b = ties = inconsistent = 0

    for q, a, b in pairs:
        v1 = judge_fn(JUDGE_PROMPT.format(question=q, response_a=a, response_b=b))
        # Swapped order: A and B change places, so a "A" verdict now means b won.
        v2 = judge_fn(JUDGE_PROMPT.format(question=q, response_a=b, response_b=a))
        v2 = {"A": "B", "B": "A", "TIE": "TIE"}.get(v2, "TIE")

        if v1 != v2:
            inconsistent += 1
            ties += 1          # disagreement across orders = no real preference
        elif v1 == "A":
            wins_a += 1
        elif v1 == "B":
            wins_b += 1
        else:
            ties += 1

    n = len(pairs)
    return {
        "win_rate_a": wins_a / n,
        "win_rate_b": wins_b / n,
        "tie_rate": ties / n,
        # If this is high, your judge is unreliable and the whole result is soft.
        "position_inconsistency": inconsistent / n,
    }


# Simulate a judge with a strong position bias to show what the control catches.
import random

def biased_judge(prompt: str) -> str:
    return "A" if random.random() < 0.65 else "B"   # 65% first-position preference


random.seed(0)
fake_pairs = [(f"question {i}", "response one", "response two") for i in range(200)]
result = judged_win_rate(fake_pairs, biased_judge)

print("judging two IDENTICAL-quality response sets with a position-biased judge:\n")
for k, v in result.items():
    print(f"  {k:<26} {v:.3f}")
print("\nTrue answer is 50/50. Without the swap control you'd have reported a")
print(f"~65% win rate. The `position_inconsistency` of {result['position_inconsistency']:.2f}")
print("is the tell: the judge disagrees with itself that often.")

# %% [markdown]
# ## Contamination: the reason to distrust benchmark scores
#
# If a benchmark's test set appeared in the pretraining corpus, the model has
# memorized the answers. Since most models train on scraped web data and most
# benchmarks are published on the web, **assume contamination unless proven
# otherwise.**
#
# A cheap, useful check: does the model assign implausibly high probability to
# the exact test text?

# %%
@torch.no_grad()
def contamination_score(model, tokenizer, test_text: str,
                        reference_texts: list[str]) -> dict:
    """Compare per-token loss on a test item vs comparable unseen text.

    Much lower loss on the benchmark item than on similar text is evidence of
    memorization. This is a smell test, not proof.
    """
    def mean_loss(text: str) -> float:
        ids = torch.tensor([tokenizer.encode(text)], device=device)
        if ids.shape[1] < 2:
            return float("nan")
        logits, _ = model(ids[:, :-1])
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), ids[0, 1:].reshape(-1)
        ).item()

    test_loss = mean_loss(test_text)
    ref_losses = [mean_loss(t) for t in reference_texts]
    ref_mean = float(np.nanmean(ref_losses))
    return {
        "test_loss": test_loss,
        "reference_mean_loss": ref_mean,
        "ratio": test_loss / ref_mean if ref_mean else float("nan"),
        "suspicious": test_loss < 0.6 * ref_mean,
    }


print("interpretation:")
print("  ratio ~1.0  -> test item looks like ordinary unseen text (good)")
print("  ratio <0.6  -> model finds the test item far too predictable")
print("                 (likely memorized; treat the benchmark score as invalid)")

# %% [markdown]
# **Other contamination checks:**
#
# - **N-gram overlap.** Search your training corpus for 13-grams from the test
#   set (the GPT-3 paper's method).
# - **Canary strings.** Some benchmarks embed a unique GUID; if the model can
#   complete it, it saw the file.
# - **Order sensitivity.** Shuffle multiple-choice options. A memorized model
#   often gets *worse*, because it memorized the letter, not the content.
#
# ## Level 4 — The evaluation that actually matters
#
# For anything you plan to use: **build a small task-specific eval set of 50–200
# examples from your real use case.** Write the inputs yourself. Grade the
# outputs yourself the first few times.
#
# This beats every public benchmark for deciding whether a change helped, because
# it measures the thing you actually care about and it cannot be contaminated.

# %%
def build_eval_harness(cases: list[dict], grade_fn) -> callable:
    """cases: [{'input':..., 'expected':..., 'tags': [...]}, ...]"""

    def run(model_fn) -> dict:
        results, by_tag = [], {}
        for c in cases:
            output = model_fn(c["input"])
            score = grade_fn(output, c["expected"])
            results.append({"input": c["input"], "output": output, "score": score})
            for tag in c.get("tags", ["untagged"]):
                by_tag.setdefault(tag, []).append(score)

        overall = float(np.mean([r["score"] for r in results]))
        n = len(results)
        # Wilson-ish 95% interval — report this, not a bare number.
        se = math.sqrt(max(overall * (1 - overall), 1e-9) / n)
        return {
            "overall": overall,
            "ci95": (max(0.0, overall - 1.96 * se), min(1.0, overall + 1.96 * se)),
            "n": n,
            "by_tag": {t: float(np.mean(s)) for t, s in by_tag.items()},
            "failures": [r for r in results if r["score"] < 0.5][:10],
        }

    return run


# Demo with an exact-match grader.
demo_cases = [
    {"input": "2+2", "expected": "4", "tags": ["arithmetic"]},
    {"input": "capital of France", "expected": "Paris", "tags": ["knowledge"]},
    {"input": "3*7", "expected": "21", "tags": ["arithmetic"]},
    {"input": "capital of Japan", "expected": "Tokyo", "tags": ["knowledge"]},
]
grade = lambda out, exp: float(exp.lower() in out.lower())
harness = build_eval_harness(demo_cases, grade)

mock_model = lambda x: {"2+2": "4", "capital of France": "Paris",
                        "3*7": "22", "capital of Japan": "Tokyo"}[x]
r = harness(mock_model)
print(f"overall {r['overall']:.2f}  95% CI [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]  n={r['n']}")
print(f"by tag: {r['by_tag']}")
print(f"failures: {[f['input'] for f in r['failures']]}")

# %% [markdown]
# Note how wide that confidence interval is at n=4. **Always report the
# interval.** It is the fastest way to stop yourself over-interpreting a small
# eval.

# %% [markdown]
# ## Statistical significance, again
#
# Benchmarks are proportions, so the standard error is
# `sqrt(p(1-p)/n)`. At the usual sizes:
#
# | n | SE at p=0.5 | 95% CI width | smallest detectable difference |
# |---|---|---|---|
# | 100 | 5.0% | ±9.8% | ~20 pp |
# | 500 | 2.2% | ±4.4% | ~9 pp |
# | 1319 (GSM8K test) | 1.4% | ±2.7% | ~5.5 pp |
# | 14042 (MMLU) | 0.4% | ±0.8% | ~1.7 pp |
#
# **A 2-point gain on a 500-item benchmark is noise** — you'd need ~9 points to
# distinguish it from chance. This single table would invalidate a large
# fraction of the improvement claims you'll read.
#
# (The last column is for detecting a difference between *two* measured
# systems, so it carries the variance of both — that's why it's roughly 4×
# the standard error rather than 2×.)

# %%
def min_detectable_difference(n: int, p: float = 0.5, alpha: float = 0.05) -> float:
    """Smallest difference detectable at ~80% power."""
    se = math.sqrt(2 * p * (1 - p) / n)
    return (1.96 + 0.84) * se


print(f"{'n':>8}{'SE':>9}{'min detectable diff':>22}")
print("-" * 39)
for n in [100, 200, 500, 1000, 1319, 14042]:
    print(f"{n:>8}{100*math.sqrt(0.25/n):>8.1f}%{100*min_detectable_difference(n):>21.1f}pp")

# %% [markdown]
# ## An evaluation checklist
#
# Before believing any result — including your own:
#
# - [ ] Same decoding settings across all compared models
# - [ ] Same prompt template and n-shot count
# - [ ] Test split only (never the split you trained on)
# - [ ] n large enough that the effect exceeds the min detectable difference
# - [ ] Confidence intervals reported
# - [ ] For MC: state whether `acc` or `acc_norm`
# - [ ] For judges: both orders evaluated, inconsistency reported
# - [ ] For judges: length controlled
# - [ ] Contamination considered
# - [ ] Harness name and version recorded
# - [ ] Multiple seeds, if the result is close
#
# ## Exercises
#
# 1. **Track perplexity across checkpoints** from notebook 04 and plot it against
#    tokens seen on a log axis.
# 2. **Run `lm_eval`** on your SFT model vs the base model, on `hellaswag` and
#    `arc_easy`. Are the differences bigger than the CIs?
# 3. **Measure judge position bias** with a real judge model, feeding it two
#    copies of the *same* response. Any deviation from 100% TIE is pure bias.
# 4. **Build your own eval set** of 50 cases for a task you care about. Use it to
#    decide every subsequent change.
#
# **Next:** `15_inference_and_capstone.ipynb` — make it fast, make it small, ship
# it.

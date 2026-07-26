# Build an LLM From Scratch — a hands-on course for one RTX 5090

Sixteen runnable notebooks that take you from raw text on Hugging Face to a
served, quantized, post-trained model. Every stage is built **twice**: once from
scratch in plain PyTorch so you understand the mechanism, then again with the
production library you'd actually use at work.

Sized specifically for **one 24 GB consumer GPU** (RTX 5090, WSL2). Nothing here
asks for a cluster.

---

## Start here

### 0. Clone onto the Linux filesystem

On WSL2 this is not a style preference. `/mnt/c/...` and `/mnt/d/...` are
Windows drives reached through the 9p mount, which is **roughly 10× slower** for
the many-small-files I/O that `pip` and `datasets` do. Notebook 01 writes
gigabytes; put the repo on the Linux side and it stays fast.

From a **WSL bash** shell (Windows Terminal → Ubuntu, or `wsl` from PowerShell):

```bash
git clone https://github.com/narinzar/llm-from-scratch ~/llm-from-scratch
cd ~/llm-from-scratch
pwd            # must print /home/<you>/... and NOT start with /mnt/
```

Already cloned to a Windows path? Move it rather than re-cloning, then throw
away the venv — a venv hard-codes absolute paths and never survives a move:

```bash
mv /mnt/d/vscode_projects/llm-from-scratch ~/ && cd ~/llm-from-scratch && rm -rf .venv
```

### 0b. Open VS Code *in* WSL, not next to it

This is the step that trips people up, and it silently costs you the 10× above.

**Launch VS Code from the bash shell you just used:**

```bash
cd ~/llm-from-scratch && code .
```

That opens a window already attached to WSL, pointed at the Linux path. Do **not**
use *File → Open Folder* to get there: that dialog is native Windows, so browsing
to `\\wsl$\...` or picking a `D:\` folder drops the window back into Windows mode.
Once a window is correct, *File → Open Recent* remembers it — a good entry looks
like `llm-from-scratch [WSL: Ubuntu-22.04]`.

Three checks that you're in the right place:

| Check | Right | Wrong |
|---|---|---|
| Status bar, bottom-left | `WSL: Ubuntu-22.04` **and** files in Explorer | no WSL badge, or badge with an empty Explorer |
| Terminal profile | `bash` | `powershell` / `pwsh` |
| `pwd` in the terminal | `/home/<you>/llm-from-scratch` | `/mnt/d/...` or `D:\...` |

If the terminal offers you PowerShell at all, that window is not in WSL — reopen
with `code .` from bash. Everything below assumes **bash inside WSL**; PowerShell
would build a Windows venv your Linux notebooks cannot use.

### 1. Create and activate the venv

```bash
python3 -m venv .venv
source .venv/bin/activate          # prompt gains a (.venv) prefix
python -m pip install -U pip wheel
```

`source .venv/bin/activate` is the whole activation step, and it has to be
rerun in **every new terminal**. In VS Code, do it once properly instead:
**Ctrl+Shift+P → Python: Select Interpreter → `./.venv/bin/python`**. New
terminals then activate automatically, and Jupyter picks the same environment.

### 2. Install PyTorch first, on its own

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**The order matters and the `--index-url` is not optional.** `requirements.txt`
deliberately leaves torch out, but `trl`, `peft` and `lm-eval` all depend on it
— so if you run `pip install -r requirements.txt` first, pip quietly resolves
torch from PyPI and hands you a build with **no sm_120 kernels**. Installing it
first means the requirements step finds torch already satisfied and leaves it
alone.

Expect this to take a while: torch plus its bundled cuDNN/cuBLAS libraries is a
**3–5 GB download**, so 10–15 minutes on a normal connection is right. It is not
hung.

Check the wheel you actually got before going further:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
```

`sm_120` must appear in that list. If it doesn't, see
[SETUP.md](SETUP.md) — you have the wrong wheel, and every notebook past 03
will fail in the same confusing way.

### 3. Everything else

```bash
pip install -r requirements.txt
python -m ipykernel install --user --name llm-fs --display-name "llm-from-scratch"

jupyter lab notebooks/
```

Then open **`notebooks/00_setup_and_hardware_check.ipynb`** and run it. It
verifies your GPU actually computes (not just that CUDA "is available") and
teaches you to budget VRAM before you waste an evening on a run that can't fit.

> **RTX 50-series users:** a stock `pip install torch` gives you a build with no
> sm_120 kernels. It will report `cuda.is_available() == True` and then die on
> the first real matmul with `no kernel image is available`. Notebook 00 catches
> this in the first cell.

---

## The curriculum

Notebooks build on each other. Run them in order the first time.

### Foundations — build a language model

| # | notebook | what you build | time |
|---|---|---|---|
| 00 | Setup & hardware check | working GPU stack, a VRAM budget model | 30 min |
| 01 | Data from Hugging Face | filtered, deduped 500M-token corpus from FineWeb-Edu | 60 min |
| 02 | Tokenizer from scratch | byte-level BPE, trained and verified | 45 min |
| 03 | Transformer from scratch | attention → multi-head → GPT, 124M params | 90 min |
| 04 | **Pretraining** | a model that writes coherent English | 15 min + 3–5 h |
| 05 | Modern architecture | RoPE, RMSNorm, SwiGLU, GQA (GPT-2 → Llama) | 60 min |
| 06 | Scaling & efficiency | Chinchilla math, MFU, `torch.compile`, Muon | 45 min |

### Post-training — make it useful

| # | notebook | what you build | time |
|---|---|---|---|
| 07 | SFT from scratch | chat templates, −100 loss masking, packing | 60 min |
| 08 | SFT with TRL + LoRA | LoRA derived by hand, then QLoRA on a real model | 90 min |
| 09 | Reward modeling | Bradley–Terry RM, and measured reward hacking | 45 min |
| 10 | DPO from scratch | the derivation, the loss, the degeneration failure | 60 min |
| 11 | DPO with TRL | production preference tuning + how to read the metrics | 60 min |
| 12 | GRPO & RLVR from scratch | group advantages, clipped objective, verifiers | 75 min |
| 13 | **GRPO on GSM8K** | real RLVR run; teach a model to do math | 2–5 h |

### Ship it

| # | notebook | what you build | time |
|---|---|---|---|
| 14 | **Evaluation** | perplexity, benchmarks, LLM-judge, contamination, statistics | 60 min |
| 15 | Inference & capstone | KV cache, sampling, quantization, serving, the capstone | 60 min |

**If you only read two notebooks, read 04 and 14.** Pretraining is where it
becomes real, and evaluation is what stops you fooling yourself.

---

## What you'll actually end up with

- A **124M-parameter model** you pretrained yourself on FineWeb-Edu, which
  writes fluent English (~3.8 val loss after an afternoon).
- A **fine-tuned assistant** (Qwen2.5-0.5B/1.5B + LoRA) that follows
  instructions.
- A **GSM8K math specialist** trained with GRPO, with a measured, statistically
  tested improvement over its baseline.
- The ability to read `nanochat`, `TRL`, or a post-training paper and know
  what's going on.

### Honest expectations

You are not going to reproduce GPT-4, or R1, on one GPU. What you *will* do is
run every mechanism those systems use, at a scale where you can see the whole
thing. A 124M model trained for an afternoon writes real English and knows
almost no facts — that's a capacity limit, not a mistake you made. The notebooks
tell you what each stage should and shouldn't produce so you can tell the
difference between "working as expected" and "broken".

---

## Hardware

Built and sized for **RTX 5090 (24 GB) / 64 GB RAM / Ubuntu on WSL2**.

| you have | what changes |
|---|---|
| 24 GB NVIDIA | everything runs as written |
| 12–16 GB | halve `micro_batch`, double `grad_accum`; use QLoRA from notebook 08 on |
| 8 GB | pretrain at `n_embd=384`; QLoRA only for post-training |
| Apple Silicon | notebooks 01–07 work on MPS; 08+ need CUDA-only libs (bitsandbytes, vLLM) |
| CPU only | 01–03 and 05–07 are fine; pretraining is limited to TinyStories-scale |
| Colab T4/L4 | works; checkpoint to Drive, the notebooks resume |

The rule when you OOM, at every stage: **halve the micro-batch and double the
gradient accumulation.** The math is unchanged.

---

## How this repo is organized

```
llm-from-scratch/
├── notebooks/     <- the course. open these.
├── src/           <- jupytext percent-format sources the notebooks are built from
├── llmfs/         <- shared model code (the GPT from notebook 03) + bench.py
├── tools/         <- build_notebooks.py
├── benchmarks/    <- runs.jsonl, the append-only record of every run
├── BENCHMARK.md   <- generated results table (see below)
└── SETUP.md       <- Blackwell/WSL2 setup and troubleshooting
```

`data/` and `artifacts/` are created on first use and are gitignored — corpora,
tokenizers and checkpoints are all reproducible from the notebooks, so none of
it belongs in git.

Notebooks are **generated** from `src/*.py`:

```bash
python tools/build_notebooks.py        # rebuild all
python tools/build_notebooks.py 04     # rebuild just notebook 04
```

Notebook JSON produces unreadable git diffs, so the reviewable source lives in
`src/`. If you edit a notebook directly and want to keep the change, mirror it
into `src/` or your next rebuild will overwrite it.

---

## Tracking your results

Almost every exercise in this course is "change one thing and see what happens" —
sweep beta, drop the KL term, untie the weights, resize the vocabulary. That only
teaches you something if you can see the *previous* number next to the new one.

So each notebook ends by recording its results:

```python
from llmfs.bench import log_run

log_run(
    stage="10_dpo_from_scratch",
    metrics={"margin": 2.89, "loss": 0.055, "logp_chosen": -52.9},
    config={"beta": 0.5, "lr": 1e-5, "steps": 400},
    notes="beta sweep: 0.1 -> 0.5",
)
```

That appends one line to `benchmarks/runs.jsonl` and regenerates
**[`BENCHMARK.md`](BENCHMARK.md)** — a summary of the latest result per stage
plus the full history per stage, each row carrying its config, git commit, and
device. Reruns show a delta against your previous attempt, labelled better or
worse:

| # | `margin` | `loss` | Config | Notes |
|---|---|---|---|---|
| 2 | 2.886 | 0.0545 | `beta=0.5` | beta sweep: 0.1 -> 0.5 |
| 1 | 0.7249 | 0.3952 | `beta=0.1` | baseline |

Nothing is automatic beyond that: run a notebook, the table updates. Commit
`runs.jsonl` and your results travel with the repo.

```bash
python -m llmfs.bench            # rebuild the table by hand
python -m llmfs.bench --check    # exit 1 if it's stale (for CI)
```

`runs.jsonl` is append-only and is the source of truth; `BENCHMARK.md` is
derived and can always be rebuilt from it. Numbers are only comparable within a
device, which is why the device is part of every row — a 5090 run and a CPU run
are different experiments.

---

## Sources and further reading

This course is a guided path through work by other people. When you finish,
read the originals:

**Code**
- [karpathy/nanochat](https://github.com/karpathy/nanochat) — full pipeline, tokenizer → RL → web UI, in one clean repo
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — the minimal pretraining classic
- [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — the speedrun; where Muon was proven
- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) — book-length version of notebooks 02–07
- [rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) — deeper on reasoning models
- [huggingface/trl](https://github.com/huggingface/trl) — the post-training library used here
- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the standard eval harness
- [volcengine/verl](https://github.com/volcengine/verl) — production RL infrastructure

**Data & models**
- [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- [HuggingFaceTB/SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — the small-model recipe this course follows
- [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)

**Papers** — Attention Is All You Need · GPT-2/3 · Chinchilla · LoRA · QLoRA ·
InstructGPT · DPO · DeepSeekMath (GRPO) · DeepSeek-R1 · FineWeb.

---

## A note on the code

Every from-scratch implementation in these notebooks was **executed and
verified** while writing them — including the parts that are supposed to fail.
Several claims were corrected because running the code contradicted them. Where
a demonstration only shows part of what it appears to show, the notebook says so
explicitly rather than overselling the result.

That habit — run it, check the number, believe the number — is the actual skill.

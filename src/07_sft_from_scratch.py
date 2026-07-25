# %% [markdown]
# # 07 — Supervised Fine-Tuning, From Scratch
#
# **Goal:** turn a base model (which only continues text) into an assistant
# (which answers questions) — implementing chat templates, loss masking, and
# packing yourself.
#
# **Time:** 45–60 min.
#
# ## Base models vs instruct models
#
# Your pretrained model from notebook 04 does exactly one thing: continue text.
# Ask it a question and you'll get something like:
#
# ```
# prompt: What is the capital of France?
# output: What is the capital of Germany? What is the capital of Spain? ...
# ```
#
# That is not a failure. It is a *correct* continuation — on the web, a question
# is most often followed by more questions. The model has no idea it's supposed
# to be helpful, because nothing ever told it.
#
# **SFT is that telling.** Same next-token objective, different data: curated
# (instruction, response) pairs, with the loss computed **only on the response**.
#
# ## The post-training pipeline
#
# ```
# base model  ──SFT──>  instruct model  ──preference──>  aligned model
#  (04)                     (07, 08)                      (09-13)
#             ^                        ^
#             |                        |
#      "behave like an           "prefer better
#       assistant"                answers"
# ```
#
# SFT teaches **format and behaviour**. Preference tuning teaches **quality**.
# You need SFT first — you can't rank responses from a model that can't produce
# responses.

# %%
import json
import math
import sys
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
# ## Step 1 — Chat templates
#
# The model needs unambiguous markers for "who is speaking" and "stop here".
# Without them it can't tell where the user's turn ends and its own begins, and
# it will happily continue writing the user's next message.
#
# **ChatML** (from OpenAI, now near-universal) looks like:
#
# ```
# <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# What is 2+2?<|im_end|>
# <|im_start|>assistant
# 4<|im_end|>
# ```
#
# The special tokens matter enormously. If `<|im_end|>` were ordinary text the
# model could generate it accidentally, or fail to generate it and never stop.

# %%
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")

SPECIAL = ["<|im_start|>", "<|im_end|>", "<|pad|>"]
n_added = tok.add_special_tokens({"additional_special_tokens": SPECIAL})
tok.pad_token = "<|pad|>"

IM_START = tok.convert_tokens_to_ids("<|im_start|>")
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
PAD = tok.convert_tokens_to_ids("<|pad|>")

print(f"added {n_added} special tokens")
print(f"  <|im_start|> = {IM_START}")
print(f"  <|im_end|>   = {IM_END}")
print(f"  <|pad|>      = {PAD}")
print(f"vocab is now {len(tok)}")

# %% [markdown]
# **Important:** adding tokens means the embedding matrix must grow. The new
# rows are randomly initialized while everything else is trained, so they start
# out as noise. A common trick is to initialize them to the **mean of the
# existing embeddings** — much closer to the trained distribution than random,
# and it converges noticeably faster.

# %%
def resize_and_init_embeddings(model: GPT, new_vocab_size: int) -> GPT:
    """Grow the embedding table, initialising new rows to the existing mean."""
    old_vocab, d = model.wte.weight.shape
    if new_vocab_size <= old_vocab:
        print(f"no resize needed ({old_vocab} >= {new_vocab_size})")
        return model

    old_w = model.wte.weight.data
    mean_emb = old_w.mean(dim=0, keepdim=True)
    # Small noise breaks the symmetry between the new rows — identical rows
    # would receive identical gradients and stay identical forever.
    noise = torch.randn(new_vocab_size - old_vocab, d, device=old_w.device) * 0.01
    new_rows = mean_emb.repeat(new_vocab_size - old_vocab, 1) + noise

    new_emb = nn.Embedding(new_vocab_size, d).to(old_w.device)
    new_emb.weight.data = torch.cat([old_w, new_rows], dim=0)
    model.wte = new_emb

    model.head = nn.Linear(d, new_vocab_size, bias=False).to(old_w.device)
    if model.cfg.tie_weights:
        model.head.weight = model.wte.weight   # re-tie after replacing both
    model.cfg.vocab_size = new_vocab_size

    print(f"resized {old_vocab} -> {new_vocab_size} "
          f"({new_vocab_size - old_vocab} new rows at the embedding mean)")
    return model


# %% [markdown]
# ## Step 2 — Format a conversation

# %%
def format_chatml(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """messages: [{'role': 'user'|'assistant'|'system', 'content': str}, ...]"""
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    if add_generation_prompt:
        # At inference, end with the assistant header so the model continues
        # AS the assistant rather than inventing another user turn.
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


convo = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]

print("--- training format (complete conversation) ---")
print(format_chatml(convo))
print("--- inference format (ends with the generation prompt) ---")
print(repr(format_chatml(convo[:2], add_generation_prompt=True)))

# %% [markdown]
# ## Step 3 — Loss masking: the heart of SFT
#
# This is the part that distinguishes SFT from plain pretraining, and the part
# most commonly implemented wrong.
#
# **Train on the assistant's tokens only.** The user's question is *given*, not
# something the model should learn to produce. If you compute loss over the
# whole sequence, you're teaching the model to generate plausible user questions
# — which at best wastes capacity and at worst makes the model interrogate you.
#
# The mechanism: set masked positions to **`-100`** in the labels.
# `F.cross_entropy` has `ignore_index=-100` by default, so those positions
# contribute nothing to the loss and nothing to the gradient.

# %%
def build_sft_example(messages: list[dict], max_len: int = 512) -> dict:
    """Tokenize a conversation, masking everything but assistant content."""
    input_ids: list[int] = []
    labels: list[int] = []

    for m in messages:
        header = tok.encode(f"<|im_start|>{m['role']}\n")
        body = tok.encode(m["content"])
        end = [IM_END] + tok.encode("\n")

        if m["role"] == "assistant":
            # Header: masked (the model doesn't choose to be the assistant).
            # Body + <|im_end|>: TRAINED. Including <|im_end|> is essential —
            # it's how the model learns to STOP. Omit it and your model will
            # ramble past the end of its answer forever.
            input_ids += header + body + end
            labels += [-100] * len(header) + body + end
        else:
            input_ids += header + body + end
            labels += [-100] * (len(header) + len(body) + len(end))

    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    return {"input_ids": input_ids, "labels": labels}


ex = build_sft_example(convo)
print(f"{'idx':>4} {'token':<22} {'label':>8}  trained?")
print("-" * 50)
for i, (t, l) in enumerate(zip(ex["input_ids"], ex["labels"])):
    if i > 42:
        print("  ...")
        break
    mark = "TRAIN" if l != -100 else "  -"
    print(f"{i:>4} {tok.decode([t])!r:<22} {l:>8}  {mark}")

n_train = sum(1 for l in ex["labels"] if l != -100)
print(f"\n{n_train} of {len(ex['labels'])} positions contribute to the loss "
      f"({100*n_train/len(ex['labels']):.0f}%)")

# %% [markdown]
# ### Why the shift is handled for you
#
# In notebook 04 we built `x` and `y` as explicit shifted slices. Here labels
# align 1:1 with inputs, and the shift happens inside the loss function:
#
# ```python
# logits[:, :-1]  predicts  labels[:, 1:]
# ```
#
# Both conventions are common. **Know which one your code uses** — mixing them
# is an off-by-one that produces a model that trains but is subtly wrong.

# %%
def sft_loss(model, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits, _ = model(input_ids)
    # Drop the last logit (nothing follows it) and the first label (nothing
    # predicts it). Now position i of shift_logits predicts shift_labels[i].
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,          # this is what makes masking work
    )


# %% [markdown]
# ## Step 4 — Padding vs packing
#
# Conversations vary in length; GPUs want rectangles. Two strategies:
#
# **Padding** — pad every sequence to the batch maximum. Simple, but if lengths
# vary a lot you can waste 50%+ of your compute on padding tokens.
#
# **Packing** — concatenate examples into fixed-length blocks with no padding at
# all. ~100% token efficiency, and the standard choice for large SFT runs.
#
# Packing's caveat: without careful attention masking, tokens can attend across
# example boundaries — example B sees example A's tokens. In practice this is
# usually tolerated (the `<|im_end|>` boundary teaches the model to ignore it),
# but "correct" packing uses block-diagonal attention. TRL exposes this as
# `packing=True` with `padding_free`/FlashAttention varlen support.

# %%
def collate_padded(batch: list[dict], pad_id: int) -> dict:
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n_pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n_pad)
        # Padding is masked from the loss too — otherwise the model learns to
        # predict <|pad|>, which is both useless and actively harmful.
        labels.append(b["labels"] + [-100] * n_pad)
        attn.append([1] * len(b["input_ids"]) + [0] * n_pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def pack_examples(examples: list[dict], block_size: int) -> list[dict]:
    """Concatenate examples into fixed-size blocks. No padding, no waste."""
    all_ids, all_labels = [], []
    for e in examples:
        all_ids.extend(e["input_ids"])
        all_labels.extend(e["labels"])

    blocks = []
    for i in range(0, len(all_ids) - block_size, block_size):
        blocks.append({
            "input_ids": all_ids[i : i + block_size],
            "labels": all_labels[i : i + block_size],
        })
    return blocks


demo = [
    build_sft_example([{"role": "user", "content": "Hi"},
                       {"role": "assistant", "content": "Hello!"}]),
    build_sft_example([{"role": "user", "content": "Explain photosynthesis in detail."},
                       {"role": "assistant", "content": "Photosynthesis is the process "
                        "by which plants convert light energy into chemical energy. " * 4}]),
    build_sft_example([{"role": "user", "content": "2+2?"},
                       {"role": "assistant", "content": "4"}]),
]

lens = [len(d["input_ids"]) for d in demo]
padded = collate_padded(demo, PAD)
real = sum(lens)
total = padded["input_ids"].numel()

print(f"example lengths: {lens}")
print(f"\npadded batch:  {tuple(padded['input_ids'].shape)} = {total} token slots")
print(f"  real tokens: {real}  ->  efficiency {100*real/total:.0f}%")

packed = pack_examples(demo, block_size=64)
print(f"\npacked into {len(packed)} blocks of 64  ->  efficiency ~100%")
print("(with very unequal lengths, padding can waste more than half your compute)")

# %% [markdown]
# ## Step 5 — SFT datasets worth knowing
#
# | dataset | size | notes |
# |---|---|---|
# | `HuggingFaceTB/smoltalk` | 1M | **Best general default.** Built for SmolLM2 |
# | `HuggingFaceTB/smol-smoltalk` | 460k | Filtered for small models — shorter, simpler |
# | `allenai/tulu-3-sft-mixture` | 940k | Tülu 3's mix; strong, well documented |
# | `teknium/OpenHermes-2.5` | 1M | Popular general-purpose mix |
# | `openai/gsm8k` | 8.8k | Grade-school math with worked solutions |
# | `HuggingFaceH4/no_robots` | 10k | **Human-written**, high quality, small |
#
# **For a small model, prefer `smol-smoltalk`.** A 124M model cannot learn from
# 2000-token expert answers on advanced mathematics; it will just learn the
# surface style. Match the data's difficulty to the model's capacity.
#
# ### Quality over quantity
#
# The LIMA paper showed 1,000 carefully curated examples beat 50,000 mediocre
# ones. SFT is teaching *format and behaviour*, and that doesn't take much data —
# but every bad example teaches something bad. Look at your data.

# %%
from datasets import load_dataset

sft_raw = load_dataset("HuggingFaceTB/smol-smoltalk", split="train", streaming=True)

print("--- 2 sample conversations ---")
samples = list(sft_raw.take(2))
for i, s in enumerate(samples):
    print(f"\n[{i}] {len(s['messages'])} messages")
    for m in s["messages"]:
        content = m["content"][:200].replace("\n", " ")
        print(f"  {m['role']:>10}: {content}{'...' if len(m['content'])>200 else ''}")

# %%
N_SFT = 20_000
MAX_LEN = 512

print(f"tokenizing {N_SFT} conversations...")
sft_examples = []
skipped = 0
for row in load_dataset("HuggingFaceTB/smol-smoltalk", split="train", streaming=True).take(N_SFT):
    msgs = row["messages"]
    if not msgs or msgs[-1]["role"] != "assistant":
        skipped += 1
        continue
    e = build_sft_example(msgs, max_len=MAX_LEN)
    # Truncation can cut off the entire assistant turn, leaving an example with
    # nothing to learn from. Those contribute a NaN-risk zero-token loss — drop
    # them rather than letting them into the batch.
    if not any(l != -100 for l in e["labels"]):
        skipped += 1
        continue
    sft_examples.append(e)

print(f"kept {len(sft_examples)}, skipped {skipped}")
tok_counts = [len(e["input_ids"]) for e in sft_examples]
trained = [sum(1 for l in e['labels'] if l != -100) for e in sft_examples]
print(f"mean length {sum(tok_counts)/len(tok_counts):.0f} tokens")
print(f"mean trained tokens {sum(trained)/len(trained):.0f} "
      f"({100*sum(trained)/sum(tok_counts):.0f}% of all tokens)")

# %% [markdown]
# That last number is worth internalizing: **only ~30–50% of your tokens carry
# gradient in SFT.** The rest is context. That's expected — but it means SFT is
# less token-efficient than pretraining, and it's another reason SFT runs are
# short.

# %% [markdown]
# ## Step 6 — The SFT training loop
#
# Nearly identical to pretraining, with three changes that matter:
#
# | | pretraining | SFT |
# |---|---|---|
# | learning rate | 6e-4 | **1e-5 to 5e-5** (10–50× lower) |
# | epochs | <1 (one pass over a huge corpus) | **1–3** |
# | loss | all tokens | assistant tokens only |
#
# **Why the much lower LR?** You are adjusting a converged model, not building
# one. A high LR causes *catastrophic forgetting* — the model learns the chat
# format while destroying the knowledge it spent hours acquiring. If your SFT'd
# model formats beautifully but has become stupid, your LR was too high.

# %%
from dataclasses import dataclass


@dataclass
class SFTConfig:
    base_ckpt: str = "tinystories_10m"
    lr: float = 2e-5
    epochs: int = 2
    batch_size: int = 8
    max_len: int = 512
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0     # usually off for short SFT runs
    grad_clip: float = 1.0
    log_every: int = 20
    out_name: str = "sft_model"


def train_sft(cfg: SFTConfig, examples: list[dict]):
    ck_path = Path("../artifacts/checkpoints") / f"{cfg.base_ckpt}.pt"
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model"])
    print(f"loaded base model from {ck_path.name} (val loss {ck['val_loss']:.3f})")

    model = resize_and_init_embeddings(model, len(tok)).to(device)

    n_steps = (len(examples) // cfg.batch_size) * cfg.epochs
    warmup = max(int(n_steps * cfg.warmup_ratio), 1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))

    print(f"\n{n_steps} steps over {cfg.epochs} epochs\n")
    print(f"{'step':>6}{'loss':>9}{'lr':>10}")
    print("-" * 25)

    model.train()
    step, running = 0, None
    for epoch in range(cfg.epochs):
        perm = torch.randperm(len(examples))
        for i in range(0, len(examples) - cfg.batch_size, cfg.batch_size):
            batch = [examples[j] for j in perm[i : i + cfg.batch_size].tolist()]
            b = collate_padded(batch, PAD)
            ids = b["input_ids"].to(device)
            lbl = b["labels"].to(device)

            lr = cfg.lr * (step + 1) / warmup if step < warmup else \
                cfg.lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(n_steps - warmup, 1)))
            for g in opt.param_groups:
                g["lr"] = lr

            with ctx:
                loss = sft_loss(model, ids, lbl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            running = loss.item() if running is None else 0.9 * running + 0.1 * loss.item()
            if step % cfg.log_every == 0:
                print(f"{step:>6}{running:>9.4f}{lr:>10.2e}")
            step += 1

    out = Path("../artifacts/checkpoints") / f"{cfg.out_name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "model_config": {**ck["model_config"], "vocab_size": len(tok)},
                "step": step, "val_loss": running}, out)
    print(f"\nsaved -> {out}")
    return model


# %%
# Uncomment once you have a base checkpoint from notebook 04.
# sft_model = train_sft(SFTConfig(), sft_examples)

# %% [markdown]
# ## Step 7 — Chat with it

# %%
@torch.no_grad()
def chat(model, user_message: str, system: str | None = None,
         max_new_tokens: int = 200, temperature: float = 0.7, top_k: int = 50) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user_message}]
    prompt = format_chatml(msgs, add_generation_prompt=True)
    ids = torch.tensor([tok.encode(prompt)], device=device)

    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(ids[:, -model.cfg.block_size :])
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        # THE stopping condition. Without it the model runs to max_new_tokens
        # and starts hallucinating the next user turn.
        if nxt.item() == IM_END:
            break
        ids = torch.cat([ids, nxt], dim=1)

    full = tok.decode(ids[0].tolist())
    return full.split("<|im_start|>assistant\n")[-1].strip()


# Example usage (after training):
# for q in ["What is 2+2?", "Tell me a story about a cat.", "What is Python?"]:
#     print(f"\nUSER: {q}\nASSISTANT: {chat(sft_model, q)}")

# %% [markdown]
# ## What to expect from a 124M SFT'd model
#
# Calibrate your expectations, or you'll conclude you did something wrong:
#
# **It will:** answer in the right format, stop at the right place, adopt an
# assistant tone, follow simple instructions.
#
# **It will not:** know many facts, do multi-step reasoning, do arithmetic
# reliably, or stay coherent past a few hundred tokens.
#
# That's a capacity limit, not a bug. SFT teaches **behaviour**, and it cannot
# add knowledge that isn't in the base model. If you want a capable assistant,
# fine-tune a capable base model — which is exactly notebook 08.
#
# ## Common SFT failure modes
#
# | symptom | cause |
# |---|---|
# | never stops generating | `<\|im_end\|>` not in the trained labels |
# | generates the user's next turn too | you masked nothing; loss over all tokens |
# | fluent but became stupid | LR too high — catastrophic forgetting |
# | ignores the system prompt | too few system-prompt examples in your data |
# | repeats the question back | assistant turn got truncated by `max_len` |
# | NaN loss on some batches | an example with zero unmasked labels |
#
# ## Exercises
#
# 1. **Break the masking.** Train with loss over all tokens. Then chat with it —
#    it should start writing your side of the conversation.
# 2. **Drop `<|im_end|>` from the labels.** Watch generation never terminate.
# 3. **LR sweep.** Try 1e-4, 2e-5, 5e-6. Track both SFT loss *and* pretraining
#    val loss. Watch the forgetting curve as LR rises.
# 4. **Packing vs padding.** Measure tokens/sec for both on the same data.
#
# ## Checkpoint
#
# - [ ] You can explain why labels are −100 on the prompt
# - [ ] You know why `<|im_end|>` must be trained
# - [ ] You know why SFT LR is 10–50× lower than pretraining
#
# **Next:** `08_sft_with_trl_and_lora.ipynb` — the same thing with production
# tooling, on a model that's actually capable.

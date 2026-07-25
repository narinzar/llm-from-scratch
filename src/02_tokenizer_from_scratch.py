# %% [markdown]
# # 02 — Tokenizer from Scratch (BPE)
#
# **Goal:** implement Byte-Pair Encoding end to end — train it, encode, decode —
# then compare against the production `tokenizers` library and understand where
# tokenization causes real model failures.
#
# **Time:** 30–45 min. **Hardware:** CPU only.
#
# ## Why tokenization deserves a whole notebook
#
# A surprising number of "the model is dumb" bugs are actually tokenizer bugs:
#
# - **Arithmetic failures.** If `1234` splits as `12|34` but `1235` splits as
#   `123|5`, the model sees no consistent digit structure. Llama and others now
#   force digits to split individually for exactly this reason.
# - **The `" strawberry"` letter-counting meme.** The model never sees letters,
#   only ~2 token ids. Asking it to count `r`s is asking it to recall a fact
#   about a string it cannot see.
# - **Trailing-whitespace bugs.** `"Hello"` + `" world"` tokenizes differently
#   from `"Hello "` + `"world"`. Prompts ending in a space often produce garbage.
# - **Non-English cost.** Some languages need 3–4× more tokens for the same
#   meaning — literally 3–4× the price and 3–4× less context.
#
# ## The problem BPE solves
#
# Two naive options, both bad:
#
# | approach | vocab | sequence length | problem |
# |---|---|---|---|
# | characters | ~100 | very long | attention is O(n²); wastes depth on spelling |
# | words | 1M+ | short | huge embedding table; every typo is `<unk>` |
#
# BPE interpolates: **start from bytes, then repeatedly merge the most frequent
# adjacent pair.** Common words become single tokens; rare words decompose into
# pieces. Nothing is ever out-of-vocabulary, because you can always fall back
# to bytes.

# %% [markdown]
# ## Step 1 — Start from bytes, not characters
#
# Modern BPE (GPT-2 onward) operates on **UTF-8 bytes**. This guarantees any
# possible string is encodable — no `<unk>` token, ever. The base vocabulary is
# exactly 256 entries.

# %%
text = "Hello, world! 你好 🚀"

print(f"as characters: {len(text)} items -> {list(text)[:8]} ...")
print(f"as bytes:      {len(text.encode('utf-8'))} items -> {list(text.encode('utf-8'))[:8]} ...")
print()
for ch in ["A", "é", "你", "🚀"]:
    b = ch.encode("utf-8")
    print(f"  {ch!r:<6} -> {len(b)} byte(s): {list(b)}")

# %% [markdown]
# Note the emoji costs 4 bytes and the Chinese character 3. This is the root of
# the "non-English is more expensive" problem: before any merges are learned,
# non-ASCII text already starts at a disadvantage.

# %% [markdown]
# ## Step 2 — The core BPE training loop
#
# The algorithm in full:
#
# 1. Represent the corpus as a list of byte values (0–255).
# 2. Count every adjacent pair.
# 3. Take the **most frequent** pair, mint a new token id for it, replace all
#    its occurrences.
# 4. Record the merge. Repeat until you hit your target vocab size.
#
# That's the whole thing. Roughly 40 lines.

# %%
from collections import Counter


def get_pair_counts(ids: list[int], counts: Counter | None = None) -> Counter:
    """Count occurrences of each adjacent pair. (1,2,3) -> {(1,2):1, (2,3):1}"""
    counts = Counter() if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        # Match the pair, but never run off the end of the list.
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# Walk through it on a toy example.
demo = list("aaabdaaabac".encode("utf-8"))
print(f"start: {demo}  (len {len(demo)})")

vocab_demo = {i: bytes([i]) for i in range(256)}
next_id = 256
for step in range(3):
    counts = get_pair_counts(demo)
    top_pair, freq = counts.most_common(1)[0]
    demo = merge(demo, top_pair, next_id)
    vocab_demo[next_id] = vocab_demo[top_pair[0]] + vocab_demo[top_pair[1]]
    print(
        f"step {step+1}: merge {top_pair} (seen {freq}x) -> id {next_id} "
        f"= {vocab_demo[next_id]!r}   now: {demo} (len {len(demo)})"
    )
    next_id += 1

# %% [markdown]
# Watch the sequence get shorter each step while the vocabulary grows. That is
# the entire trade BPE makes, made visible.

# %% [markdown]
# ## Step 3 — Pre-tokenization: the detail that makes it work
#
# Naive BPE would happily learn a token spanning `"dog."` and `" The"` — merging
# across word and punctuation boundaries. That wastes vocabulary on
# meaningless fragments.
#
# The fix: **split text into chunks with a regex first**, run BPE within each
# chunk, and never merge across chunks. GPT-2's pattern is famous and worth
# reading closely.

# %%
import regex as re  # NOT the stdlib `re` — we need \p{L} unicode property classes

GPT2_SPLIT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

sample = "Hello world! It's 2026, isn't it? Cost: $45.99 (approx)."
print("chunks:")
for c in GPT2_SPLIT.findall(sample):
    print(f"  {c!r}")

# %% [markdown]
# Decoding that pattern, alternative by alternative:
#
# | piece | matches | why |
# |---|---|---|
# | `'(?:[sdmt]\|ll\|ve\|re)` | `'s`, `'t`, `'ll`, `'ve` | English contractions stay whole |
# | ` ?\p{L}+` | ` Hello` | a word **with its leading space** |
# | ` ?\p{N}+` | ` 2026` | digit runs, separate from letters |
# | ` ?[^\s\p{L}\p{N}]+` | `!`, ` (` | punctuation runs |
# | `\s+(?!\S)` | trailing whitespace | keeps a run of spaces together |
# | `\s+` | remaining whitespace | |
#
# **The leading space is the key design choice.** `" Hello"` (with space) and
# `"Hello"` (without) are *different tokens*. This is why a prompt ending in a
# trailing space behaves badly: you've forced the model into the rare
# no-leading-space variant it saw far less during training.
#
# It also has a known weakness: `\p{N}+` groups **all** consecutive digits, so
# `1234` can become one token and `12345` a different single token, with no
# shared structure. Llama 3 and later split every digit separately to fix this.

# %%
# Demonstrate the trailing-space trap with the real GPT-2 tokenizer.
from transformers import AutoTokenizer

gpt2 = AutoTokenizer.from_pretrained("gpt2")

for prompt in ["The capital of France is", "The capital of France is "]:
    ids = gpt2.encode(prompt)
    print(f"{prompt!r:<32} -> {ids}")
    print(f"{'':<32}    {[gpt2.decode([i]) for i in ids]}")

# %% [markdown]
# The version with the trailing space ends in a lone `' '` token, and the model
# must then produce `'Paris'` with no leading space — a continuation it almost
# never saw in training. **Never end a prompt with a space.**

# %% [markdown]
# ## Step 4 — A complete BPE tokenizer class
#
# Now assemble everything into a real, working tokenizer.

# %%
class BPETokenizer:
    """A minimal but complete byte-level BPE tokenizer.

    Attributes:
        merges: {(id_a, id_b): new_id} in the order they were learned. Order
            matters enormously at encode time — see `encode`.
        vocab:  {id: bytes} for decoding.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.pattern = GPT2_SPLIT
        self.special_tokens: dict[str, int] = {}

    # ---------------------------------------------------------------- train
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256, "vocab must be at least the 256 byte values"
        n_merges = vocab_size - 256

        # Pre-tokenize, then work on a list of byte-id lists. Merges never
        # cross chunk boundaries because chunks stay separate lists.
        chunks = [list(c.encode("utf-8")) for c in self.pattern.findall(text)]

        for i in range(n_merges):
            counts: Counter = Counter()
            for chunk in chunks:
                get_pair_counts(chunk, counts)
            if not counts:
                print(f"stopping early at {i} merges — nothing left to merge")
                break

            top_pair, freq = counts.most_common(1)[0]
            if freq < 2:
                print(f"stopping early at {i} merges — no pair occurs twice")
                break

            new_id = 256 + i
            chunks = [merge(c, top_pair, new_id) for c in chunks]
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

            if verbose and (i < 5 or i % 200 == 0):
                print(f"  merge {i:>5}: {top_pair} -> {new_id} = "
                      f"{self.vocab[new_id]!r} ({freq}x)")

    # --------------------------------------------------------------- encode
    def _encode_chunk(self, raw: bytes) -> list[int]:
        ids = list(raw)
        while len(ids) >= 2:
            # Critical: apply merges in the order they were LEARNED, not by
            # frequency in this string. The lowest new_id is the earliest merge.
            # Getting this wrong yields a tokenizer that "works" but produces
            # ids inconsistent with training — a truly miserable bug to find.
            pairs = get_pair_counts(ids)
            candidate = min(
                pairs, key=lambda p: self.merges.get(p, float("inf"))
            )
            if candidate not in self.merges:
                break  # no learned merge applies; we're done
            ids = merge(ids, candidate, self.merges[candidate])
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in self.pattern.findall(text):
            out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    # --------------------------------------------------------------- decode
    def decode(self, ids: list[int]) -> str:
        inv = {v: k for k, v in self.special_tokens.items()}
        parts: list[bytes] = []
        for i in ids:
            if i in inv:
                parts.append(inv[i].encode("utf-8"))
            else:
                parts.append(self.vocab[i])
        # errors="replace" matters: a partial multi-byte sequence (e.g. you
        # sliced mid-emoji) would otherwise raise instead of degrading.
        return b"".join(parts).decode("utf-8", errors="replace")

    def register_special(self, tokens: list[str]) -> None:
        base = 256 + len(self.merges)
        for j, t in enumerate(tokens):
            self.special_tokens[t] = base + j
            self.vocab[base + j] = t.encode("utf-8")

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)


# %% [markdown]
# ## Step 5 — Train it on real text
#
# We'll use a slice of TinyStories. Vocab of 2000 is small but plenty to see
# the behaviour; real tokenizers use 32k–200k.

# %%
import time

from datasets import load_dataset

print("fetching training text...")
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
corpus = "\n\n".join(d["text"] for d in ds.take(3000))
print(f"corpus: {len(corpus):,} chars, {len(corpus.encode()):,} bytes")

tokenizer = BPETokenizer()
t0 = time.time()
tokenizer.train(corpus, vocab_size=2000, verbose=True)
print(f"\ntrained {len(tokenizer.merges)} merges in {time.time()-t0:.1f}s")

# %% [markdown]
# ### What did it learn?
#
# The merge order is a readout of the statistics of English. Early merges should
# be extremely common fragments; later ones whole words.

# %%
learned = [(nid, tokenizer.vocab[nid]) for nid in sorted(tokenizer.vocab) if nid >= 256]

print("first 30 merges (most frequent patterns):")
for nid, b in learned[:30]:
    print(f"  {nid:>5} {b.decode('utf-8', 'replace')!r}")

print("\nlast 20 merges (rarer, longer):")
for nid, b in learned[-20:]:
    print(f"  {nid:>5} {b.decode('utf-8', 'replace')!r}")

longest = sorted(learned, key=lambda x: -len(x[1]))[:15]
print("\nlongest tokens learned:")
for nid, b in longest:
    print(f"  {nid:>5} ({len(b):>2} bytes) {b.decode('utf-8', 'replace')!r}")

# %% [markdown]
# ## Step 6 — Round-trip correctness
#
# **The one property a tokenizer must have: `decode(encode(x)) == x` for all x.**
# Byte-level BPE gives this for free. Test it aggressively, including on text
# the tokenizer never trained on.

# %%
tests = [
    "Once upon a time there was a little girl.",
    "Hello, world!",
    "unseen vocabulary: xylophone quixotic",
    "numbers 12345 and 3.14159",
    "emoji 🚀🎉 and 中文字符",
    "   leading and trailing spaces   ",
    "",
    "\n\n\ttabs and newlines\n",
    "MiXeD CaSe WoRdS",
]

all_ok = True
for t in tests:
    ids = tokenizer.encode(t)
    back = tokenizer.decode(ids)
    ok = back == t
    all_ok &= ok
    ratio = len(t.encode()) / len(ids) if ids else 0
    print(f"  [{'ok ' if ok else 'FAIL'}] {len(ids):>3} tok, {ratio:4.2f} B/tok  {t[:44]!r}")

print(f"\nround-trip: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")

# %% [markdown]
# ## Step 7 — Compression ratio: the metric that matters
#
# A tokenizer's job is to represent text in fewer tokens. **Bytes per token** is
# the score. Higher = each token carries more text = longer effective context
# and cheaper inference.

# %%
def compression(tk, text: str) -> float:
    return len(text.encode("utf-8")) / max(len(tk.encode(text)), 1)


held_out = "\n\n".join(d["text"] for d in load_dataset(
    "roneneldan/TinyStories", split="validation", streaming=True).take(200))

print(f"{'vocab size':>12} {'bytes/token':>13}")
print("-" * 27)
for vs in [512, 1000, 2000]:
    t = BPETokenizer()
    t.train(corpus[:2_000_000], vocab_size=vs)
    print(f"{vs:>12} {compression(t, held_out):>13.2f}")

print(f"{'gpt2 (50k)':>12} {len(held_out.encode())/len(gpt2.encode(held_out)):>13.2f}")

# %% [markdown]
# Two things to notice:
#
# 1. **Diminishing returns.** Doubling the vocab does not double compression —
#   it grows roughly logarithmically. That's why nobody uses a 1M vocab: the
#   embedding table cost grows linearly while the benefit grows logarithmically.
# 2. GPT-2's 50k vocab, trained on far more text, beats our 2k substantially.
#
# **The vocab-size trade-off:**
#
# | larger vocab | |
# |---|---|
# | + | shorter sequences → less attention compute, longer effective context |
# | − | bigger embedding + output matrices (`vocab × d_model` each) |
# | − | rare tokens get few gradient updates and stay poorly trained |
#
# For a 124M model, 32k–50k is the sweet spot. SmolLM2 uses 49k; Llama 3 uses
# 128k (it targets multilingual, where a bigger vocab pays off more).

# %% [markdown]
# ## Step 8 — The multilingual tax, measured
#
# This is a real fairness and cost issue, not a curiosity.

# %%
same_meaning = {
    "English": "The weather is beautiful today and I want to go for a walk in the park.",
    "Spanish": "El clima está hermoso hoy y quiero salir a caminar por el parque.",
    "German":  "Das Wetter ist heute schön und ich möchte im Park spazieren gehen.",
    "Russian": "Сегодня прекрасная погода, и я хочу прогуляться в парке.",
    "Chinese": "今天天气很好，我想去公园散步。",
    "Hindi":   "आज मौसम बहुत अच्छा है और मैं पार्क में टहलने जाना चाहता हूँ।",
}

base = len(gpt2.encode(same_meaning["English"]))
print(f"{'language':<10}{'tokens':>8}{'vs English':>12}")
print("-" * 30)
for lang, s in same_meaning.items():
    n = len(gpt2.encode(s))
    print(f"{lang:<10}{n:>8}{n/base:>11.2f}x")

# %% [markdown]
# The same sentence can cost 2–4× more tokens in some languages. That means
# 2–4× the API cost, 2–4× less usable context, and worse quality — because the
# model has to spend capacity reassembling meaning from fragments. Newer
# tokenizers (Llama 3's 128k, Gemma's 256k) exist largely to narrow this gap.

# %% [markdown]
# ## Step 9 — The production path
#
# Our implementation is correct but slow (pure Python; encode is O(n·merges)).
# HuggingFace `tokenizers` is Rust-backed and ~100–1000× faster. Use it for real
# work — but now you know exactly what it's doing.

# %%
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

hf_tok = Tokenizer(models.BPE(unk_token=None))
hf_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
hf_tok.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=2000,
    special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes
    show_progress=False,
)

t0 = time.time()
hf_tok.train_from_iterator([corpus], trainer)
print(f"HF trained in {time.time()-t0:.1f}s (ours took much longer for the same job)")

enc = hf_tok.encode("Once upon a time there was a little girl.")
print(f"\nids:    {enc.ids}")
print(f"tokens: {enc.tokens}")
print(f"decode: {hf_tok.decode(enc.ids)!r}")
print(f"\ncompression on held-out: "
      f"{len(held_out.encode())/len(hf_tok.encode(held_out).ids):.2f} bytes/token")

# %% [markdown]
# ### Those special tokens
#
# We reserved four, and each has a job you'll use in later notebooks:
#
# | token | purpose | first used in |
# |---|---|---|
# | `<\|endoftext\|>` | document boundary in pretraining | 04 |
# | `<\|im_start\|>` / `<\|im_end\|>` | chat turn delimiters | 07 |
# | `<\|pad\|>` | batch padding (masked out of the loss) | 07 |
#
# **Reserve them now.** Adding tokens after pretraining means resizing the
# embedding matrix and training new rows from scratch — doable, but their
# embeddings start random while everything else is converged, which hurts.

# %%
from pathlib import Path

out_dir = Path("../artifacts")
out_dir.mkdir(parents=True, exist_ok=True)
hf_tok.save(str(out_dir / "tokenizer_tinystories_2k.json"))
print(f"saved -> {(out_dir / 'tokenizer_tinystories_2k.json').resolve()}")

# %% [markdown]
# ## Exercises
#
# 1. **Digit handling.** Modify `GPT2_SPLIT` so each digit is its own chunk
#    (change ` ?\p{N}+` to ` ?\p{N}`). Retrain and confirm `12345` now becomes
#    5 tokens. This is the Llama 3 fix.
# 2. **Domain tokenizer.** Train on Python code (`bigcode/the-stack-smol`) and
#    compare compression on code vs prose against GPT-2's. Domain-specific
#    tokenizers win big on their domain.
# 3. **Find the glitch tokens.** Encode the whole corpus, count token
#    frequencies, and list tokens appearing <5 times. These are undertrained —
#    the same phenomenon behind GPT-2's infamous `SolidGoldMagikarp`.
#
# ## Checkpoint
#
# - [ ] You can explain BPE training in three sentences
# - [ ] Round-trip passes on emoji, CJK, and whitespace
# - [ ] You know why prompts shouldn't end with a space
# - [ ] `artifacts/tokenizer_tinystories_2k.json` exists
#
# **Next:** `03_transformer_from_scratch.ipynb` — the architecture itself.

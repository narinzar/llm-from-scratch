# Setup & Troubleshooting — RTX 5090 (Blackwell) on WSL2

The one-page version of what goes wrong and how to fix it.

---

## The Blackwell problem, stated plainly

The RTX 5090 is compute capability **sm_120**. PyTorch wheels built before CUDA
12.8 contain no compiled kernels for it. The failure is unusually confusing
because the obvious check passes:

```python
torch.cuda.is_available()   # True  ✓
torch.zeros(10).cuda()      # works ✓  (allocation doesn't need a kernel)
a @ b                       # RuntimeError: CUDA error: no kernel image is
                            # available for execution on the device
```

You may also see:

```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
```

### Fix

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

The `--index-url` is the whole fix. Without it, PyPI gives you either a CPU
build or a pre-sm_120 CUDA build, silently.

### Verify properly

Don't trust `is_available()`. Force a real computation:

```python
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_arch_list())          # must contain 'sm_120'
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print((a @ a).sum().item())                # this is the actual test
```

Notebook 00 does all of this for you.

---

## WSL2 specifics

**Driver.** Install the NVIDIA driver on **Windows only**. WSL sees the GPU
through `/usr/lib/wsl/lib`. Installing a Linux driver *inside* WSL breaks the
passthrough — a common and frustrating self-inflicted wound.

**Check it works:** `nvidia-smi` inside WSL should list your 5090 with no extra
installation. If it doesn't, update the Windows driver first.

**Filesystem.** Keep this repo on the Linux filesystem (`~/code/...`), never
`/mnt/c/...`. Dataset and checkpoint I/O across the 9p mount is roughly 10×
slower, and notebook 01 writes gigabytes.

**Memory.** WSL2 defaults can be stingy. In `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=48GB
swap=16GB
processors=12
```

Then `wsl --shutdown` from PowerShell and restart. With 64 GB total, leaving
~16 GB to Windows is a sane split.

---

## Hugging Face

```bash
pip install -U "huggingface_hub[cli]"
hf auth login          # token from https://huggingface.co/settings/tokens
```

Some models (Llama, Gemma) need you to accept a licence on the model page first.

**Move the cache** if your root partition is small — models and datasets reach
hundreds of GB quickly. In `~/.bashrc`:

```bash
export HF_HOME=~/hf-cache
```

Inspect and prune with `hf cache scan` and `hf cache delete`.

---

## Common errors

### `CUDA out of memory`

In order of what to try:

1. Halve `micro_batch`, double `grad_accum`. **Mathematically identical run.**
2. Enable gradient checkpointing (`gradient_checkpointing=True`, or notebook 06's
   `CheckpointedGPT`). ~30% slower, ~60% less activation memory.
3. Shorten `block_size` / `max_length`.
4. Switch to QLoRA (notebook 08).

Set this before torch initializes CUDA — it fixes "OOM with plenty free",
which is fragmentation:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### `bitsandbytes` fails on the 5090

QLoRA needs a bnb build with sm_120 kernels. Try `pip install -U bitsandbytes`.
If it still fails, skip 4-bit — a 0.5B–3B model in bf16 fits comfortably in
24 GB with plain LoRA. Only 7B+ genuinely requires QLoRA.

### Training is much slower than expected

- Is `torch.compile` on? (1.3–2×, costs 1–2 min of compile time)
- Is bf16 autocast actually active?
- Is your `.bin` on `/mnt/c/`? Move it.
- Is something else using the GPU? Check `nvidia-smi`.
- Compute your MFU (notebook 06). Under 20% means real headroom.

### `vllm` won't install

vLLM's Blackwell support has lagged at times. Everything in this course works
without it — notebook 13 just runs slower. Try `pip install -U vllm`, or a
nightly build.

### Loss goes NaN

Almost always: learning rate too high, or an `-inf` from an attention mask
leaking into a softmax. Lower `max_lr` 3×, confirm gradient clipping is enabled.
If it appears at a reproducible step, print that batch — you may have a corrupt
region in the `.bin`.

### Loss is suspiciously low from step 1

Label leakage. Either your causal mask is broken, or your targets aren't shifted
by one. Run the causality test in notebook 03.

---

## Sanity benchmarks

Rough numbers for a 5090 so you can tell "slow" from "broken":

| workload | expected |
|---|---|
| 124M pretrain, bf16, bs12×1024 | 60–110k tok/s |
| 124M pretrain + `torch.compile` | ~1.3–1.8× the above |
| MFU on a well-tuned run | 35–50% |
| 0.5B LoRA SFT, bs4×1024 | 8–15k tok/s |
| 0.5B generation, HF `generate` | 30–60 tok/s |
| 0.5B generation, vLLM | 200–600 tok/s |

Notebook 06 computes your actual MFU.

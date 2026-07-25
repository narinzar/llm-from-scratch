"""Record benchmark results and regenerate BENCHMARK.md.

Every notebook that produces a number worth tracking ends with a call to
`log_run`. That appends one JSON line to `benchmarks/runs.jsonl` and rewrites
`BENCHMARK.md` from the full history, so the table on disk is never stale:

    from llmfs.bench import log_run

    log_run(
        stage="10_dpo_from_scratch",
        metrics={"accuracy": 0.87, "margin": 2.01, "loss": 0.126},
        config={"beta": 0.1, "lr": 5e-5, "steps": 400},
        notes="baseline",
    )

Two design choices worth knowing about:

*runs.jsonl is append-only and committed.* It is the history. BENCHMARK.md is
a derived view and can always be rebuilt from it, which is why regenerating on
every call is cheap and safe.

*Nothing here imports torch at module level.* The recorder has to work in a
notebook that failed halfway through, on a laptop with no GPU, and in CI.
Hardware details are collected best-effort and degrade to "unknown".
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "benchmarks" / "runs.jsonl"
REPORT = ROOT / "BENCHMARK.md"

# Human-readable titles for the stages we expect. Anything not listed still
# gets recorded and rendered -- this only controls the heading text.
STAGES: dict[str, str] = {
    "02_tokenizer": "BPE tokenizer",
    "03_transformer": "Transformer forward pass",
    "04_pretrain": "Pretraining a 124M GPT",
    "05_modern_architecture": "RoPE / RMSNorm / SwiGLU / GQA",
    "06_scaling_and_efficiency": "Throughput and MFU",
    "07_sft_from_scratch": "SFT from scratch",
    "08_sft_with_trl_and_lora": "SFT with TRL + LoRA",
    "09_reward_modeling": "Reward model",
    "10_dpo_from_scratch": "DPO from scratch",
    "11_dpo_with_trl": "DPO with TRL",
    "12_grpo_rlvr_from_scratch": "GRPO / RLVR from scratch",
    "13_grpo_with_trl_gsm8k": "GRPO on GSM8K",
    "14_evaluation": "Evaluation harness",
    "15_inference_and_capstone": "Inference and serving",
}

# Which direction is an improvement. Matched as substrings against the metric
# name, longest first, so "val_loss" resolves via "loss" and "tokens_per_sec"
# via "tokens_per_sec" rather than the shorter "sec".
_LOWER_IS_BETTER = ("loss", "perplexity", "ppl", "seconds", "latency", "vram", "bits_per_byte")
_HIGHER_IS_BETTER = ("tokens_per_sec", "accuracy", "acc", "reward", "margin", "mfu",
                     "compression", "pass_rate", "f1", "win_rate", "throughput")


def metric_direction(name: str) -> int:
    """+1 if higher is better, -1 if lower is better, 0 if unknown."""
    low = name.lower()
    best, direction = 0, 0
    for pat in _HIGHER_IS_BETTER:
        if pat in low and len(pat) > best:
            best, direction = len(pat), 1
    for pat in _LOWER_IS_BETTER:
        if pat in low and len(pat) > best:
            best, direction = len(pat), -1
    return direction


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _commit() -> str:
    sha = _git("rev-parse", "--short", "HEAD")
    if not sha:
        return "unknown"
    return f"{sha}-dirty" if _git("status", "--porcelain") else sha


def _hardware() -> dict:
    """Best-effort device description. Never raises, never requires torch."""
    hw = {"device": "cpu", "python": platform.python_version(), "torch": None}
    try:
        import torch

        hw["torch"] = torch.__version__
        if torch.cuda.is_available():
            hw["device"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            hw["vram_gb"] = round(props.total_memory / 1024**3, 1)
    except Exception:  # noqa: BLE001 - a broken torch must not lose the result
        pass
    return hw


def log_run(
    stage: str,
    metrics: dict[str, float],
    config: dict | None = None,
    notes: str = "",
    key: str | None = None,
    rebuild: bool = True,
) -> dict:
    """Append one benchmark run and regenerate BENCHMARK.md.

    stage    identifier grouping runs across time, e.g. "10_dpo_from_scratch"
    metrics  the numbers being tracked; the first is the headline unless `key`
    config   hyperparameters that produced them, shown so runs are comparable
    notes    short free text, e.g. "baseline" or "beta 0.1 -> 0.5"
    key      name of the headline metric for the summary table
    rebuild  set False to append without rewriting the report (batch imports)

    Returns the recorded entry.
    """
    if not metrics:
        raise ValueError("log_run needs at least one metric")
    clean = {k: float(v) for k, v in metrics.items()}
    if key is not None and key not in clean:
        raise ValueError(f"key {key!r} is not one of the metrics {list(clean)}")

    entry = {
        "stage": stage,
        "when": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": _commit(),
        "metrics": clean,
        "key": key or next(iter(clean)),
        "config": config or {},
        "notes": notes,
        "hardware": _hardware(),
    }

    RUNS.parent.mkdir(parents=True, exist_ok=True)
    with RUNS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")

    if rebuild:
        render()
    print(f"[bench] recorded {stage}: " +
          ", ".join(f"{k}={_fmt(v)}" for k, v in clean.items()))
    return entry


def load() -> list[dict]:
    """Read every recorded run, skipping corrupt lines rather than dying."""
    if not RUNS.exists():
        return []
    runs = []
    for i, line in enumerate(RUNS.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"[bench] skipping malformed line {i} of {RUNS.name}")
    return runs


def _fmt(v: float) -> str:
    """Format a metric so tables stay narrow but keep useful precision."""
    a = abs(v)
    if v == int(v) and a < 1e15:
        return f"{int(v):,}"
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:.3f}".rstrip("0").rstrip(".")
    if a >= 1e-4:
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return f"{v:.2e}"


def _delta(cur: float, prev: float | None, name: str) -> str:
    """Signed change vs the previous run, marked better/worse where known."""
    if prev is None:
        return "—"
    diff = cur - prev
    if diff == 0:
        return "no change"
    arrow = "▲" if diff > 0 else "▼"
    direction = metric_direction(name)
    verdict = ""
    if direction:
        verdict = " better" if (diff > 0) == (direction > 0) else " worse"
    pct = f" ({diff / abs(prev):+.1%})" if prev else ""
    return f"{arrow} {_fmt(abs(diff))}{pct}{verdict}"


def _config_str(cfg: dict, limit: int = 4) -> str:
    if not cfg:
        return "—"
    items = [f"{k}={_fmt(v) if isinstance(v, (int, float)) else v}"
             for k, v in list(cfg.items())[:limit]]
    if len(cfg) > limit:
        items.append(f"+{len(cfg) - limit} more")
    return "`" + "`, `".join(items) + "`"


def _table(rows: list[list[str]], header: list[str]) -> list[str]:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def render(path: Path | None = None) -> str:
    """Rebuild BENCHMARK.md from runs.jsonl. Returns the markdown written."""
    dest = path or REPORT
    runs = load()

    by_stage: dict[str, list[dict]] = {}
    for r in runs:
        by_stage.setdefault(r["stage"], []).append(r)
    for v in by_stage.values():
        v.sort(key=lambda r: r["when"])

    # Stamp the report with the newest run, not the wall clock. Rendering has to
    # be a pure function of runs.jsonl or `--check` could never pass: it would
    # diff a fresh timestamp against the committed one every single time.
    newest = max((r["when"] for r in runs), default=None)
    now = (newest.replace("T", " ").replace("Z", " UTC") if newest
           else "never — no runs yet")
    lines = [
        "# Benchmarks",
        "",
        "<!-- Generated by llmfs.bench. Do not edit by hand: your changes will be",
        "     overwritten on the next run. Edit benchmarks/runs.jsonl instead, or",
        "     rerun `python -m llmfs.bench`. -->",
        "",
        f"**{len(runs)} run{'s' if len(runs) != 1 else ''}** recorded across "
        f"**{len(by_stage)} of {len(STAGES)} stages** · last run {now}",
        "",
        "Every notebook ends by calling `llmfs.bench.log_run(...)`, which appends to",
        "[`benchmarks/runs.jsonl`](benchmarks/runs.jsonl) and regenerates this file.",
        "Tune a hyperparameter, rerun the notebook, and the new row lands here with",
        "its delta against your previous attempt — that comparison is the point.",
        "",
        "---",
        "",
        "## Latest result per stage",
        "",
    ]

    if not runs:
        lines += ["_Nothing recorded yet. Run any notebook to populate this table._", ""]
    else:
        rows = []
        for stage in sorted(by_stage):
            hist = by_stage[stage]
            last = hist[-1]
            k = last.get("key") or next(iter(last["metrics"]))
            cur = last["metrics"].get(k)
            prev = None
            for older in reversed(hist[:-1]):
                if k in older["metrics"]:
                    prev = older["metrics"][k]
                    break
            rows.append([
                f"[{stage}](#{stage.replace('_', '-')})",
                STAGES.get(stage, "—"),
                f"`{k}`",
                f"**{_fmt(cur)}**",
                _delta(cur, prev, k),
                str(len(hist)),
                last["when"][:10],
            ])
        lines += _table(rows, ["Stage", "What", "Metric", "Latest", "Δ vs prev",
                               "Runs", "When"])
        lines.append("")

    missing = [s for s in STAGES if s not in by_stage]
    if missing:
        lines += [
            f"Not yet recorded ({len(missing)}): "
            + ", ".join(f"`{s}`" for s in missing),
            "",
        ]

    lines += ["---", "", "## History", ""]

    if not runs:
        lines.append("_No history yet._")
    for stage in sorted(by_stage):
        hist = by_stage[stage]
        title = STAGES.get(stage, stage)
        lines += [f"### {stage}", "", f"{title}. {len(hist)} run"
                  f"{'s' if len(hist) != 1 else ''}, newest first.", ""]

        names: list[str] = []
        for r in hist:
            for m in r["metrics"]:
                if m not in names:
                    names.append(m)

        rows = []
        for i, r in enumerate(reversed(hist)):
            n = len(hist) - i
            cells = [str(n), r["when"][:16].replace("T", " ")]
            cells += [_fmt(r["metrics"][m]) if m in r["metrics"] else "—"
                      for m in names]
            cells += [
                _config_str(r["config"]),
                r.get("notes") or "—",
                f"`{r.get('commit', '?')}`",
                r.get("hardware", {}).get("device", "?"),
            ]
            rows.append(cells)
        lines += _table(
            rows,
            ["#", "When"] + [f"`{m}`" for m in names]
            + ["Config", "Notes", "Commit", "Device"],
        )
        lines.append("")

    lines += [
        "---",
        "",
        "## Recording a run",
        "",
        "```python",
        "from llmfs.bench import log_run",
        "",
        "log_run(",
        '    stage="10_dpo_from_scratch",',
        '    metrics={"accuracy": 0.87, "margin": 2.01, "loss": 0.126},',
        '    config={"beta": 0.1, "lr": 5e-5, "steps": 400},',
        '    notes="beta 0.1 -> 0.5",',
        ")",
        "```",
        "",
        "Deltas are labelled *better* or *worse* using the metric name: anything",
        "matching loss/perplexity/latency/vram counts down, and",
        "accuracy/reward/margin/throughput/MFU counts up. An unrecognised name",
        "still gets a signed delta, just without the verdict.",
        "",
        "Numbers are only comparable within a device — a run on an RTX 5090 and one",
        "on CPU are different experiments, so the device column is part of the row.",
        "",
    ]

    out = "\n".join(lines).rstrip() + "\n"
    dest.write_text(out, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    """python -m llmfs.bench [--check]

    Notebooks rewrite the report themselves via log_run, so this is only needed
    after editing runs.jsonl by hand, after a merge, or in CI.
    """
    import sys

    args = sys.argv[1:] if argv is None else argv
    runs = load()

    if "--check" in args:
        before = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""
        if render() != before:
            print(f"{REPORT.name} is out of date -- run `python -m llmfs.bench`")
            return 1
        print(f"{REPORT.name} is up to date ({len(runs)} run(s))")
        return 0

    render()
    stages = {r["stage"] for r in runs}
    print(f"wrote {REPORT.relative_to(ROOT)} from {len(runs)} run(s) "
          f"across {len(stages)} stage(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

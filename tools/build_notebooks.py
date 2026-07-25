#!/usr/bin/env python3
"""Convert jupytext 'percent format' .py sources into .ipynb notebooks.

The notebooks under notebooks/ are generated from src/*.py. Editing the .py
sources keeps the curriculum reviewable in git (notebook JSON diffs are awful),
and this script regenerates the .ipynb files learners actually open.

    python tools/build_notebooks.py            # build all
    python tools/build_notebooks.py 04         # build just the ones matching "04"

Percent format:

    # %% [markdown]
    # # A heading
    # Some prose.

    # %%
    print("a code cell")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "notebooks"


def parse_percent(text: str) -> list[dict]:
    """Split percent-format source into a list of {'type', 'lines'} cells."""
    cells: list[dict] = []
    cur_type = "code"
    cur: list[str] = []

    def flush() -> None:
        # Drop leading/trailing blank lines, skip cells that are entirely empty.
        body = cur[:]
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        if body:
            cells.append({"type": cur_type, "lines": body})

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            cur = []
            cur_type = "markdown" if "[markdown]" in stripped else "code"
            continue
        cur.append(line)
    flush()
    return cells


def strip_md(lines: list[str]) -> list[str]:
    """Markdown cells are stored as comments; remove the leading '# '."""
    out = []
    for line in lines:
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return out


def to_ipynb(cells: list[dict]) -> dict:
    nb_cells = []
    for i, cell in enumerate(cells):
        lines = cell["lines"]
        if cell["type"] == "markdown":
            lines = strip_md(lines)
        # nbformat wants every line to keep its trailing newline except the last.
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        # Stable, deterministic cell ids so rebuilds produce clean git diffs.
        cid = f"cell-{i:03d}"
        if cell["type"] == "markdown":
            nb_cells.append(
                {"cell_type": "markdown", "id": cid, "metadata": {}, "source": source}
            )
        else:
            nb_cells.append(
                {
                    "cell_type": "code",
                    "id": cid,
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source,
                }
            )
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    if not SRC.is_dir():
        print(f"no source dir: {SRC}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    needle = sys.argv[1] if len(sys.argv) > 1 else ""

    built = 0
    for path in sorted(SRC.glob("*.py")):
        if needle and needle not in path.name:
            continue
        cells = parse_percent(path.read_text(encoding="utf-8"))
        if not cells:
            print(f"  skip (no cells) {path.name}")
            continue
        dest = OUT / (path.stem + ".ipynb")
        dest.write_text(
            json.dumps(to_ipynb(cells), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        n_md = sum(1 for c in cells if c["type"] == "markdown")
        print(f"  {dest.name:<44} {len(cells):>3} cells ({n_md} md)")
        built += 1

    print(f"built {built} notebook(s) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

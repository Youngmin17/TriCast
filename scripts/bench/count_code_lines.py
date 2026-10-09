"""Count code lines the way the README slide 6 comparison does.

Blank lines, comment-only lines and lines that hold only brackets or separators are not counted. C-style files
(`.cu`, `.cuh`, `.h`, `.cpp`) skip `//` and `/* ... */` comments; YAML files skip `#` comments.

    python scripts/bench/count_code_lines.py src/tricast/recipes/fp8_f7_lowacc.yaml
    cd <micro26-ae>/csrc/quantization/mma_emu && python <repo>/scripts/bench/count_code_lines.py \
        core/*.cuh formats/fp8_e4m3.cuh gemm/scaled_fp8_mm.cuh fp8_gemm_kernels.cu
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

C_SUFFIXES = {".cu", ".cuh", ".h", ".cpp"}
ONLY_PUNCTUATION = re.compile(r"[{}()\[\];,]+")


def count(path: Path) -> int:
    c_style, in_block, lines = path.suffix in C_SUFFIXES, False, 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if c_style:
            if in_block:
                if "*/" not in line:
                    continue
                in_block, line = False, line.split("*/", 1)[1].strip()
            if line.startswith("/*"):
                in_block = "*/" not in line
                continue
            if line.startswith(("//", "*")):
                continue
        elif line.startswith("#"):
            continue
        if line and not ONLY_PUNCTUATION.fullmatch(line):
            lines += 1
    return lines


def main(paths: list[str]) -> None:
    total = 0
    for name in paths:
        lines = count(Path(name))
        total += lines
        print(f"{lines:6d}  {name}")
    print(f"{total:6d}  total")


if __name__ == "__main__":
    main(sys.argv[1:])

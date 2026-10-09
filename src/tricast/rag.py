"""Dependency-free BM25 over the numerical contract and registered specifications."""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

from .mma.spec import PRESETS
from .quant.spec import KV_PRESETS, SCHEMES, QuantSpec, ScaleSpec

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "k": {"type": "integer", "minimum": 0, "maximum": 50}},
    "required": ["query", "k"],
    "additionalProperties": False,
}


def _root() -> Path:
    for root in (*Path(__file__).resolve().parents, Path(sys.prefix) / "share" / "tricast"):
        if (root / "config" / "rag.yaml").is_file():
            return root
    return Path(__file__).resolve().parents[2]


def _sections(text: str) -> list[tuple[str, str]]:
    sections = []
    title, lines = "Introduction", []
    for line in text.splitlines():
        if re.match(r"^#{2,3} ", line):
            if lines:
                sections.append((title, "\n".join(lines).strip()))
            title, lines = line.lstrip("# "), [line]
        else:
            lines.append(line)
    if lines:
        sections.append((title, "\n".join(lines).strip()))
    return sections


def _documents() -> list[dict]:
    root = _root()
    path = root / "config" / "rag.yaml"
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "documents": ["docs/SPEC.md"], "glossary": "AGENTS.md",
    }
    documents = []
    for source in config["documents"]:
        source_path = root / source
        if source_path.is_file():
            documents.extend({"source": source, "section": title, "text": text}
                             for title, text in _sections(source_path.read_text(encoding="utf-8")))
    glossary = root / config["glossary"]
    if glossary.is_file():
        documents.extend({"source": config["glossary"], "section": title, "text": text}
                         for title, text in _sections(glossary.read_text(encoding="utf-8"))
                         if "용어집" in title or "glossary" in title.lower())
    for name, preset in sorted(PRESETS.items()):
        documents.append({"source": "src/tricast/mma/spec.py", "section": name,
                          "text": f"{name}: {preset.provenance}\nalgorithm={preset.algorithm}; "
                                  f"F={preset.f_bits}; CS={preset.chunk_size}; G={preset.g_bits}; "
                                  f"group_size={preset.group_size}; c_mode={preset.c_mode}"})
    for name, scheme in sorted(SCHEMES.items()):
        text = (f"{name}: format={scheme.format.name}; granularity={scheme.granularity}; "
                f"group_size={scheme.group_size}; rounding={scheme.rounding.value}.\n"
                f"{QuantSpec.__doc__ or ''}")
        if scheme.scale is not None:
            text += (f"\nscale.format={scheme.scale.format.name}; scale.method={scheme.scale.method}; "
                     f"two_level={scheme.scale.two_level}.\n{ScaleSpec.__doc__ or ''}")
        documents.append({"source": "src/tricast/quant/spec.py", "section": name, "text": text})
    for name, spec in sorted(KV_PRESETS.items()):
        documents.append({"source": "src/tricast/quant/spec.py", "section": f"KV {name}",
                          "text": f"{name}: mode={spec.mode}; residual={spec.residual}; "
                                  f"key_axis={spec.key_axis}; value_axis={spec.value_axis}.\n"
                                  f"{spec.__doc__ or ''}"})
    return documents


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+[a-z0-9]*|\d+|[가-힣]+", text.casefold())


def search(query: str, k: int = 5) -> list[dict]:
    """Rank local sections by BM25; ties use source then section lexicographically."""
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    if type(k) is not int or not 0 <= k <= 50:
        raise ValueError("k must be an integer in [0, 50]")
    terms = set(_tokens(query))
    if not terms or k == 0:
        return []
    documents = _documents()
    counts = [Counter(_tokens(doc["section"] + " " + doc["text"])) for doc in documents]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / max(len(lengths), 1)
    if not average:
        return []
    frequencies = {term: sum(term in count for count in counts) for term in terms}
    ranked = []
    for document, count, length in zip(documents, counts, lengths, strict=True):
        score = 0.0
        for term in sorted(terms):
            frequency = count[term]
            if frequency:
                idf = math.log1p((len(documents) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
                score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / average))
        if score:
            ranked.append({**document, "score": score})
    return sorted(ranked, key=lambda result: (-result["score"], result["source"], result["section"]))[:k]

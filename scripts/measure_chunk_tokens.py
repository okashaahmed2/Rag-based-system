"""One-off measurement: real token counts of prepared chunks.

Runs :func:`rag_pipeline.preparation.prepare_document` on every PDF in the
configured inbox and counts the tokens of every chunk with the embedding
model's own tokenizer, INCLUDING special tokens ([CLS]/[SEP]) and WITHOUT
truncation. The report is printed and written to ``<log_dir>/token_report.txt``.

Measurement only: nothing is embedded, stored, moved or written except the
report file.

Usage::

    .venv\\Scripts\\python.exe scripts\\measure_chunk_tokens.py
"""

from __future__ import annotations

import logging
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer

from rag_pipeline.config import get_settings
from rag_pipeline.extraction import PdfExtractionError
from rag_pipeline.preparation import DocumentChunk, prepare_document

logger = logging.getLogger("measure_chunk_tokens")

#: Model measured here; the embedding module will make this configurable later.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REPORT_NAME = "token_report.txt"
#: Sequence length the model was trained with (model README), in tokens.
TRAINING_SEQ_LENGTH = 128
LONGEST_SHOWN = 3
PREVIEW_CHARS = 80


@dataclass(frozen=True)
class ChunkTokens:
    """Token count of one chunk, with the fields needed for the report."""

    filename: str
    chunk: DocumentChunk
    tokens: int


def percentile(values: list[int], pct: float) -> int:
    """Nearest-rank percentile (``pct`` in 0..100) of a non-empty list."""
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def count_tokens(model: SentenceTransformer, texts: list[str]) -> list[int]:
    """Token counts including special tokens, without truncation."""
    encoded = model.tokenizer(texts, add_special_tokens=True, truncation=False)
    return [len(ids) for ids in encoded["input_ids"]]


def preview(text: str) -> str:
    """First ``PREVIEW_CHARS`` characters on one line."""
    return " ".join(text.split())[:PREVIEW_CHARS]


def doc_stats_lines(name: str, counts: list[int], chars: int, limit: int) -> list[str]:
    """Format min/median/p95/max, over-limit counts and chars-per-token."""
    over = sum(1 for count in counts if count > limit)
    over_training = sum(1 for count in counts if count > TRAINING_SEQ_LENGTH)
    return [
        f"  {name}",
        f"    chunks: {len(counts)}",
        f"    tokens min/median/p95/max: {min(counts)} / "
        f"{statistics.median(counts):g} / {percentile(counts, 95)} / {max(counts)}",
        f"    chunks over max_seq_length ({limit}): {over}",
        f"    chunks over training length ({TRAINING_SEQ_LENGTH}): {over_training}",
        f"    chars per token: {chars / sum(counts):.2f} "
        f"({chars} chars / {sum(counts)} tokens)",
    ]


def build_report(model: SentenceTransformer, pdfs: list[Path]) -> list[str]:
    """Prepare and measure every PDF and return the report lines."""
    settings = get_settings()
    limit = model.max_seq_length
    lines = [
        "Chunk token report",
        f"model: {MODEL_NAME}",
        f"chunk_size (chars): {settings.chunk_size}",
        f"chunk_overlap (chars): {settings.chunk_overlap}",
        f"model.max_seq_length: {limit}",
        f"tokenizer.model_max_length: {model.tokenizer.model_max_length}",
        "token counts include special tokens, no truncation",
        f"inbox: {settings.inbox_dir}",
        f"PDF files found: {len(pdfs)}",
        "",
        "Per document:",
    ]
    skipped: list[str] = []
    all_chunks: list[ChunkTokens] = []
    for pdf in pdfs:
        try:
            prepared = prepare_document(pdf, settings)
        except PdfExtractionError as exc:
            logger.warning("Skipping %s: %s", pdf.name, exc)
            skipped.append(f"{pdf.name}: extraction error: {exc}")
            continue
        if not prepared.chunks:
            skipped.append(
                f"{pdf.name}: 0 chunks (needs_ocr={prepared.needs_ocr}, "
                f"pages={prepared.page_count})"
            )
            continue
        counts = count_tokens(model, [chunk.text for chunk in prepared.chunks])
        chars = sum(len(chunk.text) for chunk in prepared.chunks)
        lines.extend(doc_stats_lines(pdf.name, counts, chars, limit))
        all_chunks.extend(
            ChunkTokens(pdf.name, chunk, count)
            for chunk, count in zip(prepared.chunks, counts)
        )

    lines += ["", "Skipped documents (0 chunks or extraction error):"]
    lines += [f"  {entry}" for entry in skipped] or ["  none"]

    lines += ["", "Overall (all measured documents):"]
    if all_chunks:
        counts = [item.tokens for item in all_chunks]
        chars = sum(len(item.chunk.text) for item in all_chunks)
        measured = len({item.filename for item in all_chunks})
        lines.extend(doc_stats_lines(f"{measured} document(s)", counts, chars, limit))
        lines += ["", f"Longest {LONGEST_SHOWN} chunks:"]
        longest = sorted(all_chunks, key=lambda item: item.tokens, reverse=True)
        for item in longest[:LONGEST_SHOWN]:
            lines += [
                f"  {item.filename}",
                f"    chunk_id: {item.chunk.chunk_id}",
                f"    page_number: {item.chunk.page_number}  tokens: {item.tokens}  "
                f"chars: {len(item.chunk.text)}",
                f"    text: {preview(item.chunk.text)}",
            ]
    else:
        lines.append("  no chunks measured")
    return lines


def main() -> int:
    """Run the measurement, print the report and write it to the log folder."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    pdfs = sorted(settings.inbox_dir.glob("*.pdf"), key=lambda path: path.name.lower())
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    report = "\n".join(build_report(model, pdfs)) + "\n"

    sys.stdout.reconfigure(errors="replace")
    sys.stdout.write(report)

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.log_dir / REPORT_NAME
    report_path.write_text(report, encoding="utf-8")
    sys.stdout.write(f"\nReport written to {report_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

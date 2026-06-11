"""
Transcript embeddings — beta(Z) in Mehta & Harchaoui (ICML 2025).

In the paper's zero-shot framework, beta: Z -> H_Z is the encoder for the
auxiliary modality Z. Here Z is the earnings call transcript (raw text from
the 8-K Item 2.02 filing), and beta maps it to a fixed-length embedding in
R^384 (the output dimension of all-MiniLM-L6-v2).

The embedding beta(Z) is the representation that the zero-shot predictor
uses in place of the inaccessible source modality. The residual dependence
I(X;Y|Z) — Theorem 1 — measures how much predictive information about Y
remains in X after conditioning on Z, i.e., how much beta(Z) fails to
capture. A smaller I(X;Y|Z) means Z is a better substitute for X.

Caching: embeddings are stored as .npy files at
    <cache_dir>/<ticker>/<YYYY-MM-DD>.npy
so the model is only run once per filing.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"

# Lazy-loaded singleton — avoids loading the 80 MB model at import time.
_model = None


def _get_model():
    """Return the shared SentenceTransformer instance, loading it on first call."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        logger.info("Loading sentence-transformers model: %s", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

# EDGAR 8-K filings arrive as HTML with embedded XBRL tags. Strip markup
# aggressively before embedding — the model was trained on plain text.
_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")

# all-MiniLM-L6-v2 has a 256-token context window (hard limit: 512 word-
# piece tokens). We chunk long transcripts and mean-pool chunk embeddings
# so no content is silently truncated — truncation would bias the embedding
# toward the filing header rather than the operational discussion.
_MAX_CHARS_PER_CHUNK = 2000  # ~400-500 tokens; keeps us safely under the limit
_CHUNK_OVERLAP_CHARS = 200   # sliding overlap to avoid cutting mid-sentence


def _clean_text(raw_html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = _TAG_RE.sub(" ", raw_html)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _chunk_text(text: str) -> list[str]:
    """
    Split `text` into overlapping windows of _MAX_CHARS_PER_CHUNK characters.
    Returns a single-element list when the text fits in one chunk.
    """
    if len(text) <= _MAX_CHARS_PER_CHUNK:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + _MAX_CHARS_PER_CHUNK
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - _CHUNK_OVERLAP_CHARS
    return chunks


# ---------------------------------------------------------------------------
# Embedding computation and caching
# ---------------------------------------------------------------------------

def embed_transcript(
    raw_text: str,
    ticker: str,
    filing_date: str,
    cache_dir: Path,
) -> Optional[np.ndarray]:
    """
    Compute beta(Z) for a single earnings call transcript.

    Paper reference: beta: Z -> H_Z (Section 2, zero-shot encoder for the
    Z modality). The output embedding lives in the Hilbert space H_Z and
    is the input to the zero-shot predictor f(beta(Z)).

    Parameters
    ----------
    raw_text : str
        Raw HTML text of the 8-K filing (as stored by data/edgar.py).
    ticker : str
        Used only for cache path construction.
    filing_date : str
        YYYY-MM-DD. Used only for cache path construction.
    cache_dir : Path
        Root directory for the .npy cache.

    Returns
    -------
    np.ndarray of shape (384,) — the mean-pooled sentence embedding.
    Returns None if raw_text is empty or whitespace-only.
    """
    cache_path = _cache_path(ticker, filing_date, cache_dir)
    if cache_path.exists():
        logger.debug("Cache hit: %s", cache_path)
        return np.load(cache_path)

    text = _clean_text(raw_text)
    if not text:
        logger.warning("Empty text after cleaning for %s %s — skipping", ticker, filing_date)
        return None

    chunks = _chunk_text(text)
    model = _get_model()

    # encode() returns shape (n_chunks, embedding_dim)
    chunk_embeddings: np.ndarray = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,  # L2-normalize before pooling
    )

    # Mean-pool across chunks, then re-normalize so the final vector is
    # unit-norm regardless of document length. This gives a consistent
    # cosine-similarity interpretation and matches the MTEB evaluation
    # convention for all-MiniLM-L6-v2.
    embedding = chunk_embeddings.mean(axis=0)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    embedding = embedding.astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embedding)
    logger.debug("Cached embedding: %s (shape %s)", cache_path, embedding.shape)

    return embedding


def embed_transcripts_batch(
    records: list[dict],
    cache_dir: Path,
) -> dict[tuple[str, str], Optional[np.ndarray]]:
    """
    Compute beta(Z) for a batch of filing records.

    Parameters
    ----------
    records : list of dicts with keys 'ticker', 'filing_date', 'raw_text'.
              Typically loaded directly from the JSON files produced by
              data/edgar.py. Records with parse_error or None raw_text are
              skipped and mapped to None in the output.
    cache_dir : Path
        Root directory for .npy cache files.

    Returns
    -------
    dict mapping (ticker, filing_date) -> np.ndarray or None.
    """
    results: dict[tuple[str, str], Optional[np.ndarray]] = {}
    for rec in records:
        ticker = rec["ticker"]
        date = rec["filing_date"]
        key = (ticker, date)

        if rec.get("parse_error"):
            logger.warning("Skipping %s %s — parse_error: %s", ticker, date, rec["parse_error"])
            results[key] = None
            continue

        raw = rec.get("raw_text") or ""
        results[key] = embed_transcript(raw, ticker, date, cache_dir)

    n_ok = sum(1 for v in results.values() if v is not None)
    logger.info("Embedded %d / %d transcripts", n_ok, len(results))
    return results


def load_embedding(
    ticker: str,
    filing_date: str,
    cache_dir: Path,
) -> Optional[np.ndarray]:
    """
    Load a previously cached beta(Z) embedding. Returns None if not cached.

    Use this for read-only access in downstream modules (e.g., estimating
    I(X;Y|Z)) to avoid accidentally re-running the model.
    """
    path = _cache_path(ticker, filing_date, cache_dir)
    if not path.exists():
        return None
    return np.load(path)


def _cache_path(ticker: str, filing_date: str, cache_dir: Path) -> Path:
    """<cache_dir>/<ticker>/<filing_date>.npy"""
    return cache_dir / ticker / f"{filing_date}.npy"


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Embed 8-K transcripts")
    parser.add_argument("--filings-dir", default="raw_filings")
    parser.add_argument("--cache-dir", default="embeddings_cache")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "JPM"])
    args = parser.parse_args()

    filings_root = Path(args.filings_dir)
    cache_root = Path(args.cache_dir)

    for ticker in args.tickers:
        ticker_dir = filings_root / ticker
        if not ticker_dir.exists():
            logger.warning("No filings dir for %s", ticker)
            continue

        records = []
        for jf in sorted(ticker_dir.glob("*.json")):
            data = json.loads(jf.read_text())
            if data.get("item_202_present"):
                records.append(data)

        print(f"{ticker}: {len(records)} Item 2.02 filings to embed")
        results = embed_transcripts_batch(records, cache_root)

        for (t, d), emb in results.items():
            if emb is not None:
                print(f"  {t} {d}: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
            else:
                print(f"  {t} {d}: FAILED")

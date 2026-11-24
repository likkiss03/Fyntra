
"""Prepare sentence-based Item 1A chunks for pilot NLP analysis.

These are approximate text chunks, not original SEC paragraphs.
The verified source corpus remains unchanged.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def normalize_text(text):
    """Normalize whitespace."""
    return re.sub(r"\s+", " ", str(text)).strip()


def split_sentences(text):
    """Heuristic sentence splitting for pilot analysis."""
    text = normalize_text(text)

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z])",
        text
    )

    return [s.strip() for s in sentences if s.strip()]


def create_chunks(text, target_words=150, max_words=220):
    """Create sentence-based chunks and merge short fragments."""

    sentences = split_sentences(text)

    chunks = []
    current = []
    current_words = 0

    def flush():
        nonlocal current, current_words

        if current:
            chunks.append(" ".join(current))
            current = []
            current_words = 0

    for sentence in sentences:

        words = sentence.split()
        count = len(words)

        # Handle unusually long sentences.
        if count > max_words:
            flush()

            for start in range(0, count, max_words):
                chunks.append(
                    " ".join(words[start:start + max_words])
                )

            continue

        # Prevent the current chunk from becoming too large.
        if current and current_words + count > max_words:
            flush()

        current.append(sentence)
        current_words += count

        # Save the chunk once it reaches the target length.
        if current_words >= target_words:
            flush()

    # Save any remaining text.
    flush()

    # Merge extremely short chunks into the previous chunk.
    # Preserve all original words.
    merged = []

    for chunk in chunks:

        word_count = len(chunk.split())

        if merged and word_count < 50:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    # Handle a short first chunk, if one exists.
    if len(merged) > 1 and len(merged[0].split()) < 50:
        merged[1] = merged[0] + " " + merged[1]
        merged.pop(0)

    # Verify that no source text was lost or reordered.
    reconstructed = " ".join(merged)

    if reconstructed != normalize_text(text):
        raise ValueError(
            "Text integrity check failed during chunking."
        )

    return merged


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="./sec_item1a_data"
    )

    args = parser.parse_args()

    root = Path(args.output_dir)
    source = root / "item1a_pilot_verified.csv"

    # Ensure the verified dataset exists.
    if not source.exists():
        raise FileNotFoundError(
            f"Verified corpus not found: {source}"
        )

    df = pd.read_csv(
        source,
        keep_default_na=False
    )

    records = []

    for _, filing in df.iterrows():

        chunks = create_chunks(
            filing["item1a_text"]
        )

        for index, chunk in enumerate(chunks, start=1):

            records.append({
                "filing_id": filing["filing_id"],
                "company": filing["company"],
                "filing_date": filing["filing_date"],
                "chunk_id": (
                    f"{filing['filing_id']}_c{index:04d}"
                ),
                "chunk_number": index,
                "chunk_text": chunk,
                "word_count": len(chunk.split()),
                "segmentation_method": (
                    "heuristic_sentence_chunking_v2"
                )
            })

    result = pd.DataFrame(records)

    if result.empty:
        raise ValueError("No chunks were generated.")

    # Check that all source filings are represented.
    if result["filing_id"].nunique() != df["filing_id"].nunique():
        raise ValueError(
            "One or more source filings are missing from the chunks."
        )

    output = root / "item1a_pilot_chunks.csv"

    result.to_csv(output, index=False)

    # Calculate total word counts.
    source_words = sum(
        len(normalize_text(text).split())
        for text in df["item1a_text"]
    )

    chunk_words = int(result["word_count"].sum())

    print("=" * 60)
    print("FYNTRA - ITEM 1A CHUNK PREPARATION")
    print("=" * 60)

    print("Source filings:", len(df))
    print("Companies:", df["company"].nunique())
    print("Generated chunks:", len(result))
    print("Saved:", output)

    print("\nChunk statistics:")
    print(result["word_count"].describe().to_string())

    print("\nChunks per company:")
    print(
        result.groupby("company").size().to_string()
    )

    print("\nChunks per filing:")
    print(
        result.groupby("filing_id")
        .size()
        .describe()
        .to_string()
    )

    print("\nQuality checks:")

    print(
        "Chunks under 50 words:",
        (result["word_count"] < 50).sum()
    )

    print(
        "Chunks over 220 words:",
        (result["word_count"] > 220).sum()
    )

    print(
        "Missing text:",
        result["chunk_text"].isna().sum()
    )

    print(
        "Duplicate chunk IDs:",
        result["chunk_id"].duplicated().sum()
    )

    print("\nText integrity checks:")
    print("Original word count:", source_words)
    print("Chunked word count:", chunk_words)
    print("Words lost:", source_words - chunk_words)

    print("\nChunk preparation completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()

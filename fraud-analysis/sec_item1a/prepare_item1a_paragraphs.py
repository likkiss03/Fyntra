
"""Prepare SEC Item 1A paragraphs for boilerplate-aware NLP analysis."""

import argparse
import re
from pathlib import Path

import pandas as pd


def normalize_text(text):
    """Normalize whitespace without removing financial terminology."""
    return re.sub(r"\s+", " ", str(text)).strip()


def split_paragraphs(text):
    """Split text at blank lines, retaining meaningful paragraphs."""
    blocks = re.split(r"\n\s*\n+", str(text))
    return [
        normalize_text(block)
        for block in blocks
        if len(normalize_text(block).split()) >= 20
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="./sec_item1a_data"
    )
    args = parser.parse_args()

    root = Path(args.output_dir)

    source = root / "item1a_pilot_verified.csv"

    if not source.exists():
        raise FileNotFoundError(
            f"Verified corpus not found: {source}"
        )

    df = pd.read_csv(source, keep_default_na=False)

    records = []

    for _, filing in df.iterrows():
        paragraphs = split_paragraphs(filing["item1a_text"])

        for index, paragraph in enumerate(paragraphs, start=1):
            records.append({
                "filing_id": filing["filing_id"],
                "company": filing["company"],
                "filing_date": filing["filing_date"],
                "paragraph_id": f"{filing['filing_id']}_p{index:04d}",
                "paragraph_number": index,
                "paragraph_text": paragraph,
                "word_count": len(paragraph.split())
            })

    output = root / "item1a_pilot_paragraphs.csv"

    pd.DataFrame(records).to_csv(output, index=False)

    print("Source filings:", len(df))
    print("Extracted paragraphs:", len(records))
    print("Saved:", output)

    if records:
        result = pd.DataFrame(records)
        print("\nParagraph counts by company:")
        print(result.groupby("company").size().to_string())


if __name__ == "__main__":
    main()

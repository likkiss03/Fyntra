
"""Fyntra: Historical boilerplate detection using TF-IDF.

Compare each current filing against the immediately
preceding available filing from the same company.

Similarity thresholds are exploratory, not validated.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def classify_similarity(score, high_threshold, low_threshold):

    if score >= high_threshold:
        return "potential_boilerplate"

    if score >= low_threshold:
        return "modified_disclosure"

    return "potential_novel_disclosure"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="./sec_item1a_data"
    )

    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.80
    )

    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.40
    )

    args = parser.parse_args()

    if not (
        0 <= args.low_threshold
        < args.high_threshold <= 1
    ):
        raise ValueError("Invalid similarity thresholds.")

    root = Path(args.output_dir)

    source = root / "item1a_pilot_chunks.csv"

    if not source.exists():
        raise FileNotFoundError(source)

    df = pd.read_csv(
        source,
        keep_default_na=False
    )

    df["filing_date"] = pd.to_datetime(
        df["filing_date"],
        errors="raise"
    )

    if df["chunk_id"].duplicated().any():
        raise ValueError("Duplicate chunk IDs detected.")

    df = df.sort_values(
        ["company", "filing_date", "chunk_number"]
    )

    records = []

    for company, company_df in df.groupby("company"):

        filings = (
            company_df[
                ["filing_id", "filing_date"]
            ]
            .drop_duplicates()
            .sort_values("filing_date")
            .reset_index(drop=True)
        )

        print(f"\nProcessing: {company}")

        for i in range(len(filings)):

            current_id = filings.iloc[i]["filing_id"]

            current = company_df.loc[
                company_df["filing_id"] == current_id
            ].reset_index(drop=True)

            # First filing has no historical reference.
            if i == 0:

                for row in current.to_dict("records"):

                    records.append({
                        **row,
                        "reference_filing_id": "",
                        "reference_filing_date": "",
                        "matched_chunk_id": "",
                        "matched_chunk_text": "",
                        "max_similarity": np.nan,
                        "novelty_score": np.nan,
                        "boilerplate_category": "no_history",
                        "scoring_status": "no_history"
                    })

                continue

            previous_id = filings.iloc[i - 1]["filing_id"]

            previous = company_df.loc[
                company_df["filing_id"] == previous_id
            ].reset_index(drop=True)

            # Fit only on historical text.
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                norm="l2"
            )

            previous_vectors = vectorizer.fit_transform(
                previous["chunk_text"]
            )

            current_vectors = vectorizer.transform(
                current["chunk_text"]
            )

            similarities = cosine_similarity(
                current_vectors,
                previous_vectors
            )

            best_indices = similarities.argmax(axis=1)

            best_scores = similarities.max(axis=1)

            nonzero_counts = current_vectors.getnnz(axis=1)

            for j, row in enumerate(
                current.to_dict("records")
            ):

                # No shared historical vocabulary:
                # do not assign an arbitrary matched chunk.
                if nonzero_counts[j] == 0 or best_scores[j] <= 0:

                    records.append({
                        **row,
                        "reference_filing_id": previous_id,
                        "reference_filing_date": (
                            filings.iloc[i - 1]["filing_date"]
                        ),
                        "matched_chunk_id": "",
                        "matched_chunk_text": "",
                        "max_similarity": 0.0,
                        "novelty_score": 1.0,
                        "boilerplate_category": (
                            "potential_novel_disclosure"
                        ),
                        "scoring_status": "no_shared_terms"
                    })

                    continue

                score = float(
                    np.clip(best_scores[j], 0, 1)
                )

                matched = previous.iloc[
                    int(best_indices[j])
                ]

                category = classify_similarity(
                    score,
                    args.high_threshold,
                    args.low_threshold
                )

                records.append({
                    **row,
                    "reference_filing_id": previous_id,
                    "reference_filing_date": (
                        matched["filing_date"]
                    ),
                    "matched_chunk_id": matched["chunk_id"],
                    "matched_chunk_text": matched["chunk_text"],
                    "max_similarity": score,
                    "novelty_score": 1.0 - score,
                    "boilerplate_category": category,
                    "scoring_status": "scored"
                })

            print(
                f"  {current.iloc[0]['filing_date'].date()}: "
                f"{len(current)} chunks compared against "
                f"{len(previous)} historical chunks"
            )

    result = pd.DataFrame(records)

    if len(result) != len(df):
        raise ValueError("Chunk coverage check failed.")

    result["high_threshold"] = args.high_threshold
    result["low_threshold"] = args.low_threshold

    result["scoring_method"] = (
        "historical_tfidf_cosine_v1"
    )

    output = root / "item1a_pilot_boilerplate_tfidf.csv"

    result.to_csv(output, index=False)

    scored = result.loc[
        result["max_similarity"].notna()
    ].copy()

    # Generate a readable filing-level summary.
    summary = (
        result.groupby(
            ["company", "filing_id", "filing_date"]
        )
        .agg(
            total_chunks=("chunk_id", "size"),
            scored_chunks=("max_similarity", "count"),
            mean_similarity=("max_similarity", "mean"),
            mean_novelty=("novelty_score", "mean")
        )
        .reset_index()
    )

    categories = pd.crosstab(
        result["filing_id"],
        result["boilerplate_category"]
    ).reset_index()

    summary = summary.merge(
        categories,
        on="filing_id",
        how="left",
        validate="one_to_one"
    )

    summary_output = (
        root / "item1a_pilot_boilerplate_summary.csv"
    )

    summary.to_csv(summary_output, index=False)

    print("\n" + "=" * 60)
    print("FYNTRA - TF-IDF BOILERPLATE ANALYSIS")
    print("=" * 60)

    print("Total chunks:", len(result))
    print("Scored chunks:", len(scored))
    print(
        "Chunks without history:",
        (result["boilerplate_category"] == "no_history").sum()
    )

    print("\nCategory distribution:")

    print(
        result["boilerplate_category"]
        .value_counts()
        .to_string()
    )

    print("\nSimilarity statistics:")

    print(
        scored["max_similarity"]
        .describe()
        .to_string()
    )

    print("\nCategories by company:")

    print(
        pd.crosstab(
            result["company"],
            result["boilerplate_category"]
        ).to_string()
    )

    print("\nQuality checks:")

    print(
        "Duplicate chunk IDs:",
        result["chunk_id"].duplicated().sum()
    )

    print(
        "Chunks preserved:",
        len(result) == len(df)
    )

    print("\nDetailed results:", output)
    print("Filing-level summary:", summary_output)

    print("\nAnalysis completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()

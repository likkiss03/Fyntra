"""Produce automated QA indicators and a human review template."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./sec_item1a_data")
    args = parser.parse_args()
    root = Path(args.output_dir)
    qa = root / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "item1a_pilot_candidates.csv", keep_default_na=False)
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").fillna(0).astype(int)
    df["qa_flag"] = df.apply(
        lambda r: "no_text" if not str(r["item1a_text"]).strip() else
        "multiple_candidates" if int(r["plausible_count"]) > 1 else
        "short_section" if r["word_count"] < 1000 else "manual_review_required", axis=1)
    cols = ["filing_id", "company", "filing_date", "source_url", "extraction_status",
            "candidate_count", "plausible_count", "word_count", "qa_flag", "extraction_error"]
    df[cols].to_csv(qa / "extraction_quality_report.csv", index=False)
    review = df[cols].copy()
    for column in ["correct_start", "correct_end", "toc_contamination", "review_decision", "review_notes"]:
        review[column] = ""
    review.to_csv(qa / "manual_review.csv", index=False)
    print("Created QA report and manual review sheet in", qa)
    print("Review ALL candidates against source filings before approving any.")

if __name__ == "__main__":
    main()

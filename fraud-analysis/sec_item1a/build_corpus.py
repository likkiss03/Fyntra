"""Build a corpus of human-accepted Item 1A candidates only."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./sec_item1a_data")
    args = parser.parse_args()
    root = Path(args.output_dir)
    candidates = pd.read_csv(root / "item1a_pilot_candidates.csv", keep_default_na=False)
    review_path = root / "qa" / "manual_review.csv"
    if not review_path.exists():
        raise FileNotFoundError("Run validate.py and manually review the resulting CSV first.")
    reviews = pd.read_csv(review_path, keep_default_na=False)
    required = {"filing_id", "correct_start", "correct_end", "toc_contamination", "review_decision"}
    if not required.issubset(reviews.columns):
        raise ValueError(f"Review CSV missing columns: {required - set(reviews.columns)}")
    if reviews["filing_id"].duplicated().any():
        raise ValueError("Duplicate filing_id in review sheet")
    reviews = reviews[["filing_id", "correct_start", "correct_end", "toc_contamination", "review_decision", "review_notes"]]
    merged = candidates.merge(reviews, on="filing_id", how="left", validate="one_to_one")
    accepted = merged.loc[
        merged["review_decision"].str.strip().str.lower().eq("accept") &
        merged["correct_start"].str.strip().str.lower().eq("yes") &
        merged["correct_end"].str.strip().str.lower().eq("yes") &
        merged["toc_contamination"].str.strip().str.lower().eq("no") &
        merged["item1a_text"].astype(str).str.strip().ne("")
    ].copy()
    accepted["verification_status"] = "manually_accepted"
    output = root / "item1a_pilot_verified.csv"
    accepted.to_csv(output, index=False)
    print(f"Accepted {len(accepted)} of {len(candidates)} filings")
    print("Saved:", output)
    print("Note: this pilot corpus is NOT yet a final research-ready dataset.")

if __name__ == "__main__":
    main()

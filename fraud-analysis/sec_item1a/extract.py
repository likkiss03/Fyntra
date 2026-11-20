"""Conservative candidate extraction of Item 1A from SEC 10-K HTML."""
import argparse
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

START = re.compile(r"^item\s*1\s*a\s*[.\-:–—]?\s*(?:risk\s*factors)?\s*$", re.I)
END = re.compile(r"^item\s*(?:1\s*b|2)\s*[.\-:–—]?\s*(?:unresolved\s*staff\s*comments|properties)?\s*$", re.I)

def extract_candidate(content):
    soup = BeautifulSoup(content, "lxml")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    lines = [re.sub(r"\s+", " ", x).strip() for x in soup.get_text("\n", strip=True).splitlines()]
    lines = [x for x in lines if x]
    starts = [i for i, line in enumerate(lines) if START.fullmatch(line)]
    candidates = []
    for start in starts:
        end = next((j for j in range(start + 1, len(lines)) if END.fullmatch(lines[j])), None)
        if end is None:
            continue
        section = "\n".join(lines[start:end])
        candidates.append({"text": section, "word_count": len(section.split()),
                           "start_line": start, "end_line": end})
    plausible = [c for c in candidates if 200 <= c["word_count"] <= 100000]
    chosen = max(plausible, key=lambda c: c["word_count"]) if plausible else None
    status = ("no_plausible_section" if not chosen else
              "ambiguous_needs_review" if len(plausible) > 1 else "candidate_needs_review")
    return {"extraction_status": status, "candidate_count": len(candidates),
            "plausible_count": len(plausible), "word_count": chosen["word_count"] if chosen else 0,
            "start_line": chosen["start_line"] if chosen else None,
            "end_line": chosen["end_line"] if chosen else None,
            "item1a_text": chosen["text"] if chosen else ""}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./sec_item1a_data")
    args = parser.parse_args()
    root = Path(args.output_dir)
    metadata = pd.read_csv(root / "metadata" / "pilot_filing_metadata.csv", dtype={"cik": str}, keep_default_na=False)
    rows = []
    for record in metadata.to_dict("records"):
        path = root / "raw" / f"{record['filing_id']}.html"
        result = {"extraction_status": "not_downloaded", "candidate_count": 0,
                  "plausible_count": 0, "word_count": 0, "start_line": None,
                  "end_line": None, "item1a_text": "", "extraction_error": ""}
        if path.exists():
            try:
                result.update(extract_candidate(path.read_bytes()))
            except Exception as exc:
                result["extraction_status"] = "error"
                result["extraction_error"] = str(exc)
        rows.append({**record, **result})
    df = pd.DataFrame(rows)
    df.to_csv(root / "item1a_pilot_candidates.csv", index=False)
    print(df["extraction_status"].value_counts().to_string() if not df.empty else "No records")
    print("Saved:", root / "item1a_pilot_candidates.csv")

if __name__ == "__main__":
    main()

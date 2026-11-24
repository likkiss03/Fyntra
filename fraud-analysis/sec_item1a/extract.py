"""Extract auditable Item 1A candidates from downloaded SEC 10-K filings.

Heuristic extraction only: every selected section requires manual verification.
"""
import argparse
import re
import warnings
from bisect import bisect_right
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ITEM_START = re.compile(r"^item\s*1\s*a\s*[.\-:–—]?\s*(.*)$", re.I)
ITEM_END = re.compile(r"^item\s*(?:1\s*b|2)\s*[.\-:–—]?\s*(.*)$", re.I)
RISK = re.compile(r"^risk\s*factors\b", re.I)
END_LABEL = re.compile(r"^(?:unresolved\s*staff\s*comments|properties)\b", re.I)
PAGE_NUMBER = re.compile(
    r"^\d{1,4}(?:\s*[-–—]\s*\d{1,4})?$"
)


def get_lines(content):
    soup = BeautifulSoup(content, "lxml")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    return [line for part in soup.get_text("\n", strip=True).splitlines()
            if (line := re.sub(r"\s+", " ", part).strip())]


def is_table_of_contents_start(lines, start_index, body_start):
    """
    Detect Item 1A headings inside a table of contents.

    A TOC entry typically has a page number or page range
    immediately after its title, followed by another Item heading.
    """
    nearby = lines[body_start:body_start + 6]

    page_number = re.compile(
        r"^\d{1,4}(?:\s*[-–—]\s*\d{1,4})?$"
    )

    item_heading = re.compile(
        r"^item\s*(?:1\s*b|2)\b",
        re.IGNORECASE
    )

    has_page_reference = any(
        page_number.fullmatch(line)
        for line in nearby[:3]
    )

    has_next_item = any(
        item_heading.match(line)
        for line in nearby
    )

    return has_page_reference and has_next_item


def normalized(s):
    return re.sub(r"[^a-z]", "", s.lower())


def title_match(lines, i, kind):
    """Return (body_start, inline_remainder, evidence) or None.

    Require a section title (possibly split over 2 lines) to reject
    cross-references and running page headers.
    """
    match = (ITEM_START if kind == "start" else ITEM_END).match(lines[i])
    if not match:
        return None
    remainder = match.group(1).strip()
    target = "riskfactors" if kind == "start" else None
    # An Item 1A line with trailing prose is not a heading.
    # Only a short remainder that contributes to the heading is allowed.
    if len(remainder) > 55:
        return None
    pieces = ([remainder] if remainder else [])
    for n in range(0, 3):
        parts = pieces + lines[i + 1:i + 1 + n]
        joined = normalized(" ".join(parts))
        if kind == "start":
            if joined == target:
                body_start = i + 1 + n
                # TOC entries usually have a page number directly after title.
                if body_start < len(lines) and PAGE_NUMBER.fullmatch(lines[body_start]):
                    return None
                if body_start < len(lines) and lines[body_start].lstrip().startswith(('" under the sub-caption', '” under the sub-caption')):
                    return None
                return body_start, "", "risk_factors_heading"
        else:
            if joined in {"unresolvedstaffcomments", "properties"}:
                body_start = i + 1 + n
                if body_start < len(lines) and PAGE_NUMBER.fullmatch(lines[body_start]):
                    return None
                return body_start, "", "end_heading"
    return None


def extract_candidate(content):
    lines = get_lines(content)
    starts = [(i, match) for i in range(len(lines))
              if (match := title_match(lines, i, "start")) is not None]
    ends = [i for i in range(len(lines)) if title_match(lines, i, "end") is not None]
    candidates = []
    for start, (body_start, _, evidence) in starts:

    # Reject Item 1A headings found in the table of contents.
        if is_table_of_contents_start(lines, start, body_start):
            continue

        pos = bisect_right(ends, start)
        if pos >= len(ends):
            continue
        end = ends[pos]
        if end <= body_start:
            continue
        text = "\n".join(lines[body_start:end]).strip()
        count = len(text.split())
        candidates.append({"text": text, "word_count": count,
                           "start_line": start, "end_line": end,
                           "heading_evidence": evidence})
    plausible = [c for c in candidates if 200 <= c["word_count"] <= 100000]
    chosen = max(plausible, key=lambda c: c["word_count"]) if plausible else None
    status = ("no_plausible_section" if chosen is None else
              "ambiguous_needs_review" if len(plausible) > 1 else
              "candidate_needs_review")
    return {"extraction_status": status, "candidate_count": len(candidates),
            "plausible_count": len(plausible),
            "word_count": chosen["word_count"] if chosen else 0,
            "start_line": chosen["start_line"] if chosen else None,
            "end_line": chosen["end_line"] if chosen else None,
            "item1a_text": chosen["text"] if chosen else "",
            "all_candidates": candidates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./sec_item1a_data")
    args = parser.parse_args()
    root = Path(args.output_dir)
    metadata = pd.read_csv(root / "metadata" / "pilot_filing_metadata.csv",
                           dtype={"cik": str}, keep_default_na=False)
    rows, candidate_rows = [], []
    for record in metadata.to_dict("records"):
        path = root / "raw" / f"{record['filing_id']}.html"
        result = {"extraction_status": "not_downloaded", "candidate_count": 0,
                  "plausible_count": 0, "word_count": 0, "start_line": None,
                  "end_line": None, "item1a_text": "", "extraction_error": ""}
        if path.exists():
            try:
                found = extract_candidate(path.read_bytes())
                for index, candidate in enumerate(found.pop("all_candidates"), 1):
                    candidate_rows.append({"filing_id": record["filing_id"],
                                           "company": record["company"],
                                           "candidate_number": index, **candidate})
                result.update(found)
            except Exception as exc:
                result["extraction_status"] = "error"
                result["extraction_error"] = str(exc)
        rows.append({**record, **result})
    df = pd.DataFrame(rows)
    df.to_csv(root / "item1a_pilot_candidates.csv", index=False)
    qa = root / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidate_rows, columns=["filing_id", "company", "candidate_number",
               "text", "word_count", "start_line", "end_line", "heading_evidence"]).to_csv(
                   qa / "all_extraction_candidates.csv", index=False)
    print(df["extraction_status"].value_counts().to_string() if not df.empty else "No records")
    print("Saved:", root / "item1a_pilot_candidates.csv")
    print("Candidate details:", qa / "all_extraction_candidates.csv")
    print("All extracted sections require manual verification.")


if __name__ == "__main__":
    main()

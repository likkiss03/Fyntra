
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import re

ROOT = Path("sec_item1a_data")
RAW = ROOT / "raw"

metadata = pd.read_csv(
    ROOT / "metadata" / "pilot_filing_metadata.csv",
    dtype={"cik": str}
)

results = []

for _, row in metadata.iterrows():
    path = RAW / f"{row['filing_id']}.html"

    if not path.exists():
        continue

    content = path.read_bytes()
    soup = BeautifulSoup(content, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n", strip=True)

    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
    ]

    lines = [line for line in lines if line]

    for i, line in enumerate(lines):
        if re.search(
            r"\bitem\s*1\s*[\.\-]?\s*a\b",
            line,
            re.IGNORECASE
        ):
            results.append({
                "company": row["company"],
                "filing_id": row["filing_id"],
                "line_number": i,
                "matched_line": line[:250],
                "previous_line": (
                    lines[i - 1][:150] if i > 0 else ""
                ),
                "next_line": (
                    lines[i + 1][:150]
                    if i + 1 < len(lines) else ""
                )
            })

report = pd.DataFrame(results)

output = ROOT / "qa" / "heading_diagnostics.csv"
output.parent.mkdir(parents=True, exist_ok=True)
report.to_csv(output, index=False)

print("Saved:", output)

for company in [
    "JPMorgan Chase",
    "Microsoft",
    "Walmart"
]:
    print("\n" + "=" * 60)
    print(company)
    print("=" * 60)

    subset = report[report["company"] == company]

    if subset.empty:
        print("No Item 1A references detected.")
    else:
        print(
            subset[
                ["filing_id", "line_number",
                 "matched_line", "next_line"]
            ].head(20).to_string(index=False)
        )

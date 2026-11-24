
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import re

ROOT = Path("sec_item1a_data")

metadata = pd.read_csv(
    ROOT / "metadata/pilot_filing_metadata.csv"
)

targets = {
    "Microsoft": ["2019", "2023"],
    "Walmart": ["2022", "2023"]
}

for _, row in metadata.iterrows():
    company = row["company"]
    year = str(row["filing_date"])[:4]

    if company not in targets or year not in targets[company]:
        continue

    filing_id = row["filing_id"]
    path = ROOT / "raw" / f"{filing_id}.html"

    soup = BeautifulSoup(path.read_bytes(), "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in soup.get_text("\n", strip=True).splitlines()
    ]
    lines = [line for line in lines if line]

    print("\n" + "=" * 75)
    print(company, year, filing_id)
    print("=" * 75)

    for i, line in enumerate(lines):
        if re.search(
            r"\bitem\s*1\s*a\b|\bitem\s*1\s*b\b|\bitem\s*2\b",
            line,
            re.IGNORECASE
        ):
            if len(line) > 300:
                continue

            print(f"\nLINE {i}: {line[:200]}")

            for j in range(
                max(0, i - 2),
                min(len(lines), i + 4)
            ):
                print(f"  {j}: {lines[j][:180]}")

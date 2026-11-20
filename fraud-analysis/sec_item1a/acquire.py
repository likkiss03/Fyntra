"""Discover and download a small, auditable SEC 10-K pilot corpus."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

COMPANIES = {
    "Apple": "0000320193", "Microsoft": "0000789019",
    "Alphabet": "0001652044", "Amazon": "0001018724",
    "JPMorgan Chase": "0000019617", "Walmart": "0000104169",
}
FIELDS = ["filing_id", "company", "cik", "accession_number", "filing_date",
          "report_date", "form", "primary_document", "source_url"]

class SecClient:
    def __init__(self, email):
        if not email or "@" not in email:
            raise ValueError("Set SEC_CONTACT_EMAIL to a real contact email.")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"FyntraAcademicResearch/0.1 ({email})",
            "Accept-Encoding": "gzip, deflate",
        })

    def get(self, url):
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in {"www.sec.gov", "data.sec.gov"}:
            raise ValueError("Unexpected SEC URL")
        for attempt in range(3):
            time.sleep(0.6)
            try:
                response = self.session.get(url, timeout=(15, 60), allow_redirects=False)
                if response.status_code in (401, 403, 429):
                    raise RuntimeError(f"SEC access restriction: HTTP {response.status_code}; stop requests")
                if 300 <= response.status_code < 400:
                    raise RuntimeError(f"Unexpected redirect: HTTP {response.status_code}")
                if response.status_code in (500, 502, 503, 504) and attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
        raise RuntimeError("SEC request failed")

def discover(client, company, cik, start_year, end_year, limit):
    payload = client.get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    filings = payload.get("filings", {})
    blocks = [filings.get("recent", {})]
    for historical in filings.get("files", []):
        if historical.get("filingTo", "") < f"{start_year}-01-01":
            continue
        if historical.get("filingFrom", "") > f"{end_year}-12-31":
            continue
        blocks.append(client.get("https://data.sec.gov/submissions/" + historical["name"]).json())
    frames = [pd.DataFrame(b) for b in blocks if b.get("accessionNumber")]
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    df = df.loc[(df["form"] == "10-K") &
                df["filingDate"].between(f"{start_year}-01-01", f"{end_year}-12-31")]
    df = df.drop_duplicates("accessionNumber").sort_values("filingDate").head(limit)
    results = []
    for _, row in df.iterrows():
        accession = str(row["accessionNumber"])
        accession_clean = accession.replace("-", "")
        document = str(row["primaryDocument"])
        results.append({
            "filing_id": f"{cik}_{accession_clean}", "company": company, "cik": cik,
            "accession_number": accession, "filing_date": row["filingDate"],
            "report_date": row.get("reportDate", ""), "form": "10-K",
            "primary_document": document,
            "source_url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{document}",
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--max-per-company", type=int, default=5)
    parser.add_argument("--output-dir", default="./sec_item1a_data")
    args = parser.parse_args()
    if args.start_year > args.end_year or args.max_per_company < 1:
        parser.error("Invalid year range or max-per-company")
    client = SecClient(os.environ.get("SEC_CONTACT_EMAIL"))
    root = Path(args.output_dir)
    raw_dir, metadata_dir = root / "raw", root / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    discovered, errors = [], []
    for company, cik in COMPANIES.items():
        print(f"Discovering {company}...", flush=True)
        try:
            records = discover(client, company, cik, args.start_year, args.end_year, args.max_per_company)
            discovered.extend(records)
            print(f"  {len(records)} filings", flush=True)
        except Exception as exc:
            errors.append({"company": company, "error": str(exc)})
            print(f"  ERROR: {exc}", flush=True)
            if "SEC access restriction" in str(exc):
                break
    pd.DataFrame(discovered, columns=FIELDS).to_csv(metadata_dir / "pilot_filing_metadata.csv", index=False)
    (metadata_dir / "discovery_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    manifest = []
    for record in discovered:
        path = raw_dir / (record["filing_id"] + ".html")
        try:
            if path.exists() and path.stat().st_size > 1000:
                content, status = path.read_bytes(), "cached"
            else:
                response = client.get(record["source_url"])
                content = response.content
                head = content[:5000].lower()
                if len(content) < 1000 or not any(x in head for x in (b"<html", b"<!doctype", b"<?xml", b"<document")):
                    raise ValueError("Unexpected or suspiciously small filing document")
                if b"access denied" in head or b"request rate threshold exceeded" in head:
                    raise RuntimeError("SEC access restriction in response body")
                path.write_bytes(content)
                status = "downloaded"
            manifest.append({"filing_id": record["filing_id"], "status": status,
                             "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                             "raw_path": str(path), "error": ""})
            print(f"{status}: {record['filing_id']}", flush=True)
        except Exception as exc:
            manifest.append({"filing_id": record["filing_id"], "status": "failed",
                             "bytes": 0, "sha256": "", "raw_path": "", "error": str(exc)})
            print(f"FAILED: {record['filing_id']}: {exc}", flush=True)
            if "SEC access restriction" in str(exc):
                break
        finally:
            pd.DataFrame(manifest).to_csv(metadata_dir / "download_manifest.csv", index=False)
    print(f"Discovered {len(discovered)}; downloaded/cached {sum(r['status'] != 'failed' for r in manifest)}")
    print("Output:", root.resolve())

if __name__ == "__main__":
    main()

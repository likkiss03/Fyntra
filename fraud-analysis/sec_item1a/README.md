# SEC 10-K Item 1A pilot (Fyntra)

Small engineering pilot: 6 SEC registrants, up to 5 filings per registrant, filing dates 2019–2023. **Not a representative fraud dataset.**

## Setup (VS Code terminal)

From `Fyntra/fraud-analysis/sec_item1a`:

```bash
python -m pip install -r requirements.txt
```

Set a genuine contact email for SEC's User-Agent policy (do not commit it):

- PowerShell: `$env:SEC_CONTACT_EMAIL="you@example.com"`
- macOS/Linux: `export SEC_CONTACT_EMAIL="you@example.com"`

Run the pipeline in order:

```bash
python acquire.py --output-dir ./sec_item1a_data
python extract.py --output-dir ./sec_item1a_data
python validate.py --output-dir ./sec_item1a_data
```

Open `sec_item1a_data/qa/manual_review.csv`, compare **every** candidate with its `source_url`, then enter:

- `correct_start`: `yes` or `no`
- `correct_end`: `yes` or `no`
- `toc_contamination`: `yes` or `no`
- `review_decision`: `accept`, `reject`, or `needs_fix`
- `review_notes`: explanation

Finally:

```bash
python build_corpus.py --output-dir ./sec_item1a_data
```

Only records marked `accept`, with correct boundaries and no TOC contamination, enter `item1a_pilot_verified.csv`.

## Google Colab

Clone your GitHub repository and mount Google Drive, then run scripts with `--output-dir /content/drive/MyDrive/FYP_results/SEC_Item1A_Pilot`. Set `SEC_CONTACT_EMAIL` in the notebook session, not in source code.

## Outputs

- `metadata/pilot_filing_metadata.csv`: SEC filing metadata and source URLs
- `metadata/download_manifest.csv`: status, SHA-256, raw path
- `raw/*.html`: downloaded filing documents
- `item1a_pilot_candidates.csv`: **unverified** extracted sections
- `qa/extraction_quality_report.csv`: extraction flags
- `qa/manual_review.csv`: human audit worksheet
- `item1a_pilot_verified.csv`: accepted pilot sections only

## Limitations

This heading-based extractor is deliberately conservative. Filings can contain TOC matches, split headings, alternate section formatting, or absent Item 1B sections. Automated QA does **not** establish extraction accuracy; human source-document review is required. The selection uses **filing dates**, not fiscal years. SEC access can be restricted from shared cloud IPs; respect the SEC's access rules and stop on 401/403/429.

Do not push raw SEC filings, generated CSVs, contact emails, or research datasets to GitHub. Use GitHub for code and Google Drive or local ignored folders for data.

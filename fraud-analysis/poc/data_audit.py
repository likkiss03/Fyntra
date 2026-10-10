"""
Dataset Quality Audit for the SEC 10-K MD&A Fraud Dataset.

Purpose
-------
Run this BEFORE B1/B2 to identify possible data-quality,
NLP-quality, class-imbalance, duplication, leakage and
temporal-label issues.

IMPORTANT:
This script DOES NOT modify or delete any data.
It only reports potential problems.

Dataset:
    fraud_text.csv

Main columns:
    mda         -> MD&A text
    fraudulent  -> evaluation label
    cik         -> company identifier
    name        -> company name
    filing_date -> filing date
"""

from pathlib import Path
import html
import re
import unicodedata

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "fraud_text.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "data_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COLUMN = "mda"
LABEL_COLUMN = "fraudulent"
COMPANY_ID_COLUMN = "cik"
COMPANY_NAME_COLUMN = "name"
DATE_COLUMN = "filing_date"

# Short documents can be problematic for NLP.
SHORT_TEXT_THRESHOLD = 200

# Very long documents deserve inspection.
LONG_TEXT_THRESHOLD = 20_000

# Number of suspicious examples to save.
MAX_EXAMPLES = 100


# ============================================================
# HELPERS
# ============================================================

def heading(title):
    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


def safe_percentage(value, total):
    if total == 0:
        return 0.0

    return (value / total) * 100


def normalise_text(text):
    """
    Used ONLY for duplicate detection.

    Does not modify the original dataset.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities such as &amp;
    text = html.unescape(text)

    # Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def word_count(text):
    if pd.isna(text):
        return 0

    return len(str(text).split())


def preview(text, length=300):
    if pd.isna(text):
        return ""

    text = re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()

    return text[:length]


# ============================================================
# LOAD DATA
# ============================================================

heading("DATASET AUDIT")

print(f"Loading:\n{DATA_PATH}")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("\nDataset loaded successfully.")


# ============================================================
# 1. DATASET SHAPE
# ============================================================

heading("1. DATASET SHAPE")

rows, columns = df.shape

print(f"Rows:    {rows:,}")
print(f"Columns: {columns:,}")

print("\nColumns:")

for column in df.columns:
    print(f"  - {column}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    TEXT_COLUMN,
    LABEL_COLUMN,
    COMPANY_ID_COLUMN,
    COMPANY_NAME_COLUMN,
    DATE_COLUMN,
]

missing_required_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_required_columns:

    raise ValueError(
        "Required columns are missing:\n"
        + "\n".join(missing_required_columns)
    )


# ============================================================
# 2. MISSING DATA
# ============================================================

heading("2. MISSING DATA")

missing_report = []

for column in df.columns:

    null_count = int(
        df[column].isna().sum()
    )

    empty_count = 0
    whitespace_count = 0

    if (
        pd.api.types.is_object_dtype(df[column])
        or pd.api.types.is_string_dtype(df[column])
    ):

        values = df[column].fillna("").astype(str)

        empty_count = int(
            (values == "").sum()
        )

        whitespace_count = int(
            (
                (values != "")
                & values.str.fullmatch(r"\s+")
            ).sum()
        )

    missing_report.append({
        "column": column,
        "null_nan": null_count,
        "empty_string": empty_count,
        "whitespace_only": whitespace_count,
        "null_percent": safe_percentage(
            null_count,
            len(df)
        ),
    })


missing_df = pd.DataFrame(
    missing_report
)

missing_df.to_csv(
    OUTPUT_DIR / "missing_data_report.csv",
    index=False
)

print(
    missing_df[
        (
            missing_df["null_nan"] > 0
        )
        |
        (
            missing_df["empty_string"] > 0
        )
        |
        (
            missing_df["whitespace_only"] > 0
        )
    ].to_string(index=False)
)


# ============================================================
# CRITICAL COLUMN MISSINGNESS
# ============================================================

print("\nCritical fields:")

for column in required_columns:

    null_count = int(
        df[column].isna().sum()
    )

    print(
        f"{column:<20} "
        f"{null_count:>8,} missing "
        f"({safe_percentage(null_count, len(df)):.2f}%)"
    )


# ============================================================
# 3. LABEL / CLASS IMBALANCE
# ============================================================

heading("3. CLASS DISTRIBUTION")

label_numeric = pd.to_numeric(
    df[LABEL_COLUMN],
    errors="coerce"
)

label_counts = (
    label_numeric
    .value_counts(dropna=False)
    .sort_index()
)

print(label_counts)

valid_labels = label_numeric[
    label_numeric.isin([0, 1])
]

print(
    f"\nValid binary labels: "
    f"{len(valid_labels):,}"
)

if len(valid_labels) > 0:

    normal_count = int(
        (valid_labels == 0).sum()
    )

    fraud_count = int(
        (valid_labels == 1).sum()
    )

    print(
        f"\nClass 0 / Non-fraud: "
        f"{normal_count:,} "
        f"({safe_percentage(normal_count, len(valid_labels)):.2f}%)"
    )

    print(
        f"Class 1 / Fraud:     "
        f"{fraud_count:,} "
        f"({safe_percentage(fraud_count, len(valid_labels)):.2f}%)"
    )

    if fraud_count > 0:

        ratio = normal_count / fraud_count

        print(
            f"\nApproximate imbalance ratio: "
            f"{ratio:.1f}:1"
        )

        if ratio >= 10:

            print(
                "WARNING: Strong class imbalance detected."
            )

            print(
                "Accuracy should NOT be used as the main "
                "evaluation metric."
            )


# ============================================================
# 4. EXACT ROW DUPLICATES
# ============================================================

heading("4. EXACT ROW DUPLICATES")

exact_duplicates = df.duplicated(
    keep=False
)

exact_duplicate_count = int(
    exact_duplicates.sum()
)

print(
    f"Rows involved in exact duplicates: "
    f"{exact_duplicate_count:,}"
)

print(
    f"Percentage: "
    f"{safe_percentage(exact_duplicate_count, len(df)):.2f}%"
)

if exact_duplicate_count > 0:

    df[
        exact_duplicates
    ].head(
        MAX_EXAMPLES
    ).to_csv(
        OUTPUT_DIR
        / "exact_duplicate_examples.csv",
        index=False
    )


# ============================================================
# 5. DUPLICATE MD&A TEXT
# ============================================================

heading("5. DUPLICATE MD&A TEXT")

print(
    "Normalising text for duplicate detection..."
)

df["_normalised_mda"] = (
    df[TEXT_COLUMN]
    .apply(normalise_text)
)

valid_text_mask = (
    df["_normalised_mda"] != ""
)

text_duplicates = (
    df.loc[
        valid_text_mask,
        "_normalised_mda"
    ]
    .duplicated(
        keep=False
    )
)

duplicate_indices = (
    text_duplicates[
        text_duplicates
    ].index
)

duplicate_text_rows = df.loc[
    duplicate_indices
].copy()

print(
    f"Rows involved in duplicate MD&A text: "
    f"{len(duplicate_text_rows):,}"
)

print(
    f"Percentage of dataset: "
    f"{safe_percentage(len(duplicate_text_rows), len(df)):.2f}%"
)


if len(duplicate_text_rows) > 0:

    duplicate_text_rows[
        [
            COMPANY_ID_COLUMN,
            COMPANY_NAME_COLUMN,
            DATE_COLUMN,
            LABEL_COLUMN,
            "_normalised_mda",
        ]
    ].head(
        MAX_EXAMPLES
    ).to_csv(
        OUTPUT_DIR
        / "duplicate_text_examples.csv",
        index=False
    )


# ============================================================
# 6. CONFLICTING LABELS FOR IDENTICAL TEXT
# ============================================================

heading(
    "6. IDENTICAL TEXT WITH CONFLICTING LABELS"
)

duplicate_analysis = (
    df[
        df["_normalised_mda"] != ""
    ]
    .groupby("_normalised_mda")
    [LABEL_COLUMN]
    .nunique(dropna=True)
)

conflicting_texts = (
    duplicate_analysis[
        duplicate_analysis > 1
    ].index
)

conflicting_rows = df[
    df["_normalised_mda"]
    .isin(conflicting_texts)
].copy()

print(
    f"Unique duplicated texts with conflicting labels: "
    f"{len(conflicting_texts):,}"
)

print(
    f"Rows involved: "
    f"{len(conflicting_rows):,}"
)

if len(conflicting_rows) > 0:

    conflicting_rows[
        [
            COMPANY_ID_COLUMN,
            COMPANY_NAME_COLUMN,
            DATE_COLUMN,
            LABEL_COLUMN,
            TEXT_COLUMN,
        ]
    ].head(
        MAX_EXAMPLES
    ).to_csv(
        OUTPUT_DIR
        / "conflicting_label_duplicates.csv",
        index=False
    )

    print(
        "WARNING: Identical text with different labels "
        "should be investigated."
    )


# ============================================================
# 7. TEXT LENGTH
# ============================================================

heading("7. TEXT LENGTH DISTRIBUTION")

print("Calculating word counts...")

df["_audit_word_count"] = (
    df[TEXT_COLUMN]
    .apply(word_count)
)

valid_lengths = df.loc[
    df["_audit_word_count"] > 0,
    "_audit_word_count"
]

if len(valid_lengths) > 0:

    percentiles = valid_lengths.quantile(
        [
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )

    print(
        f"Minimum: {valid_lengths.min():,}"
    )

    print(
        f"Median:  {valid_lengths.median():,.0f}"
    )

    print(
        f"Mean:    {valid_lengths.mean():,.0f}"
    )

    print(
        f"Maximum: {valid_lengths.max():,}"
    )

    print("\nPercentiles:")

    print(percentiles.to_string())


short_documents = df[
    (
        df["_audit_word_count"] > 0
    )
    &
    (
        df["_audit_word_count"]
        < SHORT_TEXT_THRESHOLD
    )
]

long_documents = df[
    df["_audit_word_count"]
    > LONG_TEXT_THRESHOLD
]


print(
    f"\nDocuments under "
    f"{SHORT_TEXT_THRESHOLD} words: "
    f"{len(short_documents):,}"
)

print(
    f"Documents over "
    f"{LONG_TEXT_THRESHOLD:,} words: "
    f"{len(long_documents):,}"
)


short_documents[
    [
        COMPANY_ID_COLUMN,
        COMPANY_NAME_COLUMN,
        DATE_COLUMN,
        LABEL_COLUMN,
        "_audit_word_count",
        TEXT_COLUMN,
    ]
].head(
    MAX_EXAMPLES
).to_csv(
    OUTPUT_DIR
    / "short_document_examples.csv",
    index=False
)


long_documents[
    [
        COMPANY_ID_COLUMN,
        COMPANY_NAME_COLUMN,
        DATE_COLUMN,
        LABEL_COLUMN,
        "_audit_word_count",
    ]
].head(
    MAX_EXAMPLES
).to_csv(
    OUTPUT_DIR
    / "long_document_examples.csv",
    index=False
)


# ============================================================
# 8. TEXT ARTIFACTS / NOISE
# ============================================================

heading("8. NLP NOISE / ARTIFACTS")

artifact_patterns = {

    "html_tags":
        r"<[^>]+>",

    "html_entities":
        r"&(?:amp|lt|gt|quot|nbsp|#\d+);",

    "urls":
        r"https?://\S+|www\.\S+",

    "markdown_links":
        r"\[[^\]]+\]\([^)]+\)",

    "markdown_headers":
        r"(?m)^\s*#{1,6}\s+",

    "possible_logs":
        r"(?i)\b(?:error|exception|traceback|stack trace)\b",

    # Common mojibake patterns
    "mojibake":
        r"(?:Ã.|Â.|â€|â€™|â€œ|â€|ï¿½)",

    "replacement_character":
        "\ufffd",
}


artifact_summary = []

text_series = (
    df[TEXT_COLUMN]
    .fillna("")
    .astype(str)
)


for artifact_name, pattern in artifact_patterns.items():

    mask = text_series.str.contains(
        pattern,
        regex=True,
        na=False
    )

    count = int(mask.sum())

    artifact_summary.append({
        "artifact": artifact_name,
        "documents": count,
        "percentage": safe_percentage(
            count,
            len(df)
        ),
    })

    print(
        f"{artifact_name:<25} "
        f"{count:>8,} "
        f"({safe_percentage(count, len(df)):.2f}%)"
    )


pd.DataFrame(
    artifact_summary
).to_csv(
    OUTPUT_DIR
    / "text_artifact_report.csv",
    index=False
)


# ============================================================
# 9. NON-ASCII / UNUSUAL CHARACTERS
# ============================================================

heading("9. CHARACTER / ENCODING CHECK")

def non_ascii_ratio(text):

    if pd.isna(text):
        return 0.0

    text = str(text)

    if len(text) == 0:
        return 0.0

    non_ascii = sum(
        ord(char) > 127
        for char in text
    )

    return non_ascii / len(text)


df["_non_ascii_ratio"] = (
    df[TEXT_COLUMN]
    .apply(non_ascii_ratio)
)


print(
    "Documents containing at least one "
    "non-ASCII character:",
    int(
        (
            df["_non_ascii_ratio"] > 0
        ).sum()
    )
)

print(
    "Documents with >1% non-ASCII characters:",
    int(
        (
            df["_non_ascii_ratio"] > 0.01
        ).sum()
    )
)


# ============================================================
# 10. CASING / TOKEN QUALITY
# ============================================================

heading("10. BASIC VOCABULARY / CASING CHECK")

def uppercase_ratio(text):

    if pd.isna(text):
        return 0.0

    letters = [
        c
        for c in str(text)
        if c.isalpha()
    ]

    if not letters:
        return 0.0

    uppercase = sum(
        c.isupper()
        for c in letters
    )

    return uppercase / len(letters)


df["_uppercase_ratio"] = (
    df[TEXT_COLUMN]
    .apply(uppercase_ratio)
)


high_uppercase = df[
    df["_uppercase_ratio"] > 0.50
]

print(
    "Documents with >50% uppercase letters:",
    f"{len(high_uppercase):,}"
)


# ============================================================
# 11. SIMPLE LANGUAGE CONSISTENCY HEURISTIC
# ============================================================

heading("11. LANGUAGE CONSISTENCY HEURISTIC")

print(
    "This is a heuristic only; it is NOT a full "
    "language-detection model."
)


common_english_words = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "for",
    "our",
    "we",
    "company",
    "year",
    "financial",
    "business",
    "operations",
    "management",
}


def english_indicator(text):

    if pd.isna(text):
        return 0.0

    words = re.findall(
        r"[A-Za-z]+",
        str(text).lower()
    )

    if not words:
        return 0.0

    # Limit work for huge documents.
    words = words[:2000]

    common_count = sum(
        word in common_english_words
        for word in words
    )

    return common_count / len(words)


df["_english_indicator"] = (
    df[TEXT_COLUMN]
    .apply(english_indicator)
)


possible_non_english = df[
    (
        df["_audit_word_count"] >= 50
    )
    &
    (
        df["_english_indicator"] < 0.005
    )
]


print(
    "Documents flagged for manual language review:",
    f"{len(possible_non_english):,}"
)


possible_non_english[
    [
        COMPANY_ID_COLUMN,
        COMPANY_NAME_COLUMN,
        DATE_COLUMN,
        "_english_indicator",
        TEXT_COLUMN,
    ]
].head(
    MAX_EXAMPLES
).to_csv(
    OUTPUT_DIR
    / "possible_language_issues.csv",
    index=False
)


# ============================================================
# 12. POTENTIAL LABEL LEAKAGE
# ============================================================

heading("12. POTENTIAL TEXT / LABEL LEAKAGE")

"""
These are NOT automatically leakage.

They are terms worth investigating because they may make
fraudulent filings trivially identifiable if they were inserted
after the original filing or come from enforcement metadata.
"""

leakage_terms = {

    "fraud":
        r"\bfraud(?:ulent)?\b",

    "sec_enforcement":
        r"\benforcement action\b",

    "aaer":
        r"\bAAER\b",

    "restatement":
        r"\brestatement\b|\brestated\b",

    "investigation":
        r"\binvestigation\b|\binvestigated\b",

    "securities_exchange_commission":
        (
            r"\bsecurities and exchange "
            r"commission\b"
        ),

    "accounting_irregularity":
        r"\baccounting irregularit(?:y|ies)\b",
}


leakage_report = []


for term_name, pattern in leakage_terms.items():

    mask = text_series.str.contains(
        pattern,
        case=False,
        regex=True,
        na=False
    )

    total_matches = int(
        mask.sum()
    )

    fraud_matches = int(
        (
            mask
            &
            (
                label_numeric == 1
            )
        ).sum()
    )

    nonfraud_matches = int(
        (
            mask
            &
            (
                label_numeric == 0
            )
        ).sum()
    )

    leakage_report.append({

        "term": term_name,

        "total_documents":
            total_matches,

        "fraud_documents":
            fraud_matches,

        "nonfraud_documents":
            nonfraud_matches,
    })


leakage_df = pd.DataFrame(
    leakage_report
)

print(
    leakage_df.to_string(
        index=False
    )
)


leakage_df.to_csv(
    OUTPUT_DIR
    / "potential_leakage_terms.csv",
    index=False
)


# ============================================================
# 13. TEMPORAL LABEL DISTRIBUTION
# ============================================================

heading("13. TEMPORAL LABEL DISTRIBUTION")

df["_parsed_date"] = pd.to_datetime(
    df[DATE_COLUMN],
    errors="coerce"
)

df["_filing_year"] = (
    df["_parsed_date"].dt.year
)


yearly_distribution = (
    df[
        df["_filing_year"].notna()
    ]
    .groupby("_filing_year")
    [LABEL_COLUMN]
    .agg(
        total="count",
        fraud="sum",
        fraud_rate="mean"
    )
)


print(
    yearly_distribution.to_string()
)


yearly_distribution.to_csv(
    OUTPUT_DIR
    / "yearly_label_distribution.csv"
)


# ============================================================
# 14. TEMPORAL DUPLICATE LEAKAGE
# ============================================================

heading("14. TRAIN / TEST DUPLICATE LEAKAGE CHECK")

TRAIN_END_YEAR = 2017
TEST_START_YEAR = 2018
TEST_END_YEAR = 2023


train_texts = set(
    df.loc[
        (
            df["_filing_year"]
            <= TRAIN_END_YEAR
        )
        &
        (
            df["_normalised_mda"]
            != ""
        ),
        "_normalised_mda"
    ]
)


test_mask = (
    (
        df["_filing_year"]
        >= TEST_START_YEAR
    )
    &
    (
        df["_filing_year"]
        <= TEST_END_YEAR
    )
)


test_df = df[
    test_mask
].copy()


test_df[
    "_appears_in_training"
] = (
    test_df[
        "_normalised_mda"
    ].isin(
        train_texts
    )
)


leaked_test_rows = test_df[
    test_df[
        "_appears_in_training"
    ]
]


print(
    "Test documents whose identical normalised "
    "text exists in training:"
)

print(
    f"{len(leaked_test_rows):,}"
)


print(
    "Percentage of temporal test set:",
    f"{safe_percentage(len(leaked_test_rows), len(test_df)):.2f}%"
)


if len(leaked_test_rows) > 0:

    leaked_test_rows[
        [
            COMPANY_ID_COLUMN,
            COMPANY_NAME_COLUMN,
            DATE_COLUMN,
            LABEL_COLUMN,
            TEXT_COLUMN,
        ]
    ].head(
        MAX_EXAMPLES
    ).to_csv(
        OUTPUT_DIR
        / "train_test_duplicate_leakage.csv",
        index=False
    )


# ============================================================
# 15. SAME COMPANY ACROSS TRAIN / TEST
# ============================================================

heading("15. COMPANY OVERLAP ACROSS TEMPORAL SPLIT")

train_companies = set(
    df.loc[
        df["_filing_year"]
        <= TRAIN_END_YEAR,
        COMPANY_ID_COLUMN
    ].dropna()
)


test_companies = set(
    test_df[
        COMPANY_ID_COLUMN
    ].dropna()
)


company_overlap = (
    train_companies
    .intersection(
        test_companies
    )
)


print(
    f"Unique training companies: "
    f"{len(train_companies):,}"
)

print(
    f"Unique test companies: "
    f"{len(test_companies):,}"
)

print(
    f"Companies present in both: "
    f"{len(company_overlap):,}"
)

print(
    "\nNOTE: Company overlap is not automatically leakage."
)

print(
    "For temporal anomaly detection, seeing a company's "
    "historical filings may actually be intentional."
)


# ============================================================
# 16. SUMMARY REPORT
# ============================================================

heading("16. AUDIT SUMMARY")

summary = {

    "dataset_rows":
        len(df),

    "dataset_columns":
        len(df.columns),

    "missing_mda":
        int(
            df[TEXT_COLUMN]
            .isna()
            .sum()
        ),

    "fraud_count":
        int(
            (
                label_numeric == 1
            ).sum()
        ),

    "nonfraud_count":
        int(
            (
                label_numeric == 0
            ).sum()
        ),

    "exact_duplicate_rows":
        exact_duplicate_count,

    "duplicate_text_rows":
        len(
            duplicate_text_rows
        ),

    "conflicting_label_duplicate_rows":
        len(
            conflicting_rows
        ),

    "short_documents":
        len(
            short_documents
        ),

    "very_long_documents":
        len(
            long_documents
        ),

    "possible_language_issues":
        len(
            possible_non_english
        ),

    "train_test_duplicate_text_rows":
        len(
            leaked_test_rows
        ),
}


summary_df = pd.DataFrame(
    [
        {
            "check": key,
            "value": value
        }
        for key, value
        in summary.items()
    ]
)


print(
    summary_df.to_string(
        index=False
    )
)


summary_df.to_csv(
    OUTPUT_DIR
    / "audit_summary.csv",
    index=False
)


# ============================================================
# REMOVE TEMPORARY AUDIT COLUMNS
# ============================================================

print("\nAudit complete.")

print(
    "\nReports saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nIMPORTANT: No original data was modified."
)
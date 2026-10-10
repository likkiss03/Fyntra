from pathlib import Path
import sys
import json

import joblib
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from common.evaluation import evaluate_anomaly_scores


DATA_PATH = PROJECT_ROOT / "fraud_text.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "B1_tfidf_iforest"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TEXT_COLUMN = "mda"
LABEL_COLUMN = "fraudulent"
DATE_COLUMN = "filing_date"

TRAIN_END_YEAR = 2017

TEST_START_YEAR = 2018
TEST_END_YEAR = 2023

RANDOM_STATE = 42

MAX_FEATURES = 20_000
SVD_COMPONENTS = 300

# Initial PoC assumption.
# Later we will perform sensitivity analysis.
CONTAMINATION = 0.02


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("B1 - TF-IDF + Isolation Forest")
print("=" * 70)

print("\nLoading fraud_text.csv...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print(
    f"Original dataset: "
    f"{df.shape[0]} rows x "
    f"{df.shape[1]} columns"
)


# ============================================================
# CLEAN DATA
# ============================================================

df[DATE_COLUMN] = pd.to_datetime(
    df[DATE_COLUMN],
    errors="coerce"
)

df["filing_year"] = (
    df[DATE_COLUMN].dt.year
)

df[LABEL_COLUMN] = pd.to_numeric(
    df[LABEL_COLUMN],
    errors="coerce"
)


data = df[
    df[TEXT_COLUMN].notna()
    & df[DATE_COLUMN].notna()
    & df[LABEL_COLUMN].isin([0, 1])
].copy()


data[TEXT_COLUMN] = (
    data[TEXT_COLUMN]
    .astype(str)
    .str.replace(
        r"\s+",
        " ",
        regex=True
    )
    .str.strip()
)


# Remove extremely short MD&A sections
data["word_count_calculated"] = pd.to_numeric(
    data["word_count"],
    errors="coerce"
)

data = data[
    data["word_count_calculated"] >= 200
].copy()


data[LABEL_COLUMN] = (
    data[LABEL_COLUMN]
    .astype(int)
)


print(
    f"Usable filings after cleaning: "
    f"{len(data)}"
)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

train_df = data[
    data["filing_year"]
    <= TRAIN_END_YEAR
].copy()


test_df = data[
    (data["filing_year"] >= TEST_START_YEAR)
    &
    (data["filing_year"] <= TEST_END_YEAR)
].copy()


print("\nTEMPORAL SPLIT")
print("-" * 70)

print(
    f"Training period: 1994-{TRAIN_END_YEAR}"
)

print(
    f"Testing period: "
    f"{TEST_START_YEAR}-{TEST_END_YEAR}"
)

print(
    f"Training filings: "
    f"{len(train_df)}"
)

print(
    f"Testing filings: "
    f"{len(test_df)}"
)

# Labels are printed for descriptive/evaluation purposes only.
print(
    f"Known fraud in training population: "
    f"{train_df[LABEL_COLUMN].sum()}"
)

print(
    f"Known fraud in test population: "
    f"{test_df[LABEL_COLUMN].sum()}"
)

print(
    f"Test fraud ratio: "
    f"{test_df[LABEL_COLUMN].mean():.4f}"
)


# ============================================================
# TF-IDF
# ============================================================

print("\nFitting TF-IDF on TRAINING filings only...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=MAX_FEATURES,
    min_df=5,
    max_df=0.95,
    ngram_range=(1, 2),
    sublinear_tf=True
)


# FIT only on historical filings
X_train_tfidf = (
    vectorizer.fit_transform(
        train_df[TEXT_COLUMN]
    )
)


# TRANSFORM future filings
X_test_tfidf = (
    vectorizer.transform(
        test_df[TEXT_COLUMN]
    )
)


print(
    f"Train TF-IDF shape: "
    f"{X_train_tfidf.shape}"
)

print(
    f"Test TF-IDF shape: "
    f"{X_test_tfidf.shape}"
)


# ============================================================
# SVD
# ============================================================

n_components = min(
    SVD_COMPONENTS,
    X_train_tfidf.shape[1] - 1,
    X_train_tfidf.shape[0] - 1
)


print(
    f"\nFitting TruncatedSVD "
    f"with {n_components} components..."
)


svd = TruncatedSVD(
    n_components=n_components,
    random_state=RANDOM_STATE
)


# FIT only on training data
X_train = svd.fit_transform(
    X_train_tfidf
)


# Only transform test
X_test = svd.transform(
    X_test_tfidf
)


print(
    f"Train representation: "
    f"{X_train.shape}"
)

print(
    f"Test representation: "
    f"{X_test.shape}"
)

print(
    "Explained variance: "
    f"{svd.explained_variance_ratio_.sum():.4f}"
)


# ============================================================
# ISOLATION FOREST
# ============================================================

print(
    "\nTraining Isolation Forest "
    "WITHOUT fraud labels..."
)


iforest = IsolationForest(
    n_estimators=300,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# CRITICAL:
#
# fraudulent is NOT passed here.
#
iforest.fit(
    X_train
)


print(
    "Isolation Forest fitted."
)


# ============================================================
# SCORE FUTURE FILINGS
# ============================================================

print(
    "\nGenerating anomaly scores "
    "for future filings..."
)


# sklearn:
# positive / larger = normal
#
# We negate it so:
# larger = MORE anomalous

test_scores = (
    -iforest.decision_function(
        X_test
    )
)


test_df["anomaly_score"] = (
    test_scores
)


# ============================================================
# EVALUATION
# ============================================================

print(
    "\nEvaluating against hidden "
    "fraud labels..."
)


results = evaluate_anomaly_scores(
    y_true=(
        test_df[LABEL_COLUMN]
        .values
    ),
    scores=test_scores,
    output_dir=OUTPUT_DIR,
    experiment_name=(
        "B1 - TF-IDF + SVD + "
        "Isolation Forest "
        "(Temporal Test)"
    )
)


# ============================================================
# SAVE ALL TEST SCORES
# ============================================================

score_columns = [
    "cik",
    "name",
    "filing_type",
    "filing_date",
    "filing_year",
    LABEL_COLUMN,
    "anomaly_score"
]


test_df[
    score_columns
].sort_values(
    "anomaly_score",
    ascending=False
).to_csv(
    OUTPUT_DIR
    / "all_test_anomaly_scores.csv",
    index=False
)


# ============================================================
# SAVE TOP 200 ANOMALIES
# ============================================================

top_anomalies = (
    test_df
    .sort_values(
        "anomaly_score",
        ascending=False
    )
    .head(200)
    .copy()
)


top_anomalies[
    "mda_preview"
] = (
    top_anomalies["mda"]
    .str[:1000]
)


top_columns = [
    "cik",
    "name",
    "filing_type",
    "filing_date",
    "filing_year",
    "fraudulent",
    "anomaly_score",
    "mda_preview"
]


top_anomalies[
    top_columns
].to_csv(
    OUTPUT_DIR
    / "top_200_anomalies.csv",
    index=False
)


# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(
    vectorizer,
    OUTPUT_DIR
    / "tfidf_vectorizer.joblib"
)

joblib.dump(
    svd,
    OUTPUT_DIR
    / "svd.joblib"
)

joblib.dump(
    iforest,
    OUTPUT_DIR
    / "isolation_forest.joblib"
)


print("\n" + "=" * 70)
print("B1 COMPLETE")
print("=" * 70)

print(
    f"\nResults saved in:\n"
    f"{OUTPUT_DIR}"
)

config = {
    "experiment": "B1",
    "representation": "TF-IDF + TruncatedSVD",
    "detector": "IsolationForest",
    "train_end_year": TRAIN_END_YEAR,
    "test_start_year": TEST_START_YEAR,
    "test_end_year": TEST_END_YEAR,
    "max_features": MAX_FEATURES,
    "svd_components": n_components,
    "contamination": CONTAMINATION,
    "random_state": RANDOM_STATE,
    "training_samples": len(train_df),
    "test_samples": len(test_df),
    "labels_used_for_training": False
}

with open(
    OUTPUT_DIR / "experiment_config.json",
    "w"
) as f:
    json.dump(config, f, indent=4)
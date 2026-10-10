from pathlib import Path
import sys
import json

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import IsolationForest
from transformers import AutoModel, AutoTokenizer


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
    / "B2_mbert_iforest"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EMBEDDING_PATH = (
    OUTPUT_DIR
    / "modernbert_embeddings_1994_2023.npy"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "modernbert_embedding_metadata_1994_2023.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "answerdotai/ModernBERT-base"

TEXT_COLUMN = "mda"
LABEL_COLUMN = "fraudulent"
DATE_COLUMN = "filing_date"

TRAIN_END_YEAR = 2017

TEST_START_YEAR = 2018
TEST_END_YEAR = 2023

RANDOM_STATE = 42

CONTAMINATION = 0.02

# Number of original tokens placed into each chunk.
#
# ModernBERT supports longer contexts, but 1024 is used here
# to reduce memory requirements and keep B2 practical.
CHUNK_SIZE = 1024

# Maximum number of chunks used from each document.
#
# Maximum text considered per document is therefore
# approximately:
#
#     8 * 1024 = 8192 tokens
#
# This is a computational constraint for the PoC and should
# be reported as a limitation of B2.
MAX_CHUNKS_PER_DOCUMENT = 8


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)

torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        RANDOM_STATE
    )


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("B2 - ModernBERT + Isolation Forest")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    print(
        "\nWARNING:"
        "\nModernBERT embedding generation on CPU "
        "will be very slow."
        "\nA CUDA GPU / Google Colab is strongly "
        "recommended."
    )


# ============================================================
# MEAN POOLING
# ============================================================

def mean_pool(
    last_hidden_state,
    attention_mask,
):
    """
    Perform attention-mask-aware mean pooling.

    Input:
        last_hidden_state:
            [batch_size, sequence_length, hidden_size]

        attention_mask:
            [batch_size, sequence_length]

    Output:
        [batch_size, hidden_size]
    """

    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand_as(last_hidden_state)
        .float()
    )

    summed = (
        last_hidden_state * mask
    ).sum(dim=1)

    counts = (
        mask
        .sum(dim=1)
        .clamp(min=1e-9)
    )

    return summed / counts


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading fraud_text.csv...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

print(
    f"Original dataset: "
    f"{df.shape[0]:,} rows x "
    f"{df.shape[1]} columns"
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    TEXT_COLUMN,
    LABEL_COLUMN,
    DATE_COLUMN,
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Required columns are missing:\n"
        f"{missing_columns}\n\n"
        "Available columns:\n"
        f"{df.columns.tolist()}"
    )


# ============================================================
# BASIC CLEANING
# ============================================================

df[DATE_COLUMN] = pd.to_datetime(
    df[DATE_COLUMN],
    errors="coerce",
)

df["filing_year"] = (
    df[DATE_COLUMN].dt.year
)


df[LABEL_COLUMN] = pd.to_numeric(
    df[LABEL_COLUMN],
    errors="coerce",
)


data = df[
    df[TEXT_COLUMN].notna()
    & df[DATE_COLUMN].notna()
    & df[LABEL_COLUMN].isin([0, 1])
].copy()


# Normalize whitespace only.
#
# Do not aggressively remove numbers or financial terminology
# because ModernBERT should retain contextual information.

data[TEXT_COLUMN] = (
    data[TEXT_COLUMN]
    .astype(str)
    .str.replace(
        r"\s+",
        " ",
        regex=True,
    )
    .str.strip()
)


# ============================================================
# WORD COUNT FILTER
# ============================================================

# The dataset already contains word_count.
# Use it rather than scanning every long MD&A again.

if "word_count" in data.columns:

    data["word_count_calculated"] = (
        pd.to_numeric(
            data["word_count"],
            errors="coerce",
        )
    )

else:

    print(
        "\nword_count column not found."
        "\nCalculating word counts manually..."
    )

    data["word_count_calculated"] = (
        data[TEXT_COLUMN]
        .str.count(r"\S+")
    )


data = data[
    data["word_count_calculated"] >= 200
].copy()


data[LABEL_COLUMN] = (
    data[LABEL_COLUMN]
    .astype(int)
)


# ============================================================
# REMOVE 2024-2025 FROM MAIN EXPERIMENT
# ============================================================

# Main B2 experiment uses the same period as B1:
#
# Train: 1994-2017
# Test:  2018-2023
#
# 2024-2025 are excluded because recent enforcement-linked
# fraud labels may be incomplete/right-censored.

data = data[
    data["filing_year"]
    <= TEST_END_YEAR
].copy()


# ============================================================
# EXACT DUPLICATE REMOVAL
# ============================================================

before_duplicates = len(data)

data = (
    data
    .drop_duplicates()
    .copy()
)

removed_duplicates = (
    before_duplicates - len(data)
)

print(
    f"\nExact duplicate rows removed: "
    f"{removed_duplicates}"
)


# IMPORTANT:
# Reset index AFTER all filtering so embedding row i
# corresponds exactly to metadata row i.

data = data.reset_index(
    drop=True
)


print(
    f"Usable 1994-{TEST_END_YEAR} filings: "
    f"{len(data):,}"
)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

train_mask = (
    data["filing_year"]
    <= TRAIN_END_YEAR
)

test_mask = (
    (data["filing_year"] >= TEST_START_YEAR)
    &
    (data["filing_year"] <= TEST_END_YEAR)
)


train_df = (
    data.loc[train_mask]
    .copy()
)

test_df = (
    data.loc[test_mask]
    .copy()
)


print("\nTEMPORAL SPLIT")
print("-" * 70)

print(
    f"Training period: "
    f"1994-{TRAIN_END_YEAR}"
)

print(
    f"Testing period: "
    f"{TEST_START_YEAR}-{TEST_END_YEAR}"
)

print(
    f"Training filings: "
    f"{len(train_df):,}"
)

print(
    f"Testing filings: "
    f"{len(test_df):,}"
)

print(
    f"Known fraud in training population: "
    f"{train_df[LABEL_COLUMN].sum():,}"
)

print(
    f"Known fraud in test population: "
    f"{test_df[LABEL_COLUMN].sum():,}"
)

print(
    f"Test fraud ratio: "
    f"{test_df[LABEL_COLUMN].mean():.4f}"
)


if len(train_df) == 0:

    raise ValueError(
        "Training set is empty."
    )

if len(test_df) == 0:

    raise ValueError(
        "Test set is empty."
    )


# ============================================================
# SAVE EMBEDDING METADATA
# ============================================================

metadata_columns = []

for column in [
    "cik",
    "name",
    "filing_type",
    "filing_date",
    "filing_year",
    LABEL_COLUMN,
    "word_count_calculated",
]:

    if column in data.columns:
        metadata_columns.append(column)


data[
    metadata_columns
].to_csv(
    METADATA_PATH,
    index=False,
)


# ============================================================
# LOAD MODERNBERT
# ============================================================

print(
    f"\nLoading ModernBERT:"
    f"\n{MODEL_NAME}"
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# Use float16 on CUDA to reduce GPU memory usage.
# Keep float32 on CPU.

if DEVICE.type == "cuda":

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )

else:

    model = AutoModel.from_pretrained(
        MODEL_NAME,
    )


model = model.to(
    DEVICE
)

model.eval()


print("ModernBERT loaded successfully.")


# ============================================================
# CREATE DOCUMENT EMBEDDING
# ============================================================

def create_document_embedding(text):
    """
    Convert one MD&A document into one fixed-size
    ModernBERT contextual representation.

    Procedure:

    1. Tokenize entire document WITHOUT truncation.
    2. Split token IDs into 1024-token chunks.
    3. Use at most MAX_CHUNKS_PER_DOCUMENT chunks.
    4. Add required model special tokens.
    5. Encode each chunk using ModernBERT.
    6. Mean-pool token representations.
    7. Mean-pool all chunk representations.

    Fraud labels are never used.
    """

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )

    token_ids = encoded[
        "input_ids"
    ]


    if len(token_ids) == 0:

        raise ValueError(
            "Tokenizer returned zero tokens."
        )


    chunk_embeddings = []


    # --------------------------------------------------------
    # Split into chunks
    # --------------------------------------------------------

    for chunk_number, start in enumerate(
        range(
            0,
            len(token_ids),
            CHUNK_SIZE,
        )
    ):

        if (
            chunk_number
            >= MAX_CHUNKS_PER_DOCUMENT
        ):
            break


        chunk_ids = token_ids[
            start:start + CHUNK_SIZE
        ]


        # ----------------------------------------------------
        # Add special tokens manually through tokenizer
        # ----------------------------------------------------

        prepared_ids = (
            tokenizer
            .build_inputs_with_special_tokens(
                chunk_ids
            )
        )


        # ----------------------------------------------------
        # Convert directly to [1, sequence_length]
        #
        # This avoids the prepare_for_model +
        # unsqueeze shape issue.
        # ----------------------------------------------------

        input_ids = torch.tensor(
            [prepared_ids],
            dtype=torch.long,
            device=DEVICE,
        )


        attention_mask = torch.ones(
            input_ids.shape,
            dtype=torch.long,
            device=DEVICE,
        )


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        with torch.inference_mode():

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )


            chunk_embedding = mean_pool(
                outputs.last_hidden_state,
                attention_mask,
            )


        # Convert to CPU float32 NumPy array.
        #
        # float() is important if model used float16.

        chunk_embedding = (
            chunk_embedding
            .squeeze(0)
            .float()
            .cpu()
            .numpy()
        )


        chunk_embeddings.append(
            chunk_embedding
        )


    if len(chunk_embeddings) == 0:

        raise ValueError(
            "No chunk embeddings were created."
        )


    # Mean across chunks to obtain one vector
    # representing the complete processed portion
    # of the document.

    document_embedding = np.mean(
        np.stack(
            chunk_embeddings,
            axis=0,
        ),
        axis=0,
    )


    return document_embedding.astype(
        np.float32
    )


# ============================================================
# LOAD OR GENERATE EMBEDDINGS
# ============================================================

if EMBEDDING_PATH.exists():

    print(
        "\nCached ModernBERT embeddings found."
    )

    print(
        f"Loading:\n{EMBEDDING_PATH}"
    )


    X_embeddings = np.load(
        EMBEDDING_PATH
    )


    # --------------------------------------------------------
    # CACHE SAFETY CHECK
    # --------------------------------------------------------

    if len(X_embeddings) != len(data):

        raise ValueError(
            "\nCached embedding count does not match "
            "the current dataset.\n\n"
            f"Cached embeddings: {len(X_embeddings):,}\n"
            f"Current documents:  {len(data):,}\n\n"
            "Delete the cached .npy file and run B2 again:\n"
            f"{EMBEDDING_PATH}"
        )


else:

    print(
        "\nNo cached embeddings found."
    )

    print(
        "Generating ModernBERT embeddings..."
    )

    print(
        "\nThis is the expensive part of B2."
    )

    if DEVICE.type == "cpu":

        print(
            "\nWARNING: CPU detected."
            "\nConsider stopping and running this "
            "embedding stage on Google Colab GPU."
        )


    embeddings = []

    total_documents = len(data)


    for i, text in enumerate(
        data[TEXT_COLUMN]
    ):

        try:

            embedding = (
                create_document_embedding(
                    text
                )
            )

        except RuntimeError as error:

            # Give a useful message for GPU OOM errors.

            if (
                "out of memory"
                in str(error).lower()
            ):

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                raise RuntimeError(
                    "\nGPU out of memory while generating "
                    f"document {i}.\n"
                    "Reduce CHUNK_SIZE, for example from "
                    "1024 to 512, then delete the partial "
                    "embedding cache if one exists and "
                    "restart."
                ) from error

            raise


        embeddings.append(
            embedding
        )


        # Progress output

        if (
            i == 0
            or (i + 1) % 100 == 0
            or (i + 1) == total_documents
        ):

            print(
                f"Embedded "
                f"{i + 1:,}/"
                f"{total_documents:,} documents"
            )


    X_embeddings = np.stack(
        embeddings,
        axis=0,
    ).astype(
        np.float32
    )


    np.save(
        EMBEDDING_PATH,
        X_embeddings,
    )


    print(
        "\nEmbeddings saved successfully:"
    )

    print(
        EMBEDDING_PATH
    )


# ============================================================
# EMBEDDING VALIDATION
# ============================================================

print(
    f"\nEmbedding matrix shape: "
    f"{X_embeddings.shape}"
)


if len(X_embeddings) != len(data):

    raise ValueError(
        "Embedding count does not match "
        "the dataset."
    )


if not np.isfinite(
    X_embeddings
).all():

    raise ValueError(
        "Embedding matrix contains "
        "NaN or infinite values."
    )


# ============================================================
# TEMPORAL EMBEDDING SPLIT
# ============================================================

# IMPORTANT:
#
# Embeddings themselves do not use fraud labels.
#
# We now split them according to the exact same temporal
# design used by B1.


train_indices = np.flatnonzero(
    train_mask.to_numpy()
)

test_indices = np.flatnonzero(
    test_mask.to_numpy()
)


X_train = X_embeddings[
    train_indices
]

X_test = X_embeddings[
    test_indices
]


print("\nEMBEDDING SPLIT")
print("-" * 70)

print(
    f"Train embeddings: "
    f"{X_train.shape}"
)

print(
    f"Test embeddings:  "
    f"{X_test.shape}"
)


if len(X_train) != len(train_df):

    raise ValueError(
        "Training embedding alignment error."
    )


if len(X_test) != len(test_df):

    raise ValueError(
        "Test embedding alignment error."
    )


# ============================================================
# ISOLATION FOREST
# ============================================================

print(
    "\nTraining Isolation Forest "
    "on historical embeddings..."
)

print(
    "Fraud labels are NOT supplied "
    "to the model."
)


iforest = IsolationForest(
    n_estimators=300,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# CRITICAL:
#
# Train ONLY on 1994-2017 embeddings.
#
# y / fraudulent is NOT passed.

iforest.fit(
    X_train
)


print(
    "Isolation Forest fitted."
)


# ============================================================
# SCORE TEMPORAL TEST SET
# ============================================================

print(
    "\nGenerating anomaly scores "
    f"for {TEST_START_YEAR}-{TEST_END_YEAR}..."
)


# sklearn decision_function:
#
# larger = more normal
#
# Negate so:
#
# larger = more anomalous

test_scores = (
    -iforest.decision_function(
        X_test
    )
)


test_df[
    "anomaly_score"
] = test_scores


# ============================================================
# EVALUATION
# ============================================================

print(
    "\nEvaluating against hidden "
    "AAER fraud labels..."
)


results = evaluate_anomaly_scores(
    y_true=(
        test_df[LABEL_COLUMN]
        .to_numpy()
    ),
    scores=test_scores,
    output_dir=OUTPUT_DIR,
    experiment_name=(
        "B2 - ModernBERT + Isolation Forest "
        "(Temporal Test)"
    ),
)


# ============================================================
# SAVE ALL TEST ANOMALY SCORES
# ============================================================

score_columns = []

for column in [
    "cik",
    "name",
    "filing_type",
    "filing_date",
    "filing_year",
    LABEL_COLUMN,
    "word_count_calculated",
    "anomaly_score",
]:

    if column in test_df.columns:

        score_columns.append(
            column
        )


test_df[
    score_columns
].sort_values(
    "anomaly_score",
    ascending=False,
).to_csv(
    OUTPUT_DIR
    / "all_test_anomaly_scores.csv",
    index=False,
)


# ============================================================
# SAVE TOP 200 ANOMALIES
# ============================================================

top_anomalies = (
    test_df
    .sort_values(
        "anomaly_score",
        ascending=False,
    )
    .head(200)
    .copy()
)


top_anomalies[
    "mda_preview"
] = (
    top_anomalies[
        TEXT_COLUMN
    ]
    .str[:1000]
)


top_columns = []

for column in [
    "cik",
    "name",
    "filing_type",
    "filing_date",
    "filing_year",
    LABEL_COLUMN,
    "word_count_calculated",
    "anomaly_score",
    "mda_preview",
]:

    if column in top_anomalies.columns:

        top_columns.append(
            column
        )


top_anomalies[
    top_columns
].to_csv(
    OUTPUT_DIR
    / "top_200_anomalies.csv",
    index=False,
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    iforest,
    OUTPUT_DIR
    / "isolation_forest.joblib",
)


# ============================================================
# SAVE EXPERIMENT CONFIGURATION
# ============================================================

config = {

    "experiment_id": "B2",

    "experiment_name": (
        "ModernBERT + Isolation Forest "
        "(Temporal Test)"
    ),

    "dataset": "fraud_text.csv",

    "text_column": TEXT_COLUMN,

    "label_column": LABEL_COLUMN,

    "label_used_for_training": False,

    "model_name": MODEL_NAME,

    "representation": (
        "ModernBERT contextual hidden states "
        "+ token mean pooling "
        "+ chunk mean pooling"
    ),

    "detector": "IsolationForest",

    "train_period": (
        f"1994-{TRAIN_END_YEAR}"
    ),

    "test_period": (
        f"{TEST_START_YEAR}-{TEST_END_YEAR}"
    ),

    "excluded_recent_years": [
        2024,
        2025,
    ],

    "chunk_size": CHUNK_SIZE,

    "max_chunks_per_document": (
        MAX_CHUNKS_PER_DOCUMENT
    ),

    "maximum_processed_tokens_approx": (
        CHUNK_SIZE
        * MAX_CHUNKS_PER_DOCUMENT
    ),

    "contamination": CONTAMINATION,

    "n_estimators": 300,

    "random_state": RANDOM_STATE,

    "training_samples": int(
        len(train_df)
    ),

    "test_samples": int(
        len(test_df)
    ),

    "test_fraud_cases": int(
        test_df[
            LABEL_COLUMN
        ].sum()
    ),

    "embedding_dimensions": int(
        X_embeddings.shape[1]
    ),

    "device_used": str(
        DEVICE
    ),
}


with open(
    OUTPUT_DIR
    / "experiment_config.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        config,
        file,
        indent=4,
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("B2 COMPLETE")
print("=" * 70)


print(
    f"\nResults saved in:"
    f"\n{OUTPUT_DIR}"
)


print(
    "\nGenerated files include:"
    "\n- modernbert_embeddings_1994_2023.npy"
    "\n- modernbert_embedding_metadata_1994_2023.csv"
    "\n- all_test_anomaly_scores.csv"
    "\n- top_200_anomalies.csv"
    "\n- isolation_forest.joblib"
    "\n- experiment_config.json"
)
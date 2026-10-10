"""
This script generates Modern-BERT embeddings for textual data in a CSV file and saves the resulting dataset with embeddings to a new CSV file.

Workflow:
- Loads a CSV file ("fraud_text.csv") containing a column 'mda' with text data.
- Drops rows with missing values in the 'mda' column.
- Loads the Modern-BERT tokenizer and model from HuggingFace.
- Processes the text data in batches, tokenizes, and computes the [CLS] token embeddings for each entry.
- Concatenates the embeddings with the original dataframe.
- Saves the combined dataframe.
- Logs progress and errors to "mbert_embedding.log".

Requirements:
- Sufficient GPU/CPU memory for large batch processing
"""
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import logging

import torch._dynamo
torch._dynamo.config.suppress_errors = True

logging.basicConfig(
    filename="mbert_embedding.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Script started.")

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load the ModernBERT model and tokenizer
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    logging.info("Model and tokenizer loaded successfully.")

    # Load and preprocess the dataset
    df = pd.read_csv("path/to/filtered_dataframe.csv")
    df = df.dropna(subset=['mda'])
    logging.info(f"Dataset loaded. Shape: {df.shape}")

    def get_embeddings(text_list, batch_size=16):
        """
        Generates BERT embeddings for a list of input texts using batching.

        Args:
            text_list (List[str]): List of input text strings to generate embeddings for.
            batch_size (int, optional): Number of texts to process in each batch. Defaults to 16.

        Returns:
            np.ndarray: Array of embeddings with shape (num_texts, embedding_dim), where each row corresponds to the [CLS] token embedding of an input text.

        Notes:
            - Assumes that `tokenizer`, `model`, `device`, `torch`, and `np` are defined in the global scope.
            - Uses the [CLS] token embedding from the last hidden state as the representation for each text.
            - Processes texts in batches to optimize memory usage and performance.
            - Clears CUDA cache after each batch to manage GPU memory.
        """
        logging.info(f"Generating embeddings for {len(text_list)} texts.")
        all_embeddings = []

        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i + batch_size]
            logging.info(f"Processing batch {i // batch_size + 1}/{(len(text_list) - 1) // batch_size + 1}")

            inputs = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=8192,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            all_embeddings.append(cls_embeddings)

            torch.cuda.empty_cache()

        return np.vstack(all_embeddings)

    # Generate and attach embeddings
    mda_embeddings = get_embeddings(df['mda'].tolist())
    embedding_df = pd.DataFrame(
        mda_embeddings,
        index=df.index,
        columns=[f'mbert_full_cls_{i}' for i in range(mda_embeddings.shape[1])]
    )

    df_combined = pd.concat([df, embedding_df], axis=1)
    df_combined.to_csv("fraud_mbert_full.csv", index=False)
    logging.info("Saved full dataset with BERT embeddings to df_mbert_full.csv.")

except Exception as e:
    logging.error(f"Error encountered: {e}", exc_info=True)

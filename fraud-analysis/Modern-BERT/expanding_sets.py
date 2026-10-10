"""
This script performs a rolling-expansion training and evaluation of a Modern-BERT for fraud detection on a time-series text dataset.
It loads the dataset, computes class weights, and iteratively expands the training set by year, holding out a future year as the test set.
For each iteration, it splits the training data into training and validation sets, computes and prints class ratios, and calls the
`modern_bert_tuner` function to train and evaluate the model. The process is repeated for each year in the specified range, enabling
temporal generalization analysis of model performance.

Requirements:
- Sufficient GPU/CPU memory for large batch processing
"""
import pandas as pd
from MBERT_expansion import modern_bert_tuner
import gc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np
import torch

## Load the dataset
df = pd.read_csv("path/to/filtered_dataframe.csv", usecols=["mda", "fraudulent", "reporting_date"])
df['reporting_date'] = pd.to_datetime(df['reporting_date'])
df['year'] = df['reporting_date'].dt.year
df = df.rename(columns={'mda': 'text', 'fraudulent': 'label'})

all_labels = df["label"].values
global_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=all_labels)
global_weight_tensor = torch.tensor(global_weights, dtype=torch.float)

base_year = 2003
final_year = 2015
current_year = base_year

## Uncomment to use stationary validation set
#val_df = df[(df['year'] > 1996) & (df['year'] < base_year)][['text', 'label']].copy()

while current_year + 1 <= final_year:
    year_set = list(range(base_year, current_year + 1))
    test_year = current_year + 3
    year_set.append(test_year)
    year_set = sorted(set(year_set))

    train_years = [y for y in year_set if y < test_year]
    test_years = [test_year]

    ## Uncomment to use stationary validation set
    #train_df = df[df['year'].isin(train_years)][['text', 'label']].copy()

    full_train_df = df[df['year'].isin(train_years)][['text', 'label']].copy()
    train_df, val_df = train_test_split(full_train_df, test_size=0.2, stratify=full_train_df['label'], random_state=42)

    test_df = df[df['year'].isin(test_years)][['text', 'label']].copy()

    run_name = f"testyear_{test_year}"
    
    print(f"Training years: {train_years}, Test year: {test_year}")
    print(f"Train set shape: {train_df.shape}, Test set shape: {test_df.shape}, Validation set shape: {val_df.shape}")
    print(f"Universal fraudulent ratio: {df['label'].mean():.2f}")
    print(f"Training set fraudulent ratio: {train_df['label'].mean():.2f}")
    print(f"Validation set fraudulent ratio: {val_df['label'].mean():.2f}")
    print(f"Test set fraudulent ratio: {test_df['label'].mean():.2f}")

    modern_bert_tuner(train_df, val_df, test_df, global_weight_tensor, run_name=run_name)

    del train_df, test_df
    gc.collect()

    current_year += 1

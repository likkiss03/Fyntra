"""
This script preprocesses a dataset by performing the following steps:
- Loads the dataset from a CSV file.
- Encodes categorical columns using one-hot encoding.
- Converts relevant columns to datetime format, extracting year, month, and day features.
- Drops columns that are uninformative or exclusive to fraudulent cases.
- Removes additional text-based feature columns.
- Saves the cleaned and tabularized dataset to a new CSV file for further analysis.

Intended to create a dataset with only metadata extracted from 10-K filings and prepare it for training and testing machine learning models.
"""
import pandas as pd

# Load the dataset
df = pd.read_csv("path/to/fraud_dataframe.csv")

print(f"Initial shape: {df.shape}")
print(df.head())

cols_to_encode = ['city', 'state', 'sic', 'incorp_state', 'filing_type']
df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

# Convert date columns to datetime format
df['filing_date'] = pd.to_datetime(df['filing_date'], format='%Y-%m-%d', errors='coerce')
df['reporting_date'] = pd.to_datetime(df['reporting_date'], format='%Y-%m-%d', errors='coerce')
df['dateTime'] = df['dateTime'].str.split('T').str[0]
df['dateTime'] = pd.to_datetime(df['dateTime'], format='%Y-%m-%d', errors='coerce')
df['fraud_start'] = pd.to_datetime(df['fraud_start'], errors='coerce')
df['fraud_end'] = pd.to_datetime(df['fraud_end'], errors='coerce')
df['revoked'] = pd.to_datetime(df['revoked'], format='%m-%Y', errors='coerce')
df['fye'] = pd.to_datetime(df['fye'], format='%Y-%m-%d', errors='coerce')

# Columns that are uninformative or exclusive to fraudulent cases
cols =['cik', 'name', 'url', 'mda', 'dateTime', 
       'respondents', 'fraud_start', 'fraud_end', 'revoked',
       'certainty_start', 'certainty_end', '17a', '17a2', '17a3', '17b', '5a',
       '5b1', '5c', '10b', '13a', '12b20', '12b25', '13a1', '13a10', '13a11',
       '13a13', '13a14', '13a16', '13b2A', '13b2B', '13a15', '13b5', '14a',
       '14c', '19a', '30A', '100a2', '100b', '105c7B', 'corruption', 'amis',
       'fsf']

df.drop(cols, axis=1, inplace=True)

date_cols = df.select_dtypes(include=['datetime64']).columns

for col in date_cols:
    df[f'{col}_year'] = df[col].dt.year
    df[f'{col}_month'] = df[col].dt.month
    df[f'{col}_day'] = df[col].dt.day
    df.drop(columns=[col], inplace=True)

df.drop('fye_year', axis=1, inplace=True)

print(f"Shape after preprocessing: {df.shape}")
print(df.head())

df.drop(['word_count', 'char_count', 'word_density'], axis=1, inplace=True)

df.to_csv('/path/to/tabular_dataframe.csv', index=False)
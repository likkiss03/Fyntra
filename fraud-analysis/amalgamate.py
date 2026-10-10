"""
This script prepares and amalgamates multiple datasets related to financial filings and fraud labels,
cleans and preprocesses the data, and generates a final DataFrame for experimentation and analysis.
It merges firm-year records with fraud labels, applies text preprocessing to the MD&A sections,
engineers relevant features, and outputs both a comprehensive and a filtered dataset for downstream
fraud detection experiments.
"""
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
import re
from bs4 import BeautifulSoup
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR.parent

################## Loading the datasets ##################
# Labels Data
df_labels = pd.read_csv(DATA_DIR / 'aaer_mark5.csv', sep=';')

# Firm Years Data
mda_path = DATA_DIR / 'firm_years.json'

df_mda = pd.read_json(mda_path)

# Firm Years Data from Labels
mda_labels_path = DATA_DIR / 'firm_years_labels.json'

df_aaer = pd.read_json(mda_labels_path)
##########################################################

########### Preprocessing the labels dataframe ###########
cols_tofill = ['17a', '17a2', '17a3', '17b', '5a',
       '5b1', '5c', '10b', '13a', '12b20', '12b25', '13a1', '13a10', '13a11',
       '13a13', '13a14', '13a16', '13b2A', '13b2B', '13a15', '13b5', '14a',
       '14c', '19a', '30A', '100a2', '100b', '105c7B', 'corruption', 'amis',
       'fsf']

df_labels[cols_tofill] = df_labels[cols_tofill].fillna(0) # cells in the csv file are 0 are mostly empty so they are filled with 0 here

cols_tofill_ii = ['17a', '17a2', '17a3', '17b', '5a',
       '5b1', '5c', '10b', '13a', '12b20', '12b25', '13a1', '13a10', '13a11',
       '13a13', '13a14', '13a16', '13b2A', '13b2B', '13a15', '13b5', '14a',
       '14c', '19a', '30A', '100a2', '100b', '105c7B']

ids_to_replace = ["a4735537e28f75868c45803e870c53ca", 
                  "acd05174c49c0b083ad2270aa7a23944", 
                  "55ddfddf5767b4582c51582023bb6b85"]

df_labels.loc[df_labels["id"].isin(ids_to_replace), cols_tofill_ii] = df_labels.loc[df_labels["id"].isin(ids_to_replace), cols_tofill_ii].replace(0, np.nan)
##########################################################

########## Preprocessing the firm years data from labels and labeling ##########
df_aaer["late_filing"] = (df_aaer["filing_type"] == "10-K405").astype(int)
df_aaer["transition_filing"] = (df_aaer["filing_type"].str.startswith("10-KT")).astype(int)
df_aaer["amend_filing"] = (df_aaer["filing_type"].str.endswith("/A")).astype(int)

df_aaer["reporting_date"] = pd.to_datetime(df_aaer["reporting_date"]).dt.date

df_labels['fraud_start'] = pd.to_datetime(df_labels['fraud_start'], format='%m-%Y').dt.date
df_labels['fraud_end'] = (pd.to_datetime(df_labels['fraud_end'], format='%m-%Y') + MonthEnd(0)).dt.date

cols = ['dateTime', 'respondents', 'cik', 'fraud_start', 'fraud_end', 'revoked',
       'certainty_start', 'certainty_end', '17a', '17a2', '17a3', '17b', '5a',
       '5b1', '5c', '10b', '13a', '12b20', '12b25', '13a1', '13a10', '13a11',
       '13a13', '13a14', '13a16', '13b2A', '13b2B', '13a15', '13b5', '14a',
       '14c', '19a', '30A', '100a2', '100b', '105c7B', 'corruption', 'amis',
       'fsf']

df_m  = df_aaer.merge(df_labels[cols], on="cik", how="left")

# create fraud label on merged aaer dataframe
# Initialize with 0 (non-fraudulent by default)
df_m["fraudulent"] = 0

# Identify rows where fraud_start and fraud_end are NOT missing
valid_rows = df_m["fraud_start"].notna()

# Apply logic only to those rows
df_m.loc[valid_rows, "fraudulent"] = (
    (df_m.loc[valid_rows, "reporting_date"] >= df_m.loc[valid_rows, "fraud_start"]) &
    (df_m.loc[valid_rows, "reporting_date"] <= df_m.loc[valid_rows, "fraud_end"].apply(lambda d: d + pd.DateOffset(years=1)))
).astype(int)

cols_mda = ['cik', 'name', 'city', 'state', 'sic', 'incorp_state', 'filing_type',
       'fye', 'filing_date', 'reporting_date', 'url', 'mda']

# Drop duplicates only among non-fraudulent observations
nonfraud_duplicates = (
    df_m["fraudulent"] == 0
) & (
    df_m.duplicated(subset=cols_mda, keep=False)  # `keep=False` marks *all* duplicates
)

# Drop those rows
df_m_cleaned = df_m[~nonfraud_duplicates].copy()
################################################################################

############## merging firm years from labels with mda dataframe ##############
df_mda["late_filing"] = (df_mda["filing_type"] == "10-K405").astype(int)
df_mda["transition_filing"] = (df_mda["filing_type"].str.startswith("10-KT")).astype(int)
df_mda["amend_filing"] = (df_mda["filing_type"].str.endswith("/A")).astype(int)

df_mda["reporting_date"] = pd.to_datetime(df_mda["reporting_date"]).dt.date

df_mda['fraudulent'] = 0

# Step 1: Identify duplicates of df_mda relative to df_m_cleaned based on cols_mda
is_duplicate = df_mda[cols_mda].apply(tuple, axis=1).isin(
    df_m_cleaned[cols_mda].apply(tuple, axis=1)
)

# Step 2: Filter out those duplicates from df_mda
df_mda_unique = df_mda[~is_duplicate].copy()

# Step 3: Append unique df_mda rows to df_m_cleaned
df_final = pd.concat([df_m_cleaned, df_mda_unique], ignore_index=True)
###############################################################################

############### Preprocessing functions #################
def remove_html_tags(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return BeautifulSoup(text, 'html.parser').get_text() 

url_pattern = re.compile(r'https?://\S+')

def remove_urls(text: str) -> str:
    if not isinstance(text, str):
        return text
    return url_pattern.sub(r'', text)
#########################################################

################ Final preprocessing and saving the dataframes ##################
# Convert date columns to datetime format
df_final['filing_date'] = pd.to_datetime(df_final['filing_date'], format='%d-%m-%Y', errors='coerce')
df_final['reporting_date'] = pd.to_datetime(df_final['reporting_date'], format='%Y-%m-%d', errors='coerce')
df_final['dateTime'] = df_final['dateTime'].str.split('T').str[0]
df_final['dateTime'] = pd.to_datetime(df_final['dateTime'], format='%Y-%m-%d', errors='coerce')
df_final['fraud_start'] = pd.to_datetime(df_final['fraud_start'], errors='coerce')
df_final['fraud_end'] = pd.to_datetime(df_final['fraud_end'], errors='coerce')
df_final['revoked'] = pd.to_datetime(df_final['revoked'], format='%m-%Y', errors='coerce')
df_final['fye'] = pd.to_datetime(df_final['fye'], format='%d-%m')

# apply text preprocessing to the 'mda' column
df_final['mda'] = df_final['mda'].str.lower()
df_final['mda'] = df_final['mda'].apply(remove_html_tags)
df_final['mda'] = df_final['mda'].replace(r"[\n\t\\\'\"]+", ' ', regex=True)
df_final['mda'] = df_final['mda'].replace(r'[^\w\s]', ' ', regex=True)
df_final['mda'] = df_final['mda'].apply(remove_urls)
df_final['mda'] = df_final['mda'].replace(r'\s+', ' ', regex=True)

# Adding new columns for text analysis
df_final['char_count'] = df_final['mda'].apply(lambda x: len(x) if isinstance(x, str) else 0)
df_final['word_count'] = df_final['mda'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
df_final['word_density'] = df_final.apply(lambda x: len(x['mda']) / len(x['mda'].split()) if isinstance(x['mda'], str) and len(x['mda'].split()) > 0 else 0, axis=1)

# Fixing SIC codes
df_final.loc[df_final['sic'] == "1044", 'sic'] = "1040"

df_final.loc[36239, 'sic'] = "1311"  # ENRON OIL & GAS CO
df_final.loc[36243, 'sic'] = "1311"  # ENRON OIL & GAS CO
df_final.loc[38001, 'sic'] = "2911"  # BP PRUDHOE BAY ROYALTY TRUST
df_final.loc[46302, 'sic'] = "8071"  # NATIONAL HEALTH LABORATORIES HOLDINGS INC

# Filter out rows with 'amend_filing' == 1 and 'word_count' <= 200
df_filtered = df_final[(df_final['word_count'] > 200) & (df_final['amend_filing'] == 0)]

# Save the final DataFrame and filtered DataFrame to CSV files
df_final.to_csv(PROJECT_DIR / 'fraud.csv', index=False)
df_filtered.to_csv(PROJECT_DIR / 'fraud_text.csv', index=False)
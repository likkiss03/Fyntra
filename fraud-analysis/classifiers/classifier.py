"""
This script performs fraud detection using Random Forest and XGBoost classifiers on different data. The type of data used is determined by the commented-out sections 
at the top, which can be adjusted to load different datasets (e.g., tabular data, sentence embeddings, or Modern-BERT embeddings).
It loads and preprocesses the data, splits it into training, validation, and test sets based on reporting years, and handles class imbalance using sample weights.
The script trains both classifiers, evaluates their performance using classification metrics and ROC AUC, and visualizes the ROC curves for comparison.

Key steps:
- Data loading and preprocessing (feature selection, date handling)
- Train/validation/test split based on reporting year (with a detection delay)
- Handling class imbalance with sample weights
- Training and evaluating Random Forest and XGBoost classifiers
- Custom F1-macro evaluation for XGBoost with early stopping
- Outputting classification reports, confusion matrices, ROC AUC scores, and ROC curve plots
"""
import pandas as pd
import numpy as np
import xgboost as xgb 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score
)

## Loading Data
df= pd.read_csv('path/to/dataframe_with_desired_inputs.csv', low_memory=False)
## if using tabular data, uncomment the next line
# df = df.drop(columns=[col for col in df.columns if col.startswith('filing_type_')])
# df = df[['reporting_date', 'word_count', 'fraudulent']]
# df = df[[col for col in df.columns if col.startswith('sen_emb_')] + ['reporting_date', 'fraudulent']]
df = df[[col for col in df.columns if col.startswith('mbert_full_cls')] + ['reporting_date', 'fraudulent']]
df['reporting_date'] = pd.to_datetime(df['reporting_date'])
df['reporting_date_year'] = df['reporting_date'].dt.year
print("Data Loaded with shape:", df.shape)

# testing year 2011 because fraud ratio most resembles the overal fraud ratio in tabular data of 0.029
test_df = df[df['reporting_date_year'] == 2011]
# X_test = test_df.drop('fraudulent', axis=1)
# X_test = test_df[['word_count']]
# X_test = test_df[[col for col in test_df.columns if col.startswith('sen_emb_')]]
X_test = test_df[[col for col in test_df.columns if col.startswith('mbert_full_cls')]]
y_test = test_df['fraudulent']

# validation and training set sampled from all the years up until 2008 (because 3 year detection delay)
train_val_df = df[df['reporting_date_year'] <= 2008]

# X = train_val_df.drop('fraudulent', axis=1)
# X = train_val_df[['word_count']]
# X = train_val_df[[col for col in train_val_df.columns if col.startswith('sen_emb_')]]
X = train_val_df[[col for col in train_val_df.columns if col.startswith('mbert_full_cls')]]
y = train_val_df['fraudulent']

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
    )
print("Sets generated")
print(f"X_train shape: {X_train.shape}, y_train fraudulent counts: {np.sum(y_train)}")
print(f"X_val shape: {X_val.shape}, y_val fraudulent counts: {np.sum(y_val)}")
print(f"X_test shape: {X_test.shape}, y_test fraudulent counts: {np.sum(y_test)}")

sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# fit RandomForestClassifier
rf_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    criterion='log_loss',
)

rf_clf.fit(X_train, y_train, sample_weight=sample_weights)
print("Random Forest Classifier fitted.")

print("Evaluating Random Forest Classifier...")
y_pred_rf = rf_clf.predict(X_test)
y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]

print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf, digits=2, zero_division=0))
print("Confusion Matrix for Random Forest:")
print(confusion_matrix(y_test, y_pred_rf))

roc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"ROC AUC for Random Forest: {roc_rf:.4f}")

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={roc_rf:.2f})")


# fit XGBoostClassifier
D_train = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
D_valid = xgb.DMatrix(X_val, label=y_val)
D_test = xgb.DMatrix(X_test)


def f1_macro_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    f1 = f1_score(y_true, np.round(y_pred), average='macro')
    return 'f1_macro', f1

early_stopping_rounds = 10

early_stop = xgb.callback.EarlyStopping(rounds=early_stopping_rounds,
                                        metric_name='f1_macro',
                                        data_name='Valid',
                                        maximize=True,
                                        save_best=True)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.1,
    'max_depth': 6,
    'seed': 42,
    'nthread': -1,
    # 'tree_method': 'hist',  # Use 'hist' for faster training on large datasets
    # 'subsample': 0.8,  # Subsample ratio of the training instances
    # 'sampling_method': 'gradient_based', 
}

evals = [(D_train, 'Train'), (D_valid, 'Valid')]

booster = xgb.train(
    params, D_train,
    evals=evals,
    custom_metric=f1_macro_eval,
    num_boost_round=1000,
    callbacks=[early_stop],
    verbose_eval=True)
print("xgb.train model fitted.")

print(f"Best iteration: {booster.best_iteration}")
y_prob_xgb = booster.predict(D_test, iteration_range=(0, booster.best_iteration + 1))
y_pred_xgb = np.round(y_prob_xgb)

print("Classification Report for XGBoost:")
print(classification_report(y_test, y_pred_xgb, digits=2, zero_division=0))
print("Confusion Matrix for XGBoost:")
print(confusion_matrix(y_test, y_pred_xgb))

roc_xgb = roc_auc_score(y_test, y_prob_xgb)
print(f"ROC AUC for XGBoost: {roc_xgb:.4f}")

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={roc_xgb:.2f})")

# Plotting ROC curves
plt.plot([0,1],[0,1],'k--',alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
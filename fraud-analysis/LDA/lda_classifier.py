"""
lda_classifier.py
This script performs fraud detection on textual data using topic modeling and machine learning classifiers.
It reads a dataset of financial reports, applies Latent Dirichlet Allocation (LDA) to extract topic distributions
from the text, and then uses these topic vectors as features to train and evaluate Random Forest and XGBoost classifiers.
The script supports flexible train/validation/test splits based on reporting years, handles class imbalance with sample weights,
and provides detailed evaluation metrics including classification reports, confusion matrices, and ROC curves.
Main steps:
- Load and preprocess the dataset, including tokenization and filtering.
- Train an LDA model to extract topic distributions from the text.
- Use topic distributions as features for classification.
- Train and evaluate Random Forest and XGBoost classifiers.
- Output performance metrics and save ROC curve plots for comparison.
Intended for research and analysis of fraud detection using interpretable topic-based features.
"""
import json
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from pathlib import Path

def tokenize_texts(text_col):
    """
    Tokenizes a pandas Series of text data using Gensim's simple_preprocess.

    Args:
        text_col (pandas.Series): A pandas Series containing text data.

    Returns:
        list: A list of tokenized texts, where each text is represented as a list of tokens (words).
    """
    for text in text_col.dropna():
        yield simple_preprocess(text)

def get_topic_vectors(lda_model, corpus):
    """
    Generates topic distribution vectors for each document in a corpus using a trained LDA model.

    Args:
        lda_model: A trained LDA model with a `get_document_topics` method.
        corpus: An iterable of documents in the format expected by the LDA model.

    Returns:
        np.ndarray: A 2D NumPy array where each row corresponds to a document's topic distribution vector.
                    Each vector contains the probabilities for all topics, sorted by topic index.
    """
    topic_vecs = []
    for doc in corpus:
        topics = lda_model.get_document_topics(doc, minimum_probability=0.0)
        vec = [prob for _, prob in sorted(topics, key=lambda x: x[0])]
        topic_vecs.append(vec)
    return np.array(topic_vecs)


def save_model_results(results_dir, metrics_filename, matrix_filename, model_name, y_true, y_pred, y_prob, best_iteration=None):
    results_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    results = {
        'model': model_name,
        'test_year': 2011,
        'test_samples': int(len(y_true)),
        'class_distribution': {
            'non_fraud': int(np.sum(np.asarray(y_true) == 0)),
            'fraud': int(np.sum(np.asarray(y_true) == 1)),
        },
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'classification_report': classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=['non_fraud', 'fraud'],
            output_dict=True,
            zero_division=0,
        ),
        'confusion_matrix': {
            'labels': [0, 1],
            'rows': ['actual_non_fraud', 'actual_fraud'],
            'columns': ['predicted_non_fraud', 'predicted_fraud'],
            'values': matrix.tolist(),
        },
    }
    if best_iteration is not None:
        results['best_iteration'] = int(best_iteration)

    with (results_dir / metrics_filename).open('w', encoding='utf-8') as results_file:
        json.dump(results, results_file, indent=2)

    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=['Non-fraud', 'Fraud'],
    ).plot(ax=axis, cmap='Blues', values_format='d', colorbar=False)
    axis.set_title(model_name)
    figure.tight_layout()
    figure.savefig(results_dir / matrix_filename, dpi=200, bbox_inches='tight')
    plt.close(figure)

if __name__ == "__main__":
    print("Training RF with LDA script started.")

    data_path = Path(__file__).resolve().parents[1] / 'fraud_text.csv'
    results_dir = Path(__file__).resolve().parents[1] / 'results' / 'B0_supervised'
    df = pd.read_csv(data_path)
    df['reporting_date'] = pd.to_datetime(df['reporting_date'])

    ##########################################################################
    ######## Uncomment lines below to change the train/val/test split ########
    ##########################################################################
    
    # training set is from 2003 with test and validation sets from later years
    #df_train = df[(df['reporting_date'].dt.year == 2003) & (df['word_count'] > df['word_count'].quantile(0.25))]

    #pos = df[(df['reporting_date'].dt.year>2003) & (df['fraudulent']==1)]
    #neg = df[(df['reporting_date'].dt.year>2003) & (df['fraudulent']==0)]

    #pos_sample = pos.sample(int((2*len(df_train))*df['fraudulent'].mean()+1)*2, random_state=42)
    #neg_sample = neg.sample(((2*len(df_train)) - int((2*len(df_train))*df['fraudulent'].mean()+1)) * 2, random_state=42)
    #bowl = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=42)

    #df_val, df_test = train_test_split(bowl, test_size=0.5, stratify=bowl['fraudulent'], random_state=42)

    # train 2003 and test 2011 and val sampled from post 2003 to 2008
    #bowl = df[(df['reporting_date'].dt.year>2003) & (df['reporting_date'].dt.year<=2008)]
    #_, df_val = train_test_split(bowl, test_size=0.2, stratify=bowl['fraudulent'], random_state=42)
    df_test = df[df['reporting_date'].dt.year == 2011]

    # train 2008, test 2011 and val sampled from 2003 to 2007
    #df_train = df[(df['reporting_date'].dt.year == 2008) & (df['word_count'] > df['word_count'].quantile(0.25))]
    #bowl = df[(df['reporting_date'].dt.year>2003) & (df['reporting_date'].dt.year<=2007)]
    #_, df_val = train_test_split(bowl, test_size=0.2, stratify=bowl['fraudulent'], random_state=42)

    # train and val data between 2004-2008 and test 2011
    bowl = df[(df['reporting_date'].dt.year>2003) & (df['reporting_date'].dt.year<=2008)]
    df_train, df_val = train_test_split(bowl, test_size=0.2, stratify=bowl['fraudulent'], random_state=42)
    df_train = df_train[df_train['word_count'] > df_train['word_count'].quantile(0.25)]
    ##########################################################################

    print(f"Training set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")

    print(f"Universal fraudulent ratio: {df['fraudulent'].mean():.2f}")
    print(f"Training set fraudulent ratio: {df_train['fraudulent'].mean():.2f}")
    print(f"Validation set fraudulent ratio: {df_val['fraudulent'].mean():.2f}")
    print(f"Test set fraudulent ratio: {df_test['fraudulent'].mean():.2f}")

    y_train = df_train['fraudulent'].values
    y_val = df_val['fraudulent'].values
    y_test = df_test['fraudulent'].values
    del df, bowl
    gc.collect()

    dictionary = Dictionary(tokenize_texts(df_train['mda']))
    dictionary.filter_extremes(no_below=2, no_above=0.25)

    corpus_train = [
        dictionary.doc2bow(tokens)
        for tokens in tokenize_texts(df_train['mda'])
    ]
    df_train.drop(columns=['mda'], inplace=True)
    gc.collect()

    corpus_val = [
        dictionary.doc2bow(tokens)
        for tokens in tokenize_texts(df_val['mda'])
    ]
    df_val.drop(columns=['mda'], inplace=True)
    gc.collect()

    corpus_test = [
        dictionary.doc2bow(tokens)
        for tokens in tokenize_texts(df_test['mda'])
    ]
    df_test.drop(columns=['mda'], inplace=True)
    del df_train, df_val, df_test
    gc.collect()
    
    num_topics = 100

    # Train LDA model
    lda = LdaModel(
        corpus=corpus_train,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto',
        eta='auto'
    )

    X_train = get_topic_vectors(lda, corpus_train)
    X_val = get_topic_vectors(lda, corpus_val)
    X_test = get_topic_vectors(lda, corpus_test)

    print(f"Training set topic vectors shape: {X_train.shape}")
    print(f"Test set topic vectors shape: {X_test.shape}")

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # Fit RandomForestClassifier
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        criterion='log_loss',  # Use log_loss for probabilistic output
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
    save_model_results(
        results_dir,
        'lda_rf_metrics.json',
        'lda_rf_confusion_matrix.png',
        'LDA + Random Forest',
        y_test,
        y_pred_rf,
        y_prob_rf,
    )

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
    save_model_results(
        results_dir,
        'lda_xgb_metrics.json',
        'xgboost_confusion_matrix.png',
        'LDA + XGBoost',
        y_test,
        y_pred_xgb,
        y_prob_xgb,
        best_iteration=booster.best_iteration,
    )

    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={roc_xgb:.2f})")

    # Plot ROC curves
    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure_name.png")

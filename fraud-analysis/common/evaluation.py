from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)


def precision_recall_at_k(y_true, scores, k):
    """
    Evaluate how many known fraud cases occur among the top-k
    highest anomaly scores.

    IMPORTANT:
    y_true is used only for evaluation.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)

    k = min(k, len(y_true))

    ranking = np.argsort(scores)[::-1][:k]
    top_labels = y_true[ranking]

    fraud_found = int(top_labels.sum())
    total_fraud = int(y_true.sum())

    precision_k = fraud_found / k
    recall_k = fraud_found / total_fraud if total_fraud else 0.0

    return precision_k, recall_k, fraud_found


def evaluate_anomaly_scores(
    y_true,
    scores,
    output_dir,
    experiment_name,
):
    """
    Evaluate unsupervised anomaly scores against hidden AAER fraud labels.

    Labels are NOT involved in model fitting.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "Evaluation requires both fraud and non-fraud samples."
        )

    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    results = {
        "experiment": experiment_name,
        "n_samples": len(y_true),
        "n_fraud": int(y_true.sum()),
        "fraud_ratio": float(y_true.mean()),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }

    print("\n" + "=" * 70)
    print(experiment_name)
    print("=" * 70)

    print(f"Samples:       {len(y_true)}")
    print(f"Fraud cases:   {int(y_true.sum())}")
    print(f"Fraud ratio:   {y_true.mean():.4f}")
    print(f"ROC-AUC:       {roc_auc:.4f}")
    print(f"PR-AUC:        {pr_auc:.4f}")

    print("\nTop-K anomaly evaluation")
    print("-" * 70)

    for k in [50, 100, 200]:
        if k <= len(y_true):

            precision_k, recall_k, fraud_found = (
                precision_recall_at_k(
                    y_true,
                    scores,
                    k,
                )
            )

            results[f"precision_at_{k}"] = precision_k
            results[f"recall_at_{k}"] = recall_k
            results[f"fraud_found_at_{k}"] = fraud_found

            print(
                f"Top {k:<3} | "
                f"Fraud found: {fraud_found:<3} | "
                f"Precision@{k}: {precision_k:.4f} | "
                f"Recall@{k}: {recall_k:.4f}"
            )

    # Save metrics
    pd.DataFrame([results]).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, scores)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(experiment_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "roc_curve.png",
        dpi=200,
    )
    plt.close()

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(
        y_true,
        scores,
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        recall,
        precision,
        label=f"PR-AUC = {pr_auc:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(experiment_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "precision_recall_curve.png",
        dpi=200,
    )
    plt.close()

    return results
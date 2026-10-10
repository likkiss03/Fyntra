"""
This script fine-tunes the ModernBERT-base transformer model for binary sequence classification tasks, 
specifically tailored for fraud detection. It includes custom tokenization, weighted loss for class imbalance, 
and evaluation metrics such as macro F1-score and ROC-AUC. The script uses HuggingFace Transformers and Datasets 
libraries, supports bfloat16 training, and provides detailed evaluation reports and ROC curve visualization 
after training. Model parameters outside the classifier head are frozen to focus training on the classification layer.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from huggingface_hub import HfFolder
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch._dynamo

torch._dynamo.config.suppress_errors = True

# Load the tokenizer and model
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set the input sequence length
input_seq_length = 4096

def tokenize(batch):
    """
    Tokenizes a batch of text data using the specified tokenizer.
    Args:
        batch (dict): A dictionary containing a 'text' key with a list of text strings to tokenize.
    Returns:
        dict: A dictionary of tokenized outputs with padding and truncation applied, formatted as PyTorch tensors.
    """
    return tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt", max_length=input_seq_length)

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for model predictions.

    This function calculates the macro-averaged F1 score between the predicted and true labels.
    It is intended for use with evaluation pipelines where predictions and labels are provided
    as a tuple.
    If the commented lines are uncommented, it will compute the binary F1 score instead.
    Args:
        eval_pred (Tuple[np.ndarray, np.ndarray]): A tuple containing the model predictions
            (as logits or probabilities) and the true labels.
    Returns:
        dict: A dictionary containing the macro-averaged F1 score with the key 'f1_macro'.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # score = f1_score(labels, predictions, labels=labels, pos_label=1, average="binary")
    score = f1_score(labels, predictions, average="macro")

    # return {"f1": float(score) if score == 1 else score}
    return {"f1_macro": float(score)}

class WeightedTrainer(Trainer):
    """
    Custom Trainer class that supports weighted loss for handling class imbalance.
    Args:
        *args: Variable length argument list for the base Trainer class.
        class_weights (torch.Tensor, optional): A tensor of weights for each class, used to weight the loss function.
        **kwargs: Arbitrary keyword arguments for the base Trainer class.
    Methods:
        compute_loss(model, inputs, return_outputs=False, **kwargs):
            Computes the weighted cross-entropy loss using the provided class weights.
            Args:
                model: The model to compute loss for.
                inputs (dict): Dictionary containing input tensors, including 'labels'.
                return_outputs (bool, optional): If True, returns a tuple of (loss, outputs).
                **kwargs: Additional keyword arguments.
            Returns:
                torch.Tensor or (torch.Tensor, Any): The computed loss, and optionally the model outputs.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weight_tensor = self.class_weights.to(logits.device)

        loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

def modern_bert_tuner(train_df, val_df, test_df, class_weights, run_name=""):
    """
    Fine-tunes a BERT-based sequence classification model for binary classification using weighted loss and evaluates its performance.

    This function performs the following steps:
    1. Prepares and tokenizes the input datasets (train, validation, test).
    2. Loads a pre-trained BERT model for sequence classification, freezing all layers except the classifier or not if lines commented out.
    3. Configures training arguments, including logging, evaluation, and saving strategies.
    4. Initializes a custom Trainer with class weights for handling class imbalance.
    5. Trains the model and evaluates it on the test set.
    6. Prints classification metrics and saves the ROC curve plot.

    Args:
        train_df (pandas.DataFrame): Training dataset containing 'text' and 'label' columns.
        val_df (pandas.DataFrame): Validation dataset containing 'text' and 'label' columns.
        test_df (pandas.DataFrame): Test dataset containing 'text' and 'label' columns.
        class_weights (torch.Tensor or np.ndarray): Class weights for handling class imbalance during training.
        run_name (str, optional): Name for the current run, used in output directories and logging. Defaults to "".

    Returns:
        None

    Side Effects:
        - Prints progress and evaluation metrics to stdout.
        - Saves the ROC curve plot as a PNG file in the 'expansion-rocs' directory.
        - Saves model checkpoints and logs in specified directories.
    """
    print(f"ModernBERT Tuner script started. Run name: {run_name}")

    split_dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df),
        "validation": Dataset.from_pandas(val_df)
    })

    if "label" in split_dataset["train"].features.keys():
        split_dataset = split_dataset.rename_column("label", "labels")  # to match Trainer

    tokenized_dataset = split_dataset.map(tokenize, batched=True, remove_columns=["text"])
    print("Tokenization complete.")

    num_labels = 2
    label2id = {"0": 0, "1": 1}
    id2label = {0: "0", 1: "1"}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=num_labels, 
        label2id=label2id, 
        id2label=id2label,
    )

    # freeze layers
    for name, params in model.named_parameters():
        if not name.startswith("classifier"):
            params.requires_grad = False
    print("Model loaded.")

    training_args = TrainingArguments(
        output_dir= f"expansion-Models/ModernBERT-fraud-expansion-{run_name}",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        num_train_epochs=15,
        bf16=True, # bfloat16 training 
        optim="adamw_torch_fused", # improved optimizer 
        # logging & evaluation strategies
        logging_strategy="steps",
        logging_steps=100,
        logging_dir=f"./logs/{run_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        # push to hub parameters
        push_to_hub=False,
        hub_strategy="every_save",
        hub_token=HfFolder.get_token(),
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )
    print("Trainer initialized.")
    print("Starting training...")
    trainer.train()
    print("Training complete.")
    print("Evaluating model...")
    preds_out = trainer.predict(tokenized_dataset["test"])

    logits = preds_out.predictions
    preds = np.argmax(logits, axis=1)

    true_labels = preds_out.label_ids

    print("Classification report:")
    print(classification_report(true_labels, preds, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(true_labels, preds))

    prob_pos = logits[:, 1]
    auc = roc_auc_score(true_labels, prob_pos)
    print(f"AUC: {auc:.4f}")

    fpr, tpr, _ = roc_curve(true_labels, prob_pos)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - BERT Fraud Detection")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"expansion-rocs/bert_fraud_roc_{run_name}.png", dpi=300)

# ==============================================================================
# src/evaluate.py — Unified evaluation script for saved models / raw predictions
# Usage (evaluate PhoBERT checkpoint on test set):
#   python src/evaluate.py --model phobert --checkpoint ./phobert_vsfc_results/checkpoint-best --seed 42
# Usage (re-evaluate baseline predictions from a CSV):
#   python src/evaluate.py --model baseline --preds_csv results/seed_42/baseline_preds.csv
# ==============================================================================
import argparse
import os
import sys
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import ID2LABEL, set_all_seeds, pretty_print_results


def evaluate_phobert(checkpoint: str, seed: int = 42):
    """Load PhoBERT checkpoint and evaluate on the official UIT-VSFC test set."""
    set_all_seeds(seed)
    import torch
    from datasets import load_dataset
    from pyvi import ViTokenizer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f">>> Loading checkpoint: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    dataset = load_dataset("uitnlp/vietnamese_students_feedback")
    test_sentences = [ViTokenizer.tokenize(s) for s in dataset["test"]["sentence"]]
    test_labels = dataset["test"]["sentiment"]

    all_preds = []
    BATCH = 32
    for i in range(0, len(test_sentences), BATCH):
        batch = test_sentences[i : i + BATCH]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    return np.array(test_labels), np.array(all_preds)


def evaluate_from_csv(preds_csv: str):
    """Read predictions from a CSV with columns: label, pred."""
    import csv
    labels, preds = [], []
    with open(preds_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
            preds.append(int(row["pred"]))
    return np.array(labels), np.array(preds)


def print_full_report(y_true, y_pred, model_name: str = "Model"):
    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL)]
    results = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Macro-F1":  f1_score(y_true, y_pred, average="macro"),
    }
    pretty_print_results(results, model_name=model_name)
    print("Per-Class Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    print("Confusion Matrix (rows=True, cols=Pred):")
    print(confusion_matrix(y_true, y_pred))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on UIT-VSFC test set.")
    parser.add_argument("--model",       choices=["phobert", "baseline"], required=True)
    parser.add_argument("--checkpoint",  default=None, help="(PhoBERT) path to saved checkpoint dir.")
    parser.add_argument("--preds_csv",   default=None, help="(Baseline) path to predictions CSV.")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    if args.model == "phobert":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for model=phobert")
        y_true, y_pred = evaluate_phobert(args.checkpoint, seed=args.seed)
    else:
        if not args.preds_csv:
            raise ValueError("--preds_csv is required for model=baseline")
        y_true, y_pred = evaluate_from_csv(args.preds_csv)

    print_full_report(y_true, y_pred, model_name=f"{args.model} (seed={args.seed})")


if __name__ == "__main__":
    main()

# ==============================================================================
# src/utils.py — Shared utilities for PhoBERT vs TF-IDF benchmark (UIT-VSFC)
# ==============================================================================
import os
import random
import numpy as np

LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}


def set_all_seeds(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility across NumPy, Python, and (if
    available) PyTorch.  Call this at the very top of any training script."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def compute_metrics_np(y_true, y_pred):
    """Return dict with accuracy and Macro-F1 computed from plain lists /
    NumPy arrays (no sklearn required for the basic case)."""
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def pretty_print_results(results: dict, model_name: str = "Model") -> None:
    """Print a formatted result table to stdout."""
    print(f"\n{'=' * 55}")
    print(f"  Results — {model_name}")
    print(f"{'=' * 55}")
    for k, v in results.items():
        print(f"  {k:<20}: {v:.4f}" if isinstance(v, float) else f"  {k:<20}: {v}")
    print(f"{'=' * 55}\n")

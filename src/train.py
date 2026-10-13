# ==============================================================================
# src/train.py — Main training entry-point (dispatches baseline or PhoBERT)
# Usage:
#   python src/train.py --model baseline --seed 42
#   python src/train.py --model phobert  --seed 42 --epochs 3 --lr 2e-5
# ==============================================================================
import argparse
import subprocess
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Baseline (TF-IDF + LR) or PhoBERT on UIT-VSFC."
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "phobert"],
        required=True,
        help="Which model to train.",
    )
    parser.add_argument("--seed",       type=int,   default=42,    help="Random seed.")
    parser.add_argument("--epochs",     type=int,   default=3,     help="(PhoBERT only) Number of epochs.")
    parser.add_argument("--lr",         type=float, default=2e-5,  help="(PhoBERT only) Peak learning rate.")
    parser.add_argument("--batch_size", type=int,   default=16,    help="(PhoBERT only) Train batch size.")
    return parser.parse_args()


def main():
    args = parse_args()
    src_dir = os.path.dirname(os.path.abspath(__file__))

    if args.model == "baseline":
        script = os.path.join(src_dir, "train_baseline.py")
        cmd = [sys.executable, script, "--seed", str(args.seed)]
    else:
        script = os.path.join(src_dir, "train_phobert.py")
        cmd = [
            sys.executable, script,
            "--seed",       str(args.seed),
            "--epochs",     str(args.epochs),
            "--lr",         str(args.lr),
            "--batch_size", str(args.batch_size),
        ]

    print(f">>> Launching: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

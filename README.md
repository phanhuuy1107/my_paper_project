# Vietnamese Student Feedback Sentiment Analysis (PhoBERT vs. TF-IDF)

Official research repository for the paper:  
**"An Empirical Comparison between Contextual Language Model PhoBERT and Classical TF-IDF for Vietnamese Student Feedback Sentiment Analysis"**

---

## 📌 Overview
This repository provides full source code, configuration files, and evaluation scripts to replicate the experimental benchmark on the **UIT-VSFC** corpus (16,175 sentences).

### Key Results
| Model | Accuracy | Macro-F1 | Δ Macro-F1 |
|-------|----------|----------|------------|
| Baseline (TF-IDF + LR, deterministic) | 88.82% | 65.06% | Reference |
| PhoBERT-base (avg. seeds 42, 100, 2024) | **94.08 ± 0.18%** | **82.95 ± 0.69%** | **+17.89 pp** |

- **Neutral class F1:** 13.04% → **59.64%** (+46.60 pp)
- **Ablation** (w/o `pyvi` segmentation): −5.46 pp Macro-F1
- **Error reduction:** 354 → 183 errors (−48.3%)

---

## 🗂️ Repository Structure

```
my_paper_project/
├── src/                         # Core source code
│   ├── train.py                 # Main entry-point dispatcher (baseline or phobert)
│   ├── train_baseline.py        # TF-IDF + Logistic Regression training
│   ├── train_phobert.py         # PhoBERT-base fine-tuning
│   ├── data_pipeline.py         # Data loading, preprocessing & EDA
│   ├── evaluate.py              # Evaluation on saved checkpoints / prediction CSVs
│   ├── run_experiments_mentor.py# Multi-seed statistics & 5-category error analysis
│   └── utils.py                 # Shared utilities (seed, metrics, label mapping)
│
├── paper/                       # LaTeX paper source
│   ├── main.tex                 # IEEE conference paper (full English)
│   ├── references.bib           # 12 BibTeX entries with DOIs
│   └── figures/
│       └── confusion_matrix.png # Publication-ready confusion matrix (300 DPI)
│
├── data/                        # Dataset directory (not tracked by git)
│   ├── raw/                     # Raw UIT-VSFC JSON files (download separately)
│   └── processed/               # Preprocessed / tokenized cache files
│
├── notebooks/                   # Exploratory Jupyter notebooks
│   ├── 01_explore.ipynb         # Dataset EDA
│   ├── 02_baseline.ipynb        # TF-IDF baseline walkthrough
│   ├── 03_phobert.ipynb         # PhoBERT fine-tuning walkthrough
│   └── 04_analysis.ipynb        # Error analysis & confusion matrix
│
├── results/                     # Evaluation outputs per seed
│   ├── seed_42/                 # Checkpoint metrics & prediction logs (Seed 42)
│   ├── seed_2024/               # Checkpoint metrics & prediction logs (Seed 2024)
│   └── results.csv              # Aggregated multi-seed performance table
│
├── docs/                        # Research documentation
│   ├── reproducibility_checklist_E3.md   # Appendix E.3 reproducibility checklist
│   └── Change_Log_Nop_Lai_Chuong_8.md    # Change log: old vs. new numbers
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Environment Setup

```bash
git clone https://github.com/phanhuuy1107/my_paper_project.git
cd my_paper_project
pip install -r requirements.txt
```

**Python:** 3.9+  |  **GPU recommended:** NVIDIA Tesla T4 or equivalent

---

## 📂 Dataset (UIT-VSFC)
The dataset is automatically downloaded from 🤗 HuggingFace Hub when you run any training script:
```python
from datasets import load_dataset
dataset = load_dataset("uitnlp/vietnamese_students_feedback")
```

Official 7:1:2 partition:
| Split | Sentences | Ratio |
|-------|-----------|-------|
| Train | 11,426 | 70.6% |
| Validation | 1,583 | 9.8% |
| Test | 3,166 | 19.6% |

Class distribution: 43.7% Negative · 4.8% Neutral · 51.5% Positive

---

##  Reproduction Commands

### 1. Run Baseline (TF-IDF + Logistic Regression)
```bash
python src/train.py --model baseline --seed 42
# Or directly:
python src/train_baseline.py
```

### 2. Fine-Tune PhoBERT-base (Multi-Seed Benchmark)
```bash
# Seed 42
python src/train.py --model phobert --seed 42 --epochs 3 --lr 2e-5

# Seed 100
python src/train.py --model phobert --seed 100 --epochs 3 --lr 2e-5

# Seed 2024
python src/train.py --model phobert --seed 2024 --epochs 3 --lr 2e-5
```

### 3. Ablation Study (Without Word Segmentation)
```bash
python src/train_phobert.py --seed 42 --no_pyvi --epochs 3
```

### 4. Evaluate a Saved Checkpoint
```bash
python src/evaluate.py --model phobert --checkpoint ./phobert_vsfc_results/checkpoint-best --seed 42
```

### 5. Run Multi-Seed Statistics & Error Analysis
```bash
python src/run_experiments_mentor.py
```

---

## Citation
If you find this work useful, please cite:
```bibtex
@inproceedings{vsfc2018,
  author    = {Nguyen, Quan Thi and Nguyen, Thin Luong and Luong, Ngan Huu and Nguyen, Kiet Van},
  title     = {{UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis}},
  booktitle = {Proceedings of the 10th International Conference on Knowledge and Systems Engineering (KSE)},
  pages     = {19--24},
  year      = {2018},
  publisher = {IEEE},
  doi       = {10.1109/KSE.2018.8573337}
}

@inproceedings{phobert2020,
  author    = {Nguyen, Dat Quoc and Nguyen, Anh Tuan},
  title     = {{PhoBERT: Pre-trained language models for Vietnamese}},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  pages     = {1037--1042},
  year      = {2020},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2020.findings-emnlp.92}
}
```

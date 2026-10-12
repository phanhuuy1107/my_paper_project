# Vietnamese Student Feedback Sentiment Analysis (PhoBERT vs. TF-IDF)

Official research repository for the paper:  
**"An Empirical Comparison between Contextual Language Model PhoBERT and Classical TF-IDF for Vietnamese Student Feedback Sentiment Analysis"**

---

## 📌 Overview
This repository provides full source code, configuration files, and evaluation scripts to replicate the experimental benchmark on the **UIT-VSFC** corpus ($16,175$ sentences).

### Key Results
* **Baseline (TF-IDF + Logistic Regression):** $88.82\%$ Accuracy, $65.06\%$ Macro-F1 (deterministic).
* **PhoBERT-base (Fine-tuned across 3 seeds: 42, 100, 2024):** $\mathbf{94.08 \pm 0.18\%}$ Accuracy, $\mathbf{82.95 \pm 0.69\%}$ Macro-F1 ($+17.89\text{ pp}$ leap).
* **Neutral Minority Class F1:** Improved from $13.04\%$ to $\mathbf{59.64\%}$ ($+46.60\text{ pp}$).
* **Ablation Study (w/o `pyvi` segmentation):** $-5.46\text{ pp}$ drop in Macro-F1.
* **Error Reduction:** Baseline $354$ errors $\to$ PhoBERT $183$ errors ($\mathbf{-48.3\%}$ error reduction).

---

## 🛠️ Environment Setup

```bash
git clone https://github.com/phanhuuy1107/my_paper_project.git
cd my_paper_project
pip install -r requirements.txt
```

---

## 📂 Dataset (UIT-VSFC)
We use the official 7:1:2 partition from the KSE 2018 paper:
* **Train:** $11,426$ sentences ($70.6\%$)
* **Validation:** $1,583$ sentences ($9.8\%$)
* **Test:** $3,166$ sentences ($19.6\%$)

Place the data files in `./data/`:
```
data/
  train.json
  dev.json
  test.json
```

---

## 🚀 Reproduction Commands

### 1. Run Baseline (TF-IDF + Logistic Regression)
```bash
python train_baseline.py --max_features 10000 --ngram_range 1 2
```

### 2. Fine-Tune PhoBERT-base (Multi-Seed Benchmark)
```bash
# Seed 42
python train_phobert.py --seed 42 --batch_size 16 --lr 2e-5 --epochs 3

# Seed 100
python train_phobert.py --seed 100 --batch_size 16 --lr 2e-5 --epochs 3

# Seed 2024
python train_phobert.py --seed 2024 --batch_size 16 --lr 2e-5 --epochs 3
```

### 3. Run Ablation Study (Without Word Segmentation)
```bash
python train_phobert.py --seed 42 --no_pyvi --epochs 3
```

---

## 📜 Citation
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

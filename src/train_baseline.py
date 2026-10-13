# ==============================================================================
# REPRODUCIBLE BENCHMARK: TF-IDF + LOGISTIC REGRESSION ON UIT-VSFC
# ==============================================================================
import os
import random
import numpy as np
import pandas as pd
from pyvi import ViTokenizer
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    set_seed(42)
    print(">>> Đang tải bộ dữ liệu UIT-VSFC từ HuggingFace...")
    dataset = load_dataset("uitnlp/vietnamese_students_feedback")

    train_texts = [ViTokenizer.tokenize(t) for t in dataset['train']['sentence']]
    train_labels = dataset['train']['sentiment']

    val_texts = [ViTokenizer.tokenize(t) for t in dataset['validation']['sentence']]
    val_labels = dataset['validation']['sentiment']

    test_texts = [ViTokenizer.tokenize(t) for t in dataset['test']['sentence']]
    test_labels = dataset['test']['sentiment']

    print(f"Kích thước tập dữ liệu: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

    # Trích xuất đặc trưng TF-IDF (Chỉ fit trên train để tránh Data Leakage)
    print(">>> Trích xuất đặc trưng TF-IDF (max_features=10000, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    # Huấn luyện Hồi quy Logistic
    print(">>> Đang huấn luyện Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_train, train_labels)

    # Đánh giá trên tập Test
    preds = clf.predict(X_test)
    acc = accuracy_score(test_labels, preds)
    macro_f1 = f1_score(test_labels, preds, average='macro')

    print("\n================== KẾT QUẢ ĐỐI CHỨNG TEST SET ==================")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Macro-F1 : {macro_f1 * 100:.2f}%")
    print("\nChi tiết từng lớp:")
    print(classification_report(test_labels, preds, target_names=["0: Tiêu cực", "1: Trung tính", "2: Tích cực"], digits=4))

if __name__ == "__main__":
    main()

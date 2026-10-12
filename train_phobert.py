# ==============================================================================
# REPRODUCIBLE MAIN MODEL: PHOBERT-BASE FINE-TUNING ON UIT-VSFC
# ==============================================================================
import os
import torch
import numpy as np
from pyvi import ViTokenizer
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    set_seed
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
        'accuracy': acc,
        'macro_f1': macro_f1
    }

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">>> Sử dụng thiết bị: {device}")

    model_name = "vinai/phobert-base"
    print(f">>> Đang tải Tokenizer cho {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    print(">>> Đang tải bộ dữ liệu UIT-VSFC...")
    dataset = load_dataset("uitnlp/vietnamese_students_feedback")

    # Hàm tiền xử lý: Tách từ bằng pyvi và tokenize qua BPE
    def preprocess_function(examples):
        segmented_texts = [ViTokenizer.tokenize(text) for text in examples['sentence']]
        tokenized = tokenizer(
            segmented_texts, 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
        tokenized['label'] = examples['sentiment']
        return tokenized

    print(">>> Đang tiền xử lý toàn bộ tập dữ liệu...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    print(">>> Đang khởi tạo mô hình PhoBERT-base...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3
    )

    training_args = TrainingArguments(
        output_dir="./phobert_vsfc_results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=42,
        fp16=torch.cuda.is_available(),
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        compute_metrics=compute_metrics
    )

    print(">>> Bắt đầu Fine-tuning PhoBERT...")
    trainer.train()

    print("\n>>> Đang đánh giá độc lập trên tập Kiểm tra (Test Set)...")
    test_results = trainer.predict(tokenized_dataset['test'])
    test_preds = np.argmax(test_results.predictions, axis=-1)
    test_labels = tokenized_dataset['test']['label']

    acc = accuracy_score(test_labels, test_preds)
    macro_f1 = f1_score(test_labels, test_preds, average='macro')

    print("\n================== KẾT QUẢ PHOBERT TRÊN TEST SET ==================")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Macro-F1 : {macro_f1 * 100:.2f}%")
    print("\nChi tiết từng lớp:")
    print(classification_report(test_labels, test_preds, target_names=["0: Tiêu cực", "1: Trung tính", "2: Tích cực"], digits=4))

if __name__ == "__main__":
    main()

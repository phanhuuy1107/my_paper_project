# ==============================================================================
# CHƯƠNG 3: PIPELINE TIỀN XỬ LÝ DỮ LIỆU NLP TIẾNG VIỆT (DATA PIPELINE)
# Tác vụ: Phân loại cảm xúc phản hồi sinh viên (UIT-VSFC)
# ==============================================================================

import unicodedata
import re
import hashlib
import numpy as np
from pyvi import ViTokenizer
from datasets import load_dataset
from transformers import AutoTokenizer

def normalize_unicode_nfc(text: str) -> str:
    """
    Chuẩn hóa chuỗi văn bản về bảng mã Unicode dựng sẵn (NFC).
    Giải quyết triệt để lỗi xung đột giữa Unicode Tổ hợp (NFD) và Dựng sẵn (NFC).
    Ví dụ: 'h' + 'o' + dấu huyền -> 'hò' (1 ký tự Unicode duy nhất).
    """
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFC', text)

def clean_vietnamese_text(text: str) -> str:
    """
    Làm sạch văn bản tiếng Việt có kiểm soát:
    - Chuẩn hóa Unicode NFC.
    - Chuẩn hóa khoảng trắng dư thừa, ký tự xuống dòng.
    - Giữ lại các dấu câu biểu cảm quan trọng (!, ?, .) nhưng tách rời để tránh dính từ.
    """
    text = normalize_unicode_nfc(text)
    # Tách các dấu câu quan trọng để không dính vào từ cuối câu
    text = re.sub(r'([.,!?()\[\]{}:;""\'])', r' \1 ', text)
    # Thu gọn nhiều khoảng trắng liên tiếp thành 1 khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def segment_words(text: str) -> str:
    """
    Tách từ tiếng Việt bằng thư viện PyVi.
    Ghép các âm tiết tạo thành từ phức bằng dấu gạch dưới (_).
    Ví dụ: 'học sinh đại học' -> 'học_sinh đại_học'
    """
    text = clean_vietnamese_text(text)
    return ViTokenizer.tokenize(text)

def compute_dataset_hash(texts: list) -> str:
    """
    Tính mã băm SHA-256 trên toàn bộ tập dữ liệu để khóa phiên bản dữ liệu (Data Versioning),
    đảm bảo tính bất biến và khả năng tái lập 100% trong nghiên cứu khoa học.
    """
    hasher = hashlib.sha256()
    for t in texts:
        hasher.update(t.encode('utf-8'))
    return hasher.hexdigest()[:16]

def compute_corpus_statistics(sentences: list, labels: list, split_name: str = "Train"):
    """
    Phân tích thống kê chuyên sâu (EDA - Exploratory Data Analysis):
    - Phân phối nhãn.
    - Thống kê độ dài câu (Min, Max, Mean, Median, 95th Percentile).
    """
    lengths = [len(s.split()) for s in sentences]
    label_counts = {0: 0, 1: 0, 2: 0}
    for lbl in labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    
    total = len(sentences)
    stats = {
        "split": split_name,
        "total_samples": total,
        "hash_sha256": compute_dataset_hash(sentences),
        "neg_0": f"{label_counts[0]} ({label_counts[0]/total*100:.1f}%)",
        "neu_1": f"{label_counts[1]} ({label_counts[1]/total*100:.1f}%)",
        "pos_2": f"{label_counts[2]} ({label_counts[2]/total*100:.1f}%)",
        "len_min": int(np.min(lengths)),
        "len_max": int(np.max(lengths)),
        "len_mean": float(np.mean(lengths)),
        "len_median": float(np.median(lengths)),
        "len_p95": float(np.percentile(lengths, 95)),
    }
    return stats

def main():
    print("=" * 70)
    print(">>> [CHƯƠNG 3] KHỞI ĐỘNG DATA PIPELINE CHO BỘ DỮ LIỆU UIT-VSFC")
    print("=" * 70)
    
    print("\n1. Đang tải bộ dữ liệu UIT-VSFC từ HuggingFace Hub...")
    raw_dataset = load_dataset("uitnlp/vietnamese_students_feedback")
    
    splits = ['train', 'validation', 'test']
    eda_summary = []
    
    for sp in splits:
        texts = raw_dataset[sp]['sentence']
        labels = raw_dataset[sp]['sentiment']
        # Tách từ tiếng Việt
        seg_texts = [segment_words(t) for t in texts]
        stats = compute_corpus_statistics(seg_texts, labels, split_name=sp)
        eda_summary.append(stats)
        
    print("\n2. BẢNG THỐNG KÊ CHI TIẾT (EDA & CORPUS PROFILE):")
    print("-" * 70)
    for s in eda_summary:
        print(f"[{s['split'].upper()}] - Tổng mẫu: {s['total_samples']} | Hash: {s['hash_sha256']}")
        print(f"   • Phân phối nhãn: Tiêu cực={s['neg_0']} | Trung tính={s['neu_1']} | Tích cực={s['pos_2']}")
        print(f"   • Thống kê độ dài từ: Min={s['len_min']}, Max={s['len_max']}, Mean={s['len_mean']:.1f}, Median={s['len_median']}, 95th-Percentile={s['len_p95']:.1f}")
        print("-" * 70)

    print("\n3. THỬ NGHIỆM TOKENIZER BPE CỦA PHOBERT (vinai/phobert-base):")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    sample_raw = "Thầy dạy nhiệt tình và dễ hiểu nhưng bài tập hơi khó ."
    sample_seg = segment_words(sample_raw)
    tokens = tokenizer.tokenize(sample_seg)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    print(f"   • Câu gốc      : {sample_raw}")
    print(f"   • Sau tách từ  : {sample_seg}")
    print(f"   • Tokens (BPE) : {tokens}")
    print(f"   • Input IDs    : {input_ids}")
    print("\n>>> Pipeline dữ liệu đã hoàn thiện và sẵn sàng cho huấn luyện!")

if __name__ == "__main__":
    main()

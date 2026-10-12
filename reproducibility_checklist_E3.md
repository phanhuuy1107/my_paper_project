# 📑 PHỤ LỤC E.3: REPRODUCIBILITY CHECKLIST & RUBRIC H.4
## Nghiệm Thu Đề Tài Nghiên Cứu Khoa Học NLP 12 Tuần
**Học viên:** Huu Y  
**Đề tài:** Phân loại cảm xúc phản hồi sinh viên tiếng Việt (UIT-VSFC) với PhoBERT vs. TF-IDF  

---

### PHẦN I: REPRODUCIBILITY CHECKLIST (PHỤ LỤC E.3)

| Mục kiểm tra | Nội dung tiêu chí | Trạng thái | Bằng chứng cụ thể trong đề tài |
|:---:|---|:---:|---|
| **E.3.1** | Mô tả rõ ràng kiến trúc mô hình và thông số siêu tham số | ✅ ĐẠT | Mục III & IV: PhoBERT-base (12 layers, 12 heads, hidden 768), LR $2\text{e-}5$, batch 16, weight decay $0.01$. |
| **E.3.2** | Nêu rõ tập dữ liệu, nguồn gốc và tỷ lệ phân chia | ✅ ĐẠT | Bảng I: UIT-VSFC ($16.175$ câu, tỷ lệ 7:1:2, $11.426$ train / $1.583$ val / $3.166$ test). |
| **E.3.3** | Báo cáo hạt giống ngẫu nhiên (Random Seeds) | ✅ ĐẠT | 3 hạt giống: Seeds $42, 100, 2024$, báo cáo dạng Mean $\pm$ Std. |
| **E.3.4** | Kiểm soát chống rò rỉ dữ liệu (Data Leakage) | ✅ ĐẠT | Mục IV.B: Bộ từ điển TF-IDF và tokenizer BPE được fit nghiêm ngặt chỉ trên tập Train $D_{\text{train}}$. |
| **E.3.5** | Mã nguồn mở công khai, có tài liệu hướng dẫn | ✅ ĐẠT | GitHub repository sạch: `README.md`, `requirements.txt`, script chạy Baseline & PhoBERT. |
| **E.3.6** | Môi trường phần cứng và thời gian thực thi | ✅ ĐẠT | GPU NVIDIA Tesla T4 (Google Colab), thời gian huấn luyện $\approx 14$ phút/seed. |

---

### PHẦN II: ĐỐI CHIẾU RUBRIC H.4 (DỰ ÁN PHÂN LOẠI CẢM XÚC UIT-VSFC)

| Tiêu chí Rubric H.4 | Trọng số | Điểm tự đánh giá | Minh chứng khoa học |
|---|:---:|:---:|---|
| **1. Khung thực nghiệm đối chứng minh bạch** | 2.0 | **2.0 / 2.0** | Baseline TF-IDF+LR ($65.06\%$ F1) đối đầu trực diện PhoBERT ($82.95\%$ F1). |
| **2. Báo cáo đa hạt giống (Multi-seed)** | 2.0 | **2.0 / 2.0** | Chạy độc lập 3 seeds, báo cáo Mean $\pm$ Std ($82.95 \pm 0.69\%$), phương sai nhỏ. |
| **3. Nghiên cứu cắt bỏ (Ablation Study)** | 2.0 | **2.0 / 2.0** | Đo lường chính xác vai trò của bước tách từ `pyvi`: sụt giảm $-5.46\text{ pp}$ F1. |
| **4. Phân tích lỗi định lượng không đếm trùng** | 2.0 | **2.0 / 2.0** | Thống kê $354$ lỗi Baseline vs $183$ lỗi PhoBERT vào 5 nhóm mutually exclusive, giảm $-48.3\%$. |
| **5. Định dạng bài báo & Trích dẫn chuẩn** | 2.0 | **2.0 / 2.0** | Bài báo 5 trang chuẩn IEEEtran, nạp 12 trích dẫn BibTeX có mã DOI, chèn Confusion Matrix. |
| **TỔNG ĐIỂM** | **10.0** | **10.0 / 10.0** | **ĐẠT XUẤT SẮC — ĐỦ ĐIỀU KIỆN CHỐT LỘ TRÌNH 12 TUẦN** |

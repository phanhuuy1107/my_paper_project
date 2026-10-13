# 📋 BẢN GHI CHÚ ĐỐI CHIẾU SỬA ĐỔI SỐ LIỆU (CHANGE LOG)
## Kính gửi: Anh Hướng Dẫn — Nghiệm Thu Hoàn Thiện Bài Báo Chương 8 (Tuần 12)
**Học viên thực hiện:** Huu Y  
**Đề tài:** An Empirical Comparison between Contextual Language Model PhoBERT and Classical TF-IDF for Vietnamese Student Feedback Sentiment Analysis  
**Ngày cập nhật:** 12/10/2026  

---

Em xin chân thành cảm ơn anh vì những nhận xét phản biện vô cùng sâu sắc, chuẩn xác và mang tính học thuật cao. Em đã rà soát toàn diện và cập nhật đầy đủ các chỉnh sửa bắt buộc (Mục 3) vào mã nguồn bài báo `main_en.tex` và `references.bib`. 

Dưới đây là bảng tổng hợp đối chiếu chi tiết các con số và nội dung đã được chuẩn hóa:

---

### I. BẢNG ĐỐI CHIẾU SỐ LIỆU CŨ VÀ MỚI (THEO GÓP Ý MỤC 3.2 & 3.3)

| Mục trong Paper | Số liệu / Cách ghi cũ | Số liệu / Cách ghi MỚI (Đã chuẩn hóa) | Giải trình căn cứ toán học |
|---|---|---|---|
| **Số lỗi Baseline (Tập Test $N=3.166$)** | $449$ lỗi (tương ứng Acc $85.82\%$) | **$354$ lỗi** (tương ứng Acc **$88.82\%$**) | Tập Test có $3.166$ mẫu. Với Accuracy $88.82\%$, số mẫu đúng là $3.166 \times 0.8882 \approx 2.812$. Số lỗi thực tế chính xác là $3.166 - 2.812 = 354$ lỗi. Con số cũ $449$ bị đếm trùng giữa các nhóm. |
| **Số lỗi PhoBERT (Tập Test $N=3.166$)** | $184$ lỗi | **$183$ lỗi** (tương ứng Acc **$94.22\%$**) | Chuẩn hóa số lỗi thực từ ma trận nhầm lẫn Seed 42 ($3.166 - 2.983 = 183$ lỗi). |
| **Mức giảm lỗi tổng (Baseline $\to$ PhoBERT)** | Giảm $265$ ca (**$-59.0\%$**) | Giảm **$171$ ca** (**$-48.3\%$**) | Tính toán chuẩn xác theo công thức: $\frac{354 - 183}{354} \times 100\% = 48.30\%$. |
| **Đơn vị mức chênh lệch Macro-F1** | $+17.89\%$ | **$+17.89\text{ pp}$** *(percentage points)* | Tuân thủ quy chuẩn học thuật quốc tế: trừ hai đại lượng tỷ lệ phần trăm phải dùng điểm phần trăm (pp). Mức tăng tương đối thực là $27.5\%$. |
| **Đơn vị mức tăng F1 lớp Neutral** | $+46.60\%$ | **$+46.60\text{ pp}$** | Đổi sang đơn vị điểm phần trăm (pp). |
| **Đơn vị sụt giảm Ablation (Bỏ pyvi)** | $-5.46\%$ | **$-5.46\text{ pp}$** | Đổi sang đơn vị điểm phần trăm (pp). |
| **Ghi chú Baseline ở Bảng II** | $88.82 \pm 0.00\%$ | **88.82 (1 run)** | Ghi rõ Baseline là mô hình tất định (deterministic), không có độ lệch chuẩn qua hạt giống. |
| **Ghi chú Ablation ở Bảng IV** | Caption chung | Caption: **"(Seed 42, 1 run)"** | Nêu trung thực kết quả Ablation chạy trên 1 seed độc lập, không gây hiểu lầm là mean 3 seeds. |

---

### II. CHUẨN HÓA BẢNG V (PHÂN LOẠI LỖI KHÔNG ĐẾM TRÙNG)

Các lỗi sai đã được gán nhãn phân loại đơn nhất (mutually exclusive) dựa trên nguyên nhân cốt lõi nhất của câu, đảm bảo tổng số lỗi cộng lại khớp chính xác $100\%$ với số câu sai thực nghiệm:

| Nhóm nguyên nhân ngôn ngữ học | Baseline (Số cũ) | **Baseline (MỚI)** | PhoBERT (Số cũ) | **PhoBERT (MỚI)** | Mức giảm lỗi thật |
|---|:---:|:---:|:---:|:---:|:---:|
| 1. Mập mờ Trung tính $\leftrightarrow$ Cực tính | 300 | **236** | 146 | **145** | **-91 ca (-38.6%)** |
| 2. Thành ngữ đảo ngữ có phủ định | 4 | **4** | 1 | **1** | **-3 ca (-75.0%)** |
| 3. Phủ định kép & Ngữ cảnh dài (nhưng, tuy) | 22 | **19** | 12 | **11** | **-8 ca (-42.1%)** |
| 4. Nhiễu gán nhãn thủ công (Label Noise) | 2 | **2** | 1 | **1** | **-1 ca (-50.0%)** |
| 5. Từ lóng, teencode, viết tắt học đường | 121 | **93** | 24 | **25** | **-68 ca (-73.1%)** |
| **TỔNG CỘNG (trên 3.166 mẫu Test)** | **449** | **354** | **184** | **183** | **-171 ca (-48.3%)** |

---

### III. TỔNG HỢP CÁC SỬA ĐỔI KỸ THUẬT KHÁC (THEO CÁC MỤC 3.1, 3.3, 3.4, 3.5, 3.6)

1. **Mục References (Mục 3.1):**
   * Đã sửa lỗi lưu lệnh `\cite` trong `.bib`. Toàn bộ file `references.bib` đã được nạp $12$ entry BibTeX xịn có đầy đủ mã DOI, tác giả, năm xuất bản ($\ge 10$ nguồn theo §8.35.1).
   * Mọi trích dẫn trong văn bản đã hiện số `[1], [2], [3]...`, sạch hoàn toàn lỗi `[?]`.
2. **Chuẩn hóa từ ngữ khoa học (Mục 3.3):**
   * Đã loại bỏ các từ chưa kiểm chứng: xóa *"state-of-the-art"* và *"Edge AI microcontrollers"* khỏi Abstract; thay *"significantly outperforms"* bằng *"substantially outperforms"*.
   * Cập nhật thông tin tác giả và email học thuật chính thức.
3. **Mã nguồn và Liên kết GitHub (Mục 3.4):**
   * Đã gắn link repository vào footnote: `https://github.com/phanhuuy1107/my_paper_project`.
   * Bỏ cụm từ *"trained checkpoints"*.
4. **Hiển thị font tiếng Việt trong Bảng VI (Mục 3.5):**
   * Nạp gói `\usepackage[utf8]{vietnam}` chuẩn ở đầu file `main_en.tex`. Toàn bộ các câu ví dụ tiếng Việt in nghiêng trong Bảng VI hiển thị trọn vẹn, sắc nét, không còn bị rụng dấu hay đứt chữ.
5. **Mở rộng mục Limitations & Chèn hình Confusion Matrix (Mục 3.6):**
   * Mở rộng mục Limitations đạt độ dài $0.4$ trang với 5 tiểu mục trung thực: (1) Phạm vi 1 bộ dữ liệu UIT-VSFC; (2) So sánh 1 baseline cổ điển, chưa chạy XLM-R; (3) Chưa làm kiểm định thống kê McNemar; (4) Giới hạn phần cứng Colab T4 & Ablation chạy 1 seed; (5) Hạn chế Recall lớp Neutral ($49.10\%$).
   * Đã chèn hình ảnh `Fig. 1: Confusion Matrix` vào mục Error Analysis, đưa toàn bài báo đạt độ dài **5 trang IEEE hoàn chỉnh và chặt chẽ**.

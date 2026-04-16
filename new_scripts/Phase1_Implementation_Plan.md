# Kế Hoạch Triển Khai Giai Đoạn 1: Định Vị H-Neuron (Full Pipeline Cũ & Mới)

Mục tiêu là tự tay trích xuất 2 tập hợp H-Neurons (từ phương pháp cũ và phương pháp mới) trên mô hình `gemma-3-4b` và đo lường độ trùng khớp (IoU).

## Phân Hệ 1: Trích Xuất Theo Phương Pháp Cũ (Supervised L1 Pipeline)

Để có danh sách $H_{old\_method}$, chúng ta sẽ chạy lại toàn bộ mã nguồn cũ trên mô hình `gemma-3-4b`. 

### [NEW] `run_old_pipeline.sh` (Script tự động hóa)
1. **Thu thập dữ liệu QA**: Chạy `collect_responses.py` với TriviaQA. Bước này sinh ra các mẫu trả lời và dùng LLM/Rule gán nhãn True/False.
2. **Lọc token trọng tâm**: Chạy `extract_answer_tokens.py` dùng LLM (`gpt-4o` hoặc Rule-based) để trích xuất list "answer tokens".
3. **Cân bằng dữ liệu**: Chạy `sample_balanced_ids.py` cân bằng nhãn True/False vào `train_qids.json`.
4. **Trích xuất CETT Activation**: Chạy `extract_activations.py` với cơ `--locations answer_tokens`. Hook vào `down_proj` để tính giá trị CETT chuẩn hóa.
5. **Huấn luyện L1 Classifier**: Chạy `classifier.py` dùng Logistic Regression L1 Penalty.
6. **Lưu H-Neuron**: Chạy/Viết phụ thêm một hàm xuất danh sách nơ-ron có `weight > 0` thành tệp `old_h_neurons.json`.

---

## Phân Hệ 2: Trích Xuất Theo Phương Pháp Mới (Dream Probing)

### [NEW] `create_datasets.py`
Sinh 2 bộ dataset khảo sát (Tiếng Anh, quy mô 500-1000 câu):
- `faithful_data.jsonl`: Lấy một subset sự thật hiển nhiên (Từ TriviaQA hoặc TruthfulQA).
- `nonsense_data.jsonl`: Sinh ra các prompt như *Semantic Salad* và *Non-existent Entities*.

### [NEW] `extract_variance_spike.py`
Khảo sát sự bất thường (Variance Spike) mà không cần Labeling:
1. Đọc lại `gemma-3-4b`, hook vào các tầng `down_proj`.
2. Chạy với `faithful_data` -> Ra Phương sai mức nền ($Var_{faithful}$).
3. Chạy với `nonsense_data` -> H-Neuron bùng nổ phương sai ($Var_{nonsense}$).
4. Lưu **0.1% nơ-ron** có $\Delta Var$ cao nhất thành `new_h_neurons.json`.

---

## Phân Hệ 3: Đối Chiếu

### [NEW] `compare_h_neurons.py`
1. Đọc $H_{old\_method}$ và $H_{new\_method}$.
2. Tính toán độ trùng khớp **Intersection over Union (IoU)**.
3. Vẽ biểu đồ Heatmap (trùng khớp theo Layer).

## Hướng giải quyết tiếp theo
- Nếu xác nhận có thể tốn chi phí API cho `gpt-4o` của phương pháp cũ (hoặc chỉ định mock rule-based).
- Chốt số lượng sample 500 - 1000 câu.

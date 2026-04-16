# Hướng Dẫn Triển Khai Nghiên Cứu: Dream Probing & Soft-Pruning Sleep Cycle
**Chủ đề:** Giảm thiểu Hallucination trong LLM thông qua cơ chế nơ-ron vi mô và mô phỏng giấc ngủ sinh học.

---

## 1. Tổng Quan Kỹ Thuật (Core Architecture)

[cite_start]Hướng nghiên cứu này tập trung vào **H-Neurons** (Hallucination-Associated Neurons) — một tập con nơ-ron thưa thớt (chiếm chưa đến 0.1% tổng số nơ-ron) trong mạng Feed-Forward (FFN) có khả năng dự đoán sự xuất hiện của ảo giác một cách đáng tin cậy[cite: 10, 633]. [cite_start]Thay vì sử dụng phương pháp cắt tỉa (pruning) truyền thống làm mất tham số vĩnh viễn, hướng này đề xuất một **"chu kỳ ngủ" (sleep cycle)** mềm mại để giảm dần trọng số của H-Neurons mà không phá vỡ cấu trúc mô hình[cite: 635].

### Chỉ số đóng góp nơ-ron (CETT)
[cite_start]Để xác định mức độ ảnh hưởng của từng nơ-ron, ta sử dụng độ đo **CETT** (Contribution of Neurons)[cite: 88, 348]. [cite_start]Chỉ số này định lượng phần dòng thông tin tại token $t$ được tạo ra bởi nơ-ron $j$[cite: 365]:
$$CETT = \frac{|n(x)|}{|y|}$$
[cite_start]Trong đó $|n(x)|$ là độ lớn vector hình chiếu của nơ-ron và $|y|$ là tổng vector đầu ra của lớp[cite: 64, 75].

---

## 2. Giai Đoạn 1: Dream Probing Protocol (Định vị không giám sát)

[cite_start]Mục tiêu là định vị H-Neurons mà không cần các bộ dữ liệu gán nhãn đúng/sai tốn kém[cite: 663, 671].

* [cite_start]**Cơ chế nonsense inputs:** Đưa vào mô hình các chuỗi token ngẫu nhiên, văn bản đảo lộn ngữ pháp (semantic salad) hoặc các thực thể không tồn tại[cite: 671].
* [cite_start]**Giả thuyết phương sai (Variance Spike):** H-Neurons, vốn mã hóa xu hướng "tuân thủ thái quá" (over-compliance), sẽ phản ứng cực mạnh với các đầu vào vô nghĩa[cite: 44, 671]. [cite_start]Ta xác định nơ-ron có phương sai CETT cao bất thường khi gặp nonsense prompts so với faithful prompts[cite: 671, 672].
* [cite_start]**Lợi ích:** Phương pháp này hoàn toàn không giám sát (unsupervised), cho phép dò tìm H-Neurons trên bất kỳ kiến trúc mô hình nào[cite: 671].

---

## 3. Giai Đoạn 2: Thuật Toán Soft Sleep Cycle

[cite_start]Sau khi định vị được H-Neurons, hệ thống thực hiện làm suy yếu chúng một cách có lộ trình thay vì loại bỏ hoàn toàn[cite: 635].

### Công thức suy giảm (Multiplicative Downscaling)
[cite_start]Áp dụng lên trọng số $W_H$ của các H-Neurons qua $k$ chu kỳ[cite: 673]:
$$W_H \leftarrow W_H \cdot \gamma^k$$
[cite_start]Trong đó hệ số suy giảm $\gamma \in (0.7, 0.95)$[cite: 673].

### Bảo vệ kiến thức cốt lõi (EWC)
[cite_start]Để đảm bảo việc suy giảm không gây ra hiện tượng sụp đổ kiến thức hữu ích (useful knowledge)[cite: 664, 668]:
* [cite_start]Sử dụng hàm phạt **EWC** (Elastic Weight Consolidation) để bảo vệ các nơ-ron quan trọng (critical neurons)[cite: 673, 900].
* [cite_start]Chỉ tập trung can thiệp vào các nơ-ron có trọng số dương trong bộ phân loại ảo giác (positive weight)[cite: 78, 194].

---

## 4. Giai Đoạn 3: Thiết Kế Giấc Ngủ Hai Pha (Two-Phase Sleep Design)

[cite_start]Mô phỏng chu kỳ ngủ sinh học để tối ưu hóa sự ổn định của mô hình[cite: 675].

1.  [cite_start]**Pha SWS Analog (Slow-Wave Sleep):** * **Hành động:** Thực hiện giảm độ lớn (magnitude downscaling) toàn cục trên toàn bộ FFN[cite: 659, 675].
    * [cite_start]**Mục tiêu:** Cân bằng lại các synapse, loại bỏ nhiễu tích lũy trong quá trình suy luận[cite: 635].
2.  [cite_start]**Pha REM Analog (Rapid Eye Movement):** * **Hành động:** Phát lại có đối chiếu (contrastive replay) giữa các cặp dữ liệu đúng và ảo giác[cite: 675].
    * [cite_start]**Mục tiêu:** Củng cố ranh giới giữa bộ nhớ tham số (parametric memory) và ngữ cảnh thực tế (context)[cite: 675].

---

## 5. Lộ Trình Triển Khai Cho antigravity (Implementation Roadmap)

### Bước 1: Trích xuất đặc trưng (Feature Extraction)
* [cite_start]Sử dụng PyTorch hooks để ghi lại giá trị kích hoạt của các tầng MLP[cite: 351, 413].
* [cite_start]Tính toán chỉ số CETT tại vị trí các answer tokens[cite: 77, 369].

### Bước 2: Thử nghiệm Nonsense Probing
* [cite_start]Tạo dataset nonsense bao gồm: token ngẫu nhiên và câu hỏi về thực thể giả (NonExist)[cite: 101, 671].
* [cite_start]Lọc ra danh sách 0.1% nơ-ron có phương sai CETT lớn nhất[cite: 10, 672].

### Bước 3: Can thiệp và Đánh giá
* [cite_start]Triển khai can thiệp `Activation Scaling` trong quá trình suy luận để kiểm tra tính nhân quả[cite: 45, 195, 415].
* [cite_start]Sử dụng hệ số $\alpha < 1$ để triệt tiêu nơ-ron gây lỗi và đo lường sự thay đổi trên các benchmark: **TruthfulQA** và **HaluEval**[cite: 191, 679].

---

> [cite_start]**Ghi chú quan trọng:** H-Neurons không chỉ đơn thuần lưu trữ lỗi kiến thức mà đại diện cho xu hướng ưu tiên sự hài lòng của người dùng hơn là tính xác thực (over-compliance)[cite: 47, 198]. [cite_start]Việc triệt tiêu chúng một cách thông minh sẽ giúp mô hình "biết nói không" khi không chắc chắn[cite: 208, 211].
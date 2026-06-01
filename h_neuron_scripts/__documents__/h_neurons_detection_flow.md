# H-Neurons Detection Flow trong `h_neuron_scripts`

Tài liệu này mô tả chi tiết luồng xử lí của toàn bộ thư mục `h_neuron_scripts`, từ tạo dữ liệu, lọc mẫu, trích xuất answer tokens, trích xuất activation/CETT, train classifier phát hiện H-Neurons, đến can thiệp model.

> Mục tiêu tổng quát: tìm các neuron trong MLP/FFN của LLM có tương quan dương với hallucinated/false answers. Trong implementation này, một H-Neuron được hiểu thực dụng là một feature `(layer, neuron)` có trọng số dương trong sparse Logistic Regression classifier được train để phân biệt false-answer activations với true/non-answer activations.

---

## 1. Bức tranh tổng thể

```text
TriviaQA parquet
    |
    v
collect_responses.py
    - sample nhiều response cho mỗi question bằng vLLM
    - judge từng response là true / false / uncertain / error
    v
samples.jsonl
    |
    +--> filter_consistent_samples.py
    |       - giữ các sample có toàn bộ judges đồng nhất all-true hoặc all-false
    |
    +--> preselect_balanced_samples.py
    |       - optional: chọn số lượng all-true/all-false cân bằng từ raw samples
    v
consistent/balanced samples.jsonl
    |
    v
extract_answer_tokens.py
    - chọn representative response
    - tokenize response bằng tokenizer của target model
    - gọi LLM để chọn minimal answer-token indices
    v
answer_tokens.jsonl
    |
    v
sample_balanced_ids.py
    - đọc nhãn judge trong answer_tokens.jsonl
    - sample cân bằng qid true/false
    v
train_qids.json / test_qids.json
    |
    v
extract_activations.py
    - chạy target model forward trên full chat question+response
    - hook mọi down_proj trong MLP
    - tính CETT-like tensor [layers, tokens, neurons]
    - slice theo region: input/output/answer_tokens/all_except_answer_tokens
    - aggregate token dimension bằng mean hoặc max
    v
activation directories
    - answer_tokens/act_<qid>.npy
    - all_except_answer_tokens/act_<qid>.npy
    - input/act_<qid>.npy
    - output/act_<qid>.npy
    |
    v
classifier.py
    - load .npy, flatten [layers, neurons]
    - train LogisticRegression
    - positive class = false answer tokens
    - weights > 0 = candidate H-Neurons
    v
detector.pkl / classifier.pkl
    |
    v
intervene_model.py
    - map flat classifier weights -> layer/neuron
    - scale selected columns in down_proj weights
```

---

## 2. Chuẩn dữ liệu chính giữa các bước

### 2.1. Raw sampled responses JSONL

Được tạo bởi `collect_responses.py`.

Mỗi dòng là một object có đúng một key là `qid`:

```json
{
  "<qid>": {
    "question": "<question plus answer-only suffix>",
    "responses": ["response 1", "response 2", "..."],
    "judges": ["true", "false", "uncertain", "error"],
    "ground_truth": ["alias 1", "alias 2"]
  }
}
```

Ý nghĩa:

- `question`: câu hỏi đã được thêm suffix yêu cầu trả lời ngắn.
- `responses`: danh sách câu trả lời sampled từ model.
- `judges`: nhãn correctness tương ứng từng response.
- `ground_truth`: các alias/normalized aliases lấy từ TriviaQA.

### 2.2. Consistent samples JSONL

Được tạo bởi `filter_consistent_samples.py` hoặc được lọc trực tiếp trong `extract_answer_tokens.py`.

Điều kiện consistent:

```text
judges == ["true", "true", ..., "true"]
hoặc
judges == ["false", "false", ..., "false"]
```

Các sample mixed, uncertain, error bị loại khỏi luồng H-Neuron detection vì nhãn không ổn định.

### 2.3. Answer-token JSONL

Được tạo bởi `extract_answer_tokens.py`.

```json
{
  "<qid>": {
    "question": "<question>",
    "response": "<representative response>",
    "tokenized_response": ["token 0", "token 1"],
    "answer_tokens": ["token selected by LLM"],
    "judge": "true"
  }
}
```

Ý nghĩa:

- `response`: representative response, hiện lấy bằng response xuất hiện nhiều nhất.
- `tokenized_response`: response tokenized bằng tokenizer của target model.
- `answer_tokens`: minimal answer span do LLM extractor chọn.
- `judge`: nhãn nhất quán của toàn sample: `true` hoặc `false`.

### 2.4. Balanced IDs JSON

Được tạo bởi `sample_balanced_ids.py`.

```json
{
  "t": ["qid_true_1", "qid_true_2"],
  "f": ["qid_false_1", "qid_false_2"]
}
```

Ý nghĩa:

- `t`: danh sách qid có answer consistent-true.
- `f`: danh sách qid có answer consistent-false/hallucinated.

### 2.5. Activation `.npy`

Được tạo bởi `extract_activations.py`.

Mỗi file:

```text
act_<qid>.npy
```

Có shape logic:

```text
[layers, neurons]
```

Vì trước đó tensor đầy đủ có shape:

```text
[layers, tokens, neurons]
```

Sau khi slice token region và aggregate theo token dimension, còn lại `[layers, neurons]`.

---

## 3. Bước 1 — `collect_responses.py`: sample response và judge correctness

### 3.1. Mục đích

Script này tạo dữ liệu ban đầu cho hallucination detection bằng cách:

1. Load TriviaQA parquet.
2. Với mỗi question, sample nhiều câu trả lời từ target model bằng vLLM.
3. Judge từng response là đúng/sai/uncertain.
4. Ghi JSONL để dùng cho lọc consistency.

### 3.2. Input chính

Arguments quan trọng:

```text
--model_path      path/model id của model dùng để sinh response
--data_path       parquet file của TriviaQA
--output_path     nơi ghi samples JSONL
--sample_num      số response sampled cho mỗi question
--max_samples     giới hạn số question cần xử lí
--judge_type      rule hoặc llm
--api_key         API key nếu dùng LLM judge
--base_url        OpenAI-compatible endpoint
--judge_model     model judge
--gpu_util        vLLM GPU utilization
--tp_size         tensor parallel size
```

### 3.3. Model sampling

`ConsistencySampler.__init__` khởi tạo vLLM:

```text
LLM(
  model=args.model_path,
  tensor_parallel_size=args.tp_size or torch.cuda.device_count(),
  gpu_memory_utilization=args.gpu_util,
  trust_remote_code=True
)
```

Sampling config:

```text
temperature = 1.0
top_p       = 0.9
top_k       = 50
max_tokens  = 50
```

Với mỗi question, prompt được tạo dạng:

```text
<question> Respond with the answer only, without any explanation.
```

Sau đó gọi:

```text
sampling_llm.chat(messages, sampling_params)
```

Lặp `sample_num` lần để có nhiều response cho cùng question.

### 3.4. Rule judge

Nếu `--judge_type rule`, script normalize response và ground truth aliases.

Normalize gồm:

- lowercase
- thay `_` bằng space
- bỏ punctuation
- bỏ article `a/an/the`
- fix whitespace

Một response được xem là `true` nếu bất kỳ normalized ground truth alias nào là substring của normalized response.

```text
if normalized_ground_truth in normalized_response:
    judge = "true"
else:
    judge = "false"
```

Ưu điểm:

- Rẻ, không cần API.
- Dễ reproducible.

Nhược điểm:

- Dễ false negative nếu paraphrase.
- Dễ false positive nếu alias xuất hiện trong ngữ cảnh phủ định.

### 3.5. LLM judge

Nếu `--judge_type llm`, script gọi OpenAI-compatible client với prompt:

```text
Question: ...
Response: ...
Correct Answers: ...
Please judge whether the response is correct or not.
Return 't' if correct, 'f' if incorrect.
```

Kết quả:

- nếu response chứa `t` -> `true`
- nếu response chứa `f` -> `false`
- nếu lỗi liên tục -> `error`

Script có retry tối đa 5 lần cho judge API.

### 3.6. Uncertainty pre-filter

Trước khi judge correctness, script kiểm tra một số cụm từ bất định:

```text
"don't know", "cannot", "not provided", "no information"
```

Nếu response chứa các cụm này, judge sẽ là:

```text
uncertain
```

Những mẫu này về sau thường bị loại vì không phải all-true/all-false.

### 3.7. Resume/skip processed qids

`load_existing_qids(output_path)` đọc output JSONL cũ và lấy các qid đã xử lí. Nếu gặp qid đã tồn tại, script skip để tránh xử lí lại.

### 3.8. Output

Mỗi qid được ghi một dòng JSONL:

```json
{
  "qid": {
    "question": "... Respond with the answer only, without any explanation.",
    "responses": ["..."],
    "judges": ["true", "false"],
    "ground_truth": ["..."]
  }
}
```

---

## 4. Bước 2A — `filter_consistent_samples.py`: lọc sample nhất quán

### 4.1. Mục đích

Script này giữ lại những qid có tất cả judges cùng một nhãn:

```text
all true  -> keep
all false -> keep
mixed     -> drop
invalid   -> drop
```

### 4.2. Vì sao cần consistency?

H-Neuron detection cần nhãn rõ ràng:

- `true`: model biết và trả lời đúng ổn định.
- `false`: model hallucinate hoặc trả lời sai ổn định.

Nếu cùng một question lúc đúng lúc sai, activation có thể phản ánh stochasticity, ambiguity, prompt sensitivity, hoặc judging noise, không còn là nhãn sạch cho classifier.

### 4.3. Logic

`judge_label(content)`:

```text
judges = content["judges"]
labels = set(lowercase(judges))

if labels == {"true"}:  return "true"
if labels == {"false"}: return "false"
else:                    return "mixed"
```

### 4.4. Output

JSONL giữ nguyên format input, chỉ bỏ dòng không đạt điều kiện.

---

## 5. Bước 2B — `preselect_balanced_samples.py`: chọn cân bằng all-true/all-false ở cấp raw sample

### 5.1. Mục đích

Script này optional. Nó chọn trước một số lượng bằng nhau giữa all-true và all-false samples từ raw samples JSONL.

Khác với `sample_balanced_ids.py`:

- `preselect_balanced_samples.py` hoạt động trước answer-token extraction.
- `sample_balanced_ids.py` hoạt động sau answer-token extraction.

### 5.2. Input/Output

Input:

```text
samples.jsonl
```

Output:

```text
balanced_samples.jsonl
```

Arguments:

```text
--num_per_class   số sample mỗi class, default 91
--seed            random seed
```

### 5.3. Logic

1. Đọc toàn bộ rows.
2. Phân loại row thành `true`, `false`, `mixed`, `invalid`.
3. Lấy:

```text
selected_per_class = min(num_per_class, len(true_rows), len(false_rows))
```

4. Random sample `selected_per_class` từ mỗi class.
5. Gộp và shuffle.
6. Ghi JSONL.

### 5.4. Khi nào dùng?

Dùng khi muốn giảm chi phí API ở bước `extract_answer_tokens.py`. Vì extraction answer tokens cần gọi LLM, chọn cân bằng trước giúp không gọi API quá nhiều cho class dư thừa.

---

## 6. Bước 3 — `extract_answer_tokens.py`: xác định answer-token span

### 6.1. Mục đích

Script này biến một consistent sample thành một record có answer span cụ thể trong response.

H-Neuron paper/ý tưởng muốn nhìn activation ở các token liên quan trực tiếp đến answer, không phải toàn response. Vì vậy cần biết token nào trong generated response là answer.

### 6.2. Input

Input là JSONL từ `collect_responses.py`, `filter_consistent_samples.py`, hoặc `preselect_balanced_samples.py`.

Arguments quan trọng:

```text
--input_path       samples JSONL
--output_path      answer_tokens JSONL
--tokenizer_path   tokenizer của target model
--resume           append/skip processed ids thay vì ghi đè
--api_key          single API key
--api_keys         nhiều API keys để rotate
--base_url         OpenAI-compatible endpoint
--llm_model        LLM dùng để chọn answer token indices
```

### 6.3. Tokenizer

Script load tokenizer:

```text
AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
```

Response được tokenize bằng:

```text
tokenizer.encode(response, add_special_tokens=False)
tokenizer.decode([token_id]) cho từng token
```

Kết quả là list token string như:

```text
["▁The", "▁Spirit", "▁of", "▁Ec", "st", "asy"]
```

### 6.4. Điều kiện sample được xử lí

Trong `run()`, script chỉ xử lí sample nếu:

```text
len(set(judges)) == 1
and "uncertain" not in judges
and "error" not in judges
```

Tức là chỉ all-true hoặc all-false.

### 6.5. Chọn representative response

Nếu có nhiều responses giống/khác nhau trong một qid, script chọn response xuất hiện nhiều nhất:

```text
rep_response = max(set(responses), key=responses.count)
```

Điểm cần chú ý:

- Nếu nhiều response có cùng tần suất, Python có thể chọn theo thứ tự set không ổn định tuyệt đối.
- Nhưng trong thực tế nếu model lặp lại answer nhiều lần, cách này lấy mode response.

### 6.6. LLM answer-token extractor

Prompt chính:

```text
Question: {question}
Response: {response}
Tokenized Response with indices: [(0, token0), (1, token1), ...]
Please identify the answer span in the tokenized response.
Return only a JSON array of integer token indices...
```

Có một few-shot example về Rolls Royce / Spirit of Ecstasy.

LLM phải trả về JSON array integer indices:

```json
[21, 22, 23, 24, 25, 26]
```

Validation:

```text
- output là list
- mọi phần tử là int
- indices sorted ascending
- mọi index nằm trong [0, len(tokens))
```

Nếu valid, script convert indices sang token strings:

```text
answer_tokens = [tokens[i] for i in extracted_indices]
```

### 6.7. API-key rotation and resume

Script hỗ trợ nhiều API keys và bắt đầu với key đầu tiên. Nếu API trả về quota/rate-limit style error, script rotate sang key kế tiếp trong cùng process và retry. Script không lưu RPM/RPD quota state ra file.

Resume chỉ dựa vào `output_path`: khi `--resume` được bật, script đọc các qid đã có trong output JSONL, append vào file hiện tại, và skip các qid đã xử lí. Các file quota state cũ như `*.usage.json` không còn được đọc hoặc ghi.

### 6.8. Output

Mỗi qid output:

```json
{
  "qid": {
    "question": "...",
    "response": "representative response",
    "tokenized_response": ["..."],
    "answer_tokens": ["..."],
    "judge": "true"
  }
}
```

---

## 7. Bước 4 — `sample_balanced_ids.py`: tạo train/test qid map cân bằng

### 7.1. Mục đích

Sau khi có answer-token JSONL, script này tạo danh sách qid cân bằng giữa true và false.

Classifier train tốt hơn nếu số positive/negative không quá lệch.

### 7.2. Logic

Đọc từng dòng answer-token JSONL:

```text
qid = top-level key
label = data[qid]["judge"]

if label == "true":  true_ids.append(qid)
if label == "false": false_ids.append(qid)
```

Số sample thực tế mỗi class:

```text
actual_samples = min(num_samples, len(true_ids), len(false_ids))
```

Random sample:

```text
sampled_t = random.sample(true_ids, actual_samples)
sampled_f = random.sample(false_ids, actual_samples)
```

### 7.3. Output

```json
{
  "t": ["..."],
  "f": ["..."]
}
```

File này được dùng bởi:

- `extract_activations.py` để chỉ extract qid cần train/test.
- `classifier.py` để gán label khi load activations.

---

## 8. Bước 5 — `extract_activations.py`: trích xuất CETT-like neuron activations

Đây là script lõi của H-Neuron detection.

### 8.1. Mục đích

Với mỗi sample đã có:

- question
- response
- answer_tokens
- judge label

script chạy model forward trên full chat sequence và hook các MLP `down_proj` để lấy tín hiệu neuron contribution.

Output cuối cùng là vector/tensor feature cho từng qid:

```text
[layers, neurons]
```

Mỗi phần tử đại diện cho mức đóng góp/độ kích hoạt của một neuron tại region token được chọn.

### 8.2. Arguments

```text
--model_path        target model
--input_path        answer_tokens.jsonl
--train_ids_path    qid map {"t": [...], "f": [...]}
--output_root       root directory để save .npy
--locations         regions cần extract
--method            mean hoặc max aggregation
--use_mag           nhân activation với norm của down_proj weight column
--use_abs           lấy trị tuyệt đối activation
```

Locations hợp lệ:

```text
input
output
answer_tokens
all_except_answer_tokens
```

### 8.3. Load model/tokenizer

```text
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### 8.4. CETTManager: hook `down_proj`

`CETTManager` quản lí hook và tính tensor contribution.

Khi khởi tạo:

```text
self.activations = []
self.output_norms = []
self.hooks = []
self._register_hooks()
self.weight_norms = self._get_weight_norms()
```

### 8.5. Hook function

Với mọi module có tên chứa `down_proj`:

```text
module.register_forward_hook(hook_fn)
```

Hook lưu:

```text
input[0].detach()
```

và:

```text
torch.norm(output.detach(), dim=-1, keepdim=True)
```

Trong MLP kiểu Llama/Mistral/Qwen, `down_proj` thường map từ intermediate dimension về hidden dimension:

```text
intermediate activations [tokens, intermediate_size]
    -> down_proj weight [hidden_size, intermediate_size]
    -> hidden contribution [tokens, hidden_size]
```

Do đó `input[0]` của `down_proj` tương ứng activation của từng MLP neuron.

### 8.6. Weight norms

`_get_weight_norms()` lấy norm theo cột của `down_proj.weight`:

```text
torch.norm(module.weight.data, dim=0)
```

Vì mỗi cột ứng với output direction của một intermediate neuron khi đi qua `down_proj`.

Shape logic:

```text
weight_norms: [layers, neurons]
```

### 8.7. Tính CETT-like tensor

`get_cett_tensor(use_abs=True, use_mag=True)`:

1. Chuẩn hóa shape, bỏ batch dimension nếu batch size = 1.
2. Stack activations từ mọi layer:

```text
acts = torch.stack(self.activations).transpose(0, 1)
```

Comment trong code:

```text
acts: [tokens, layers, neurons]
```

3. Stack output norms:

```text
norms: [tokens, layers, 1]
```

4. Nếu `use_abs`:

```text
acts = abs(acts)
```

5. Nếu `use_mag`:

```text
acts = acts * weight_norms.unsqueeze(0)
```

6. Normalize theo output norm:

```text
cett = acts / (norms + 1e-8)
```

7. Transpose về:

```text
[layers, tokens, neurons]
```

### 8.8. Trực giác CETT-like score

Mỗi score xấp xỉ:

```text
score(layer l, token t, neuron n)
  = |activation(l,t,n)| * ||down_proj_weight_column(l,n)||
    / ||down_proj_output(l,t)||
```

Nếu không dùng abs hoặc mag, công thức thay đổi theo flags.

Ý nghĩa:

- activation lớn: neuron đang firing mạnh;
- weight column norm lớn: neuron có khả năng ảnh hưởng output lớn;
- output norm normalization: chuẩn hóa theo magnitude của layer output tại token đó.

Đây là contribution-style feature, không chỉ là raw activation.

### 8.9. Forward input construction

Với mỗi sample:

```text
msgs = [
  {"role": "user", "content": data["question"]},
  {"role": "assistant", "content": data["response"]}
]
```

Tokenized bằng chat template:

```text
tokenizer.apply_chat_template(
    msgs,
    return_tensors="pt",
    add_generation_prompt=False
)
```

Sau đó chạy model forward không gradient:

```text
with torch.no_grad():
    model(...)
```

Hook tự động thu activations/output norms.

### 8.10. Xác định token regions

Function `get_region_indices(...)` trả về các region:

```text
{
  "input": (0, input_len),
  "output": (output_start, output_end),
  "answer_tokens": (ans_start, ans_end) hoặc None
}
```

#### 8.10.1. Full tokens

```text
full_tokens = [tokenizer.decode([tid]) for tid in full_ids[0]]
```

#### 8.10.2. Input region

Script tokenize user-only message:

```text
user_encoding = tokenizer.apply_chat_template([
  {"role": "user", "content": question}
], return_tensors="pt")
```

Sau đó:

```text
input_len = user_ids.shape[1] - 1
```

Trừ 1 để loại potential separator/header.

Region:

```text
input = (0, input_len)
```

#### 8.10.3. Output region

```text
output_start = input_len + 1
output_end = len(full_tokens) - 1
```

Trừ 1 cuối để loại EOS.

Region:

```text
output = (output_start, output_end)
```

#### 8.10.4. Answer-token region

Answer tokens được normalize nhẹ:

```text
token.replace("▁", " ").replace("Ġ", " ")
```

Sau đó script tìm contiguous subsequence trong `full_tokens`:

```text
for i in range(output_start, len(full_tokens) - m + 1):
    if full_tokens[i : i + m] == answer_tokens:
        ans_start = i
        ans_end = i + m
        break
```

Nếu match được:

```text
answer_tokens = (ans_start, ans_end)
```

Nếu không match:

```text
answer_tokens = None
```

### 8.11. Slice locations

Với từng `loc` trong `--locations`:

#### `input`

```text
selected_cett = cett_full[:, input_start:input_end, :]
```

#### `output`

```text
selected_cett = cett_full[:, output_start:output_end, :]
```

#### `answer_tokens`

```text
selected_cett = cett_full[:, ans_start:ans_end, :]
```

Nếu không tìm được answer span, skip location này.

#### `all_except_answer_tokens`

Nếu có answer span:

```text
seg1 = cett_full[:, :ans_s, :]
seg2 = cett_full[:, ans_e:, :]
selected_cett = torch.cat([seg1, seg2], dim=1)
```

Lưu ý: implementation hiện tại lấy tất cả token trước answer và sau answer trong full sequence, không chỉ output tokens ngoài answer. Tên location là `all_except_answer_tokens`, không phải `output_except_answer_tokens`.

### 8.12. Aggregate token dimension

Sau khi chọn region, tensor có shape:

```text
[layers, selected_tokens, neurons]
```

Nếu `--method mean`:

```text
final_act = selected_cett.mean(dim=1)
```

Nếu `--method max`:

```text
final_act, _ = selected_cett.max(dim=1)
```

Output shape:

```text
[layers, neurons]
```

### 8.13. Save activations

Mỗi location có folder riêng:

```text
<output_root>/<location>/act_<qid>.npy
```

Ví dụ:

```text
activations/answer_tokens/act_123.npy
activations/all_except_answer_tokens/act_123.npy
```

---

## 9. Bước 6 — `classifier.py`: train detector và xác định H-Neurons

### 9.1. Mục đích

Train một classifier tuyến tính trên feature `[layers, neurons]` để phân biệt hallucinated answer tokens với true/non-answer tokens.

Classifier hiện dùng:

```text
sklearn.linear_model.LogisticRegression
```

Default:

```text
penalty = l1
solver  = liblinear
C       = 1.0
```

L1 penalty làm classifier sparse, tức chỉ một phần nhỏ features có weight non-zero. Điều này phù hợp với mục tiêu tìm neuron cụ thể.

### 9.2. Input arguments

```text
--model_path          model path để đọc config
--train_ids           JSON {"t": [...], "f": [...]}
--train_ans_acts      folder activations answer_tokens
--train_other_acts    folder activations other tokens, cần cho 3-vs-1
--test_ids            JSON test ids
--test_acts           folder test activations
--save_model          nơi lưu detector.pkl/classifier.pkl
--load_model          load pretrained classifier
--train_mode          1-vs-1 hoặc 3-vs-1
--penalty             l1 hoặc l2
--C                   inverse regularization strength
--solver              liblinear hoặc saga nếu cần
```

### 9.3. Feature loading

`load_data(ids_path, ans_acts_dir, other_acts_dir=None, mode="1-vs-1")` đọc id map:

```json
{
  "t": [...],
  "f": [...]
}
```

Với mỗi qid, load:

```text
np.load("act_<qid>.npy").flatten()
```

Nếu activation shape là `[layers, neurons]`, flatten thành:

```text
[layers * neurons]
```

Thứ tự flatten mặc định của NumPy là C-order:

```text
flat_index = layer_idx * num_neurons + neuron_idx
```

Điều này rất quan trọng cho bước map ngược trong `intervene_model.py`.

### 9.4. Label trong mode `1-vs-1`

Mode này so sánh trực tiếp false answer tokens với true answer tokens.

```text
False answer-token activations -> label 1
True answer-token activations  -> label 0
```

Dataset:

```text
X = [false_answer_features, true_answer_features]
y = [1, 0]
```

Ý nghĩa:

- classifier học những neurons/features cao ở false answer hơn true answer.
- Weight dương gợi ý feature liên quan hallucination.

### 9.5. Label trong mode `3-vs-1`

Mode này làm positive class hẹp hơn và negative class rộng hơn.

```text
False answer-token activations -> label 1
True answer-token activations  -> label 0
True other-token activations   -> label 0
False other-token activations  -> label 0
```

Ý nghĩa:

- Positive không chỉ là “false sample” mà cụ thể là “answer tokens của false sample”.
- Negative bao gồm true answer và non-answer context.
- Mục tiêu là tìm neuron đặc biệt liên quan đến hallucinated answer tokens, không chỉ toàn bộ sequence hoặc style của false sample.

### 9.6. Train Logistic Regression

```text
model = LogisticRegression(
    penalty=args.penalty,
    C=args.C,
    solver=args.solver,
    max_iter=1000,
    random_state=42,
    verbose=1
)
model.fit(X_train, y_train)
```

Sau khi train:

```text
joblib.dump(model, save_model)
```

Script in:

```text
Identified {np.sum(model.coef_[0] > 0)} potential H-Neurons.
```

Đây là detection rule chính trong implementation.

### 9.7. Evaluation metrics

`run_evaluation()` tính:

```text
accuracy
precision
recall
f1
AUROC
```

Dùng:

```text
preds = model.predict(X)
probs = model.predict_proba(X)[:, 1]
```

Evaluation test hiện load theo `mode="1-vs-1"`, tức test thường là true answer vs false answer.

### 9.8. Định nghĩa H-Neuron trong implementation

Sau training, mỗi feature ứng với một neuron ở một layer.

Classifier coefficient:

```text
weights = model.coef_[0]
```

Nếu:

```text
weights[flat_index] > 0
```

thì feature đó đẩy prediction về class `1`, tức false/hallucinated answer.

Do đó candidate H-Neurons:

```text
H = {flat_index | classifier.coef_[0][flat_index] > 0}
```

Map về layer/neuron:

```text
layer_idx  = flat_index // intermediate_size
neuron_idx = flat_index % intermediate_size
```

---

## 10. Bước 7 — `intervene_model.py`: map và can thiệp H-Neurons

### 10.1. Mục đích

Script này cung cấp utility để:

1. lấy H-Neuron indices từ classifier;
2. scale các neuron tương ứng trong `down_proj` weights của model.

Nó không có CLI trong file hiện tại, mà là helper functions để import/use ở nơi khác.

### 10.2. `get_h_neuron_indices(classifier, config)`

Input:

- `classifier`: sklearn LogisticRegression đã train.
- `config`: model config có `intermediate_size` hoặc `text_config.intermediate_size`.

Logic:

```text
weights = classifier.coef_[0]
inter_size = config.intermediate_size
selected_flat_indices = np.where(weights > 0)[0]
```

Map:

```text
layer_idx = idx // inter_size
neuron_idx = idx % inter_size
```

Output:

```python
{
  layer_idx: [neuron_idx_1, neuron_idx_2, ...]
}
```

### 10.3. `apply_scaling(model, neuron_map, scale_factor)`

Duyệt tất cả modules:

```text
for name, module in model.named_modules():
    if 'down_proj' in name and isinstance(module, torch.nn.Linear):
        ...
```

Lấy layer index bằng cách tìm phần numeric trong module name:

```text
parts = name.split('.')
layer_idx = next(int(p) for p in parts if p.isdigit())
```

Nếu layer đó có H-Neurons:

```text
module.weight.data[:, target_neurons] *= scale_factor
```

Vì `down_proj.weight` shape thường là:

```text
[hidden_size, intermediate_size]
```

cột `target_neurons` tương ứng output vector của từng intermediate neuron.

### 10.4. Ý nghĩa intervention

Nếu `scale_factor < 1`:

- giảm ảnh hưởng của H-Neurons;
- kỳ vọng giảm hallucinated answer behavior.

Nếu `scale_factor = 0`:

- gần như ablate các neuron đó trong `down_proj`.

Nếu `scale_factor > 1`:

- khuếch đại H-Neurons;
- có thể dùng như causal sanity check: nếu hallucination tăng, neuron set có tính nhân quả hơn.

---

## 11. Luồng phát hiện H-Neuron chi tiết theo data/control flow

```text
[1] Collect responses
    input: TriviaQA parquet + model_path
    output: samples.jsonl
    label source: rule judge hoặc LLM judge

        samples.jsonl row:
        qid -> question, responses[], judges[], ground_truth[]

[2] Keep clean labels
    option A: filter_consistent_samples.py
    option B: extract_answer_tokens.py tự skip non-consistent rows
    optional: preselect_balanced_samples.py để giảm chi phí extraction

        condition:
        set(judges) == {"true"} hoặc {"false"}

[3] Extract answer tokens
    input: consistent samples
    process:
        - chọn representative response
        - tokenize response
        - LLM chọn minimal answer span bằng token indices
    output: answer_tokens.jsonl

        qid -> question, response, tokenized_response, answer_tokens, judge

[4] Balance qids
    input: answer_tokens.jsonl
    process:
        - true_ids = qids có judge true
        - false_ids = qids có judge false
        - random sample equal count
    output: train_qids.json/test_qids.json

[5] Extract neuron features
    input:
        - target model
        - answer_tokens.jsonl
        - qid map
    process:
        - build chat message: user question + assistant response
        - forward model
        - hooks collect down_proj input activation and output norm
        - compute CETT-like score
        - locate answer token span in full sequence
        - slice selected token region
        - aggregate over tokens
    output:
        .npy per qid with shape [layers, neurons]

[6] Train classifier
    input:
        - qid map
        - answer_tokens activations
        - optionally all_except_answer_tokens activations
    process:
        - flatten [layers, neurons]
        - label false answer as 1
        - label true answer/other tokens as 0
        - train sparse LogisticRegression
    output:
        classifier.pkl/detector.pkl

[7] Detect H-Neurons
    detection rule:
        flat indices where classifier.coef_[0] > 0

    mapping:
        layer_idx  = flat_index // intermediate_size
        neuron_idx = flat_index % intermediate_size

[8] Optional intervention
    process:
        - scale down_proj.weight[:, neuron_idx]
        - measure behavior change
```

---

## 12. Các assumptions quan trọng

### 12.1. Model architecture có `down_proj`

Các script assume MLP có module name chứa `down_proj`, thường đúng với Llama/Mistral/Qwen-style architectures.

Nếu model dùng tên khác như `c_proj`, `fc2`, `dense_4h_to_h`, cần chỉnh hook logic.

### 12.2. Flatten order phải khớp mapping

`classifier.py` flatten activation `[layers, neurons]` bằng `np.flatten()`.

`intervene_model.py` map ngược bằng:

```text
layer_idx = flat_idx // intermediate_size
neuron_idx = flat_idx % intermediate_size
```

Điều này chỉ đúng nếu:

- activation array shape thật sự là `[num_layers, intermediate_size]`;
- flatten theo C-order;
- `intermediate_size` trong config khớp số neuron trong activation.

### 12.3. Answer-token matching có thể fragile

`extract_answer_tokens.py` lưu `answer_tokens` dưới dạng decoded token strings từ representative response.

`extract_activations.py` lại tìm các token string đó trong full chat-template token sequence.

Rủi ro:

- token string decode khác nhau giữa response-only và chat-template full sequence;
- normalization `▁`/`Ġ` không đối xứng hoàn toàn với `full_tokens`;
- answer xuất hiện nhiều lần trong output;
- answer tokens không contiguous sau khi qua chat template;
- special tokens/EOS/header làm lệch region.

Nếu không match được, sample bị skip cho `answer_tokens` location.

### 12.4. `all_except_answer_tokens` không chỉ nằm trong assistant output

Implementation hiện tại:

```text
seg1 = cett_full[:, :ans_s, :]
seg2 = cett_full[:, ans_e:, :]
```

Nghĩa là lấy cả input prompt, chat headers, tokens trước answer, tokens sau answer, có thể gồm EOS/special tokens tùy tokenizer.

Nếu mục tiêu là “output non-answer tokens”, cần định nghĩa khác. Nhưng tài liệu này mô tả đúng implementation hiện tại.

### 12.5. Nhãn hallucination phụ thuộc judge

Nếu judge sai, classifier học sai. Rule judge đặc biệt có thể noisy với:

- alias không đủ;
- response dài chứa alias nhưng không trả lời đúng;
- paraphrase đúng nhưng không chứa exact alias;
- câu trả lời có nhiều entities.

LLM judge tốt hơn nhưng có cost và bias riêng.

---

## 13. Interpretation: classifier weight nghĩa là gì?

Với Logistic Regression nhị phân:

```text
p(false | x) = sigmoid(w · x + b)
```

Trong đó:

```text
x[flat_idx] = activation/contribution của neuron tại layer/token-region
```

Nếu:

```text
w[flat_idx] > 0
```

thì feature đó làm tăng logit cho class `false`.

Nếu dùng L1 penalty, nhiều weights sẽ bằng 0. Những weights dương còn lại được xem là candidate H-Neurons.

Nhưng cần phân biệt:

- correlation: neuron activation tương quan với false answers;
- causation: neuron thật sự gây hallucination.

`intervene_model.py` được dùng để kiểm tra causal hơn bằng cách scale/ablate các neuron này và đo thay đổi behavior.

---

## 14. Checklist chạy pipeline điển hình

Ví dụ logic, không phải command bắt buộc:

```text
1. collect_responses.py
   -> train_qwen_samples.jsonl

2. filter_consistent_samples.py hoặc preselect_balanced_samples.py
   -> train_qwen_samples_consistent.jsonl
   -> train_qwen_samples_91_true_91_false.jsonl

3. extract_answer_tokens.py
   -> train_answer_tokens_qwen.jsonl

4. sample_balanced_ids.py
   -> train_qwen_balanced_ids.json

5. extract_activations.py với locations:
   - answer_tokens
   - all_except_answer_tokens
   -> activations/answer_tokens/act_<qid>.npy
   -> activations/all_except_answer_tokens/act_<qid>.npy

6. classifier.py train_mode 1-vs-1 hoặc 3-vs-1
   -> classifier.pkl

7. intervene_model.py utilities
   -> neuron_map
   -> apply scaling
```

---

## 15. Những điểm cần verify nếu muốn tin kết quả detection

1. **Số lượng samples thực tế**
   - Bao nhiêu all-true?
   - Bao nhiêu all-false?
   - Bao nhiêu answer-token extraction thành công?
   - Bao nhiêu activation extraction thành công?

2. **Train/test leakage**
   - `train_qids.json` và `test_qids.json` có disjoint không?
   - Có dùng cùng response/activation ở train và test không?

3. **Answer span match rate**
   - Tỷ lệ sample không tìm được answer span trong `extract_activations.py` là bao nhiêu?
   - Có bias class nào bị skip nhiều hơn không?

4. **Classifier sparsity**
   - Bao nhiêu coefficients non-zero?
   - Bao nhiêu positive weights?
   - Positive neurons tập trung ở layer nào?

5. **Metric đầy đủ**
   - Accuracy/Precision/Recall/F1/AUROC trên balanced test.
   - False-only recall không đủ; cần cả false-positive rate.

6. **Causal validation**
   - Ablate/scale H-Neurons có giảm hallucination không?
   - Scale tăng có làm hallucination tăng không?
   - Có ảnh hưởng năng lực trả lời đúng không?

---

## 16. Debug log — Colab vLLM/CUDA runtime mismatch

### 16.1. Symptom

Khi chạy `h_neuron_scripts/collect_responses.py` trên Colab, script fail ngay tại import vLLM:

```text
from vllm import LLM, SamplingParams
ImportError: libcudart.so.13: cannot open shared object file: No such file or directory
```

Traceback đi qua package `vllm` rồi fail khi import CUDA extension:

```text
import vllm._C
ImportError: libcudart.so.13
```

### 16.2. Diagnosis

Đây là lỗi mismatch giữa vLLM wheel và CUDA runtime trong môi trường Colab, không phải lỗi import path của project.

Các khả năng đã cân nhắc:

1. Sai path script hoặc sai working directory.
2. Thiếu dependency Python thông thường.
3. Lỗi import nội bộ trong `h_neuron_scripts`.
4. PyTorch/CUDA không có GPU.
5. Model path hoặc data path sai.
6. vLLM wheel được build/packaged cho CUDA runtime khác môi trường đang chạy.
7. Notebook Colab còn cell cài/reinstall vLLM hoặc CUDA stack không tương thích.

Nguồn lỗi có khả năng cao nhất:

- installed `vllm` wheel đang expect CUDA 13 runtime;
- Colab runtime hiện tại không có `libcudart.so.13`;
- vì `collect_responses.py` import `vllm` ở module import time nên lỗi xảy ra trước khi argparse hoặc logic pipeline chạy.

### 16.3. Fix/workaround đã áp dụng

Workaround an toàn cho Colab/T4 là không dùng vLLM cho bước response collection, mà dùng Hugging Face Transformers-only notebook path.

Đã chạy script:

```text
python new_scripts/disable_vllm_notebook.py
```

Script này update notebook:

```text
new_scripts/H_neurons_1.ipynb
```

Các thay đổi chính trong notebook:

- không install `vllm`;
- không uninstall/reinstall PyTorch/CUDA stack trong Colab;
- đặt một số env var phòng ngừa accidental vLLM import;
- dùng `transformers.AutoModelForCausalLM` và `transformers.AutoTokenizer` để sample response;
- giữ logic judge/output JSONL tương đương bước collection.

### 16.4. Cách chạy sau fix

Trên Colab, dùng notebook đã update:

```text
new_scripts/H_neurons_1.ipynb
```

Chạy notebook từ đầu để cài dependencies và load model bằng Transformers. Không chạy lại command cài vLLM trong notebook Colab nếu mục tiêu là tránh lỗi CUDA runtime mismatch này.

Nếu bắt buộc muốn dùng `collect_responses.py` với vLLM, cần tự align toàn bộ stack vLLM/PyTorch/CUDA sao cho wheel vLLM khớp CUDA runtime có sẵn trong môi trường. Với Colab, hướng này kém ổn định hơn Transformers-only workaround.

---

## 17. Tóm tắt ngắn

`h_neuron_scripts` triển khai một supervised H-Neuron detection pipeline:

```text
consistent false answers
    -> answer-token spans
    -> MLP down_proj contribution features
    -> sparse linear classifier
    -> positive classifier weights
    -> candidate H-Neurons
```

Core detection rule:

```text
H-Neuron = (layer, neuron) such that LogisticRegression coefficient > 0
```

Feature được dùng để train classifier là CETT-like score từ `extract_activations.py`, được aggregate trên answer-token region hoặc các region khác.

Can thiệp model được thực hiện bằng cách scale cột tương ứng trong `down_proj.weight`, tức giảm/tăng output contribution của neuron đã phát hiện.

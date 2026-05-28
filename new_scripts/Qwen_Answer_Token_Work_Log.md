# H-Neuron Implementation Work Log 29/5/2026

This document summarizes the changes, scripts, commands, fixes, and outputs created during the recent workflow for Qwen/TriviaQA response filtering, answer-token extraction, API-key handling, and balanced ID sampling.

## 1. Initial Problem

The original sample file contained more rows than the answer-token output file. The main reason was that `h_neuron_scripts/extract_answer_tokens.py` skips samples that are not judge-consistent and also skips samples where answer-token extraction fails.

Relevant skip logic in `h_neuron_scripts/extract_answer_tokens.py`:

```python
judges = content["judges"]
if len(set(judges)) != 1 or "uncertain" in judges or "error" in judges:
    continue
```

This means rows with mixed labels such as:

```json
"judges": ["false", "true", "false", "false", "true"]
```

are skipped during answer-token extraction.

## 2. Resume Logic in `extract_answer_tokens.py`

The resume mechanism was explained and used.

When `--resume` is enabled, the script:

1. Reads the existing output JSONL file.
2. Collects all already-processed question IDs.
3. Opens the output file in append mode.
4. Skips any input row whose ID already exists in the output file.
5. Reuses or updates the API usage/quota state file.

Relevant logic:

```python
if self.args.resume:
    processed_ids = self.load_processed_ids()
    output_mode = "a"
else:
    processed_ids = set()
    output_mode = "w"
```

```python
if qid in processed_ids:
    continue
```

A resume command was prepared for the user to run with their own API key.

## 3. Tokenization Logic in `extract_answer_tokens.py`

The tokenizer is loaded with Hugging Face Transformers:

```python
self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
```

Tokenization is performed by encoding the response and decoding each token ID back into a token string:

```python
def get_tokenized_list(self, text: str) -> List[str]:
    token_ids = self.tokenizer.encode(text, add_special_tokens=False)
    tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
    return tokens
```

The tokenizer path used in commands was:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

## 4. Fix: Fragile Token-String Matching

### Original issue

The script originally asked the LLM to return token strings and validated them with:

```python
all(t in tokens for t in extracted)
```

This failed often because LLM-returned strings do not always exactly match tokenizer fragments. Tokenizer artifacts such as leading spaces, Unicode spacing, or subword markers can cause semantically correct answers to fail exact string comparison.

### Fix implemented

`h_neuron_scripts/extract_answer_tokens.py` was changed to ask the LLM for integer token indices instead of token strings.

The prompt now includes indexed response tokens and asks for a JSON array of indices:

```python
USER_INPUT_TEMPLATE = """Question: {question}
Response: {response}
Tokenized Response with indices: {indexed_response_tokens}
Please identify the answer span in the tokenized response. Return only a JSON array of integer token indices from the indexed tokenized response. The selected indices must be valid, in ascending order, and should form the minimal answer span with redundant context removed."""
```

The extraction function now builds indexed tokens:

```python
indexed_tokens = list(enumerate(tokens))
prompt = USER_INPUT_TEMPLATE.format(
    question=question,
    response=response,
    indexed_response_tokens=str(indexed_tokens)
)
```

The returned indices are validated:

```python
extracted_indices = json.loads(reply)

if (
    isinstance(extracted_indices, list)
    and all(isinstance(i, int) for i in extracted_indices)
    and extracted_indices == sorted(extracted_indices)
    and all(0 <= i < len(tokens) for i in extracted_indices)
):
    return [tokens[i] for i in extracted_indices]
```

Syntax was validated with:

```cmd
python -m py_compile h_neuron_scripts\extract_answer_tokens.py
```

## 5. `.env` File for API Keys

A `.env` file was created at:

```text
h_neuron_scripts/.env
```

The user later edited it with Gemini API settings and multiple keys.

Important note: `extract_answer_tokens.py` does not automatically load `.env`. The environment variables need to be loaded manually in PowerShell or passed directly as command-line arguments.

PowerShell loader used:

```powershell
Get-Content h_neuron_scripts\.env | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    if ($_ -notmatch '=') { return }
    $name, $value = $_ -split '=', 2
    $name = $name.Trim()
    $value = $value.Trim().Trim('"')
    [Environment]::SetEnvironmentVariable($name, $value, 'Process')
}
```

PowerShell variable syntax:

```powershell
echo $env:BASE_URL
echo $env:LLM_MODEL
echo $env:phucbill_key
```

Two API keys can be passed with `--api_keys`:

```powershell
python h_neuron_scripts\extract_answer_tokens.py --input_path data\small_subset\train_qwen_samples_91_true_91_false.jsonl --output_path data\small_subset\train_answer_tokens_qwen_91_true_91_false.jsonl --usage_path data\small_subset\qwen_91_usage.json --tokenizer_path Qwen/Qwen2.5-1.5B-Instruct --api_keys $env:phucbill_key $env:biha_key --base_url $env:BASE_URL --llm_model $env:LLM_MODEL --resume
```

## 6. `.gitignore` Created

A `.gitignore` file was created to avoid committing secrets and local generated files.

Content added:

```gitignore
# Environment files with secrets
.env
*.env
h_neuron_scripts/.env

# Python cache
__pycache__/
*.py[cod]

# Local outputs
*.usage.json
```

## 7. New Script: Filter Consistent Samples

Created:

```text
h_neuron_scripts/filter_consistent_samples.py
```

Purpose:

- Keep only samples whose `judges` list is all `true` or all `false`.
- Drop samples with mixed true/false judgments.
- Drop invalid rows.
- Print useful counts.

Main logic:

```python
labels = set(normalized)

if labels == {"true"}:
    return "true"
if labels == {"false"}:
    return "false"

return "mixed"
```

Example command:

```cmd
python h_neuron_scripts\filter_consistent_samples.py --input_path data\small_subset\train_qwen_samples.jsonl --output_path data\small_subset\train_qwen_samples_consistent.jsonl
```

Syntax was validated with:

```cmd
python -m py_compile h_neuron_scripts\filter_consistent_samples.py
```

## 8. New Script: Preselect Balanced Samples

Created:

```text
h_neuron_scripts/preselect_balanced_samples.py
```

Purpose:

- Randomly select an equal number of all-true and all-false rows before running expensive answer-token extraction.
- Default is 91 samples per class.
- Uses a random seed for reproducible sampling.

Main command run:

```cmd
python h_neuron_scripts\preselect_balanced_samples.py --input_path data\small_subset\train_qwen_samples.jsonl --output_path data\small_subset\train_qwen_samples_91_true_91_false.jsonl --num_per_class 91 --seed 42
```

Result:

```text
Input rows: 800
Available all-true rows: 91
Available all-false rows: 551
Dropped mixed rows: 158
Dropped invalid rows: 0
Selected all-true rows: 91
Selected all-false rows: 91
Output rows: 182
Saved to: data\small_subset\train_qwen_samples_91_true_91_false.jsonl
```

Output file:

```text
data/small_subset/train_qwen_samples_91_true_91_false.jsonl
```

## 9. API Connection and PowerShell Issues

Several issues were diagnosed:

### `cmd.exe` syntax was used in PowerShell

This failed:

```cmd
for /f "usebackq tokens=1,* delims==" %A in ("h_neuron_scripts\.env") do set %A=%B
```

because the terminal was PowerShell, not `cmd.exe`.

Correct PowerShell environment variable syntax is:

```powershell
$env:VARIABLE_NAME
```

not:

```cmd
%VARIABLE_NAME%
```

### Empty environment variable output

`echo $env:OPENAI_BASE_URL` printed nothing because the `.env` file used `BASE_URL`, not `OPENAI_BASE_URL`.

The correct check for the user's `.env` was:

```powershell
echo $env:BASE_URL
```

### Passing multiple API keys

The script expects multiple keys as separate arguments because `--api_keys` uses `nargs="+"`:

```python
parser.add_argument("--api_keys", nargs="+", default=None, help="Multiple OpenAI-compatible API keys for rotation")
```

Correct form:

```powershell
--api_keys $env:phucbill_key $env:biha_key
```

## 10. Fix: Windows `PermissionError` for Usage File

### Error

The script crashed with:

```text
PermissionError: [WinError 5] Access is denied: 'data\\small_subset\\qwen_91_usage.json.tmp' -> 'data\\small_subset\\qwen_91_usage.json'
```

This happened at:

```python
os.replace(tmp_path, self.usage_path)
```

Likely cause: Windows, VS Code, antivirus, or file sync temporarily locked the usage file or temp file.

### Fix implemented

`h_neuron_scripts/extract_answer_tokens.py` was patched so `_save_usage_state()` falls back to direct writing if atomic replacement fails.

New logic:

```python
def _save_usage_state(self):
    os.makedirs(os.path.dirname(self.usage_path) or ".", exist_ok=True)
    tmp_path = f"{self.usage_path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.usage_state, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.usage_path)
    except PermissionError as e:
        print(f"Warning: could not atomically replace usage file ({e}); writing directly.")
        with open(self.usage_path, "w", encoding="utf-8") as f:
            json.dump(self.usage_state, f, ensure_ascii=False, indent=2)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except PermissionError:
            pass
```

Syntax was validated with:

```cmd
python -m py_compile h_neuron_scripts\extract_answer_tokens.py
```

## 11. Answer-Token Extraction on Preselected Samples

The intended extraction command was:

```powershell
python h_neuron_scripts\extract_answer_tokens.py --input_path data\small_subset\train_qwen_samples_91_true_91_false.jsonl --output_path data\small_subset\train_answer_tokens_qwen_91_true_91_false.jsonl --usage_path data\small_subset\qwen_91_usage.json --tokenizer_path Qwen/Qwen2.5-1.5B-Instruct --api_keys $env:phucbill_key $env:biha_key --base_url $env:BASE_URL --llm_model $env:LLM_MODEL --resume
```

The generated token output used later was:

```text
data/small_subset/train_answer_tokens_qwen_91_true_91_false.jsonl
```

## 12. Balanced ID Sampling

After answer-token extraction, `sample_balanced_ids.py` was run:

```cmd
python h_neuron_scripts\sample_balanced_ids.py --input_path data\small_subset\train_answer_tokens_qwen_91_true_91_false.jsonl --output_path data\small_subset\train_qwen_91_balanced_ids.json --num_samples 91
```

Result:

```text
Reading IDs: 173it [00:00, 194900.51it/s]
Total available - True: 91, False: 82
Warning: Only 82 samples per class available. Sampling maximum possible.
Successfully saved 164 balanced IDs to data\small_subset\train_qwen_91_balanced_ids.json
```

Final balanced output:

```text
data/small_subset/train_qwen_91_balanced_ids.json
```

Final balance:

- 82 true IDs
- 82 false IDs
- 164 total balanced IDs

The reason it is 82 per class instead of 91 per class is that answer-token extraction produced 173 usable rows:

- 91 true
- 82 false

So the false class became the limiting class.

## 13. Files Created

### `h_neuron_scripts/filter_consistent_samples.py`

Filters input JSONL to keep only homogeneous all-true or all-false samples.

### `h_neuron_scripts/preselect_balanced_samples.py`

Randomly preselects an equal number of all-true and all-false samples before token extraction.

### `h_neuron_scripts/.env`

Stores local API configuration and keys. This file is ignored by Git.

### `.gitignore`

Prevents committing `.env`, Python cache files, and usage-state files.

### `data/small_subset/train_qwen_samples_91_true_91_false.jsonl`

Balanced preselected raw sample file containing 91 all-true and 91 all-false samples.

### `data/small_subset/train_qwen_91_balanced_ids.json`

Final balanced ID file containing 82 true IDs and 82 false IDs.

## 14. Files Modified

### `h_neuron_scripts/extract_answer_tokens.py`

Modified in two major ways:

1. LLM answer extraction now returns token indices instead of exact token strings.
2. Usage-state saving now handles Windows `PermissionError` by falling back from atomic replace to direct write.

## 15. Recommended Final Workflow

For a cost-conscious balanced Qwen answer-token dataset:

```powershell
# 1. Load .env into PowerShell process
Get-Content h_neuron_scripts\.env | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    if ($_ -notmatch '=') { return }
    $name, $value = $_ -split '=', 2
    $name = $name.Trim()
    $value = $value.Trim().Trim('"')
    [Environment]::SetEnvironmentVariable($name, $value, 'Process')
}

# 2. Preselect balanced raw samples
python h_neuron_scripts\preselect_balanced_samples.py --input_path data\small_subset\train_qwen_samples.jsonl --output_path data\small_subset\train_qwen_samples_91_true_91_false.jsonl --num_per_class 91 --seed 42

# 3. Extract answer tokens with two API keys
python h_neuron_scripts\extract_answer_tokens.py --input_path data\small_subset\train_qwen_samples_91_true_91_false.jsonl --output_path data\small_subset\train_answer_tokens_qwen_91_true_91_false.jsonl --usage_path data\small_subset\qwen_91_usage.json --tokenizer_path Qwen/Qwen2.5-1.5B-Instruct --api_keys $env:phucbill_key $env:biha_key --base_url $env:BASE_URL --llm_model $env:LLM_MODEL --resume

# 4. Sample balanced IDs after successful extraction
python h_neuron_scripts\sample_balanced_ids.py --input_path data\small_subset\train_answer_tokens_qwen_91_true_91_false.jsonl --output_path data\small_subset\train_qwen_91_balanced_ids.json --num_samples 91
```

## 16. Current Final State

The current usable balanced ID file is:

```text
data/small_subset/train_qwen_91_balanced_ids.json
```

It contains:

```text
82 true IDs + 82 false IDs = 164 total IDs
```

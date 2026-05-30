# Week1_log

## 1. Overview

This log summarizes the first week of work on reproducing and testing the H-Neuron-style hallucination detection workflow using Qwen on a TriviaQA-style subset.

The main goals were:

- collect and filter model responses into true/false samples,
- extract answer tokens from Qwen responses,
- extract FFN/down-projection activations at answer-token locations,
- train or reuse a hallucination detector,
- evaluate the detector on a held-out test set,
- document implementation bugs and fixes.

Main model used:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

## 2. BatchEncoding activation extraction bug

During activation extraction, the script crashed with:

```text
TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not BatchEncoding
```

### Cause

`tokenizer.apply_chat_template()` can return either:

- a tensor directly, or
- a Hugging Face `BatchEncoding` / dictionary-like object containing fields such as `input_ids` and `attention_mask`.

The old code treated the whole output as if it were only token IDs. That caused the model embedding layer to receive a `BatchEncoding` object instead of an integer token tensor.

### Fix

The extraction code now checks whether the tokenizer output is mapping-like. If yes, it passes the full encoding to the model using keyword unpacking. If no, it passes the tensor as `input_ids`.

```python
encoding = tokenizer.apply_chat_template(
    msgs,
    return_tensors="pt",
    add_generation_prompt=False,
)

if hasattr(encoding, "keys"):
    encoding = {
        k: v.to(model.device)
        for k, v in encoding.items()
        if torch.is_tensor(v)
    }
    input_ids = encoding["input_ids"]
    with torch.no_grad():
        model(**encoding)
else:
    input_ids = encoding.to(model.device)
    with torch.no_grad():
        model(input_ids=input_ids)
```

Simple explanation:

```text
Wrong: give the model the whole tokenizer package as token IDs.
Right: if the tokenizer gives a package, unpack it; if it gives only token IDs, pass token IDs directly.
```

## 3. Answer-token extraction workflow

Answer-token extraction was run using `h_neuron_scripts/extract_answer_tokens.py`.

The script tokenizes each response with the target model tokenizer, asks an external LLM to identify which token indices contain the final answer, and saves only usable samples.

Important tokenization helper:

```python
def get_tokenized_list(self, text: str) -> List[str]:
    token_ids = self.tokenizer.encode(text, add_special_tokens=False)
    tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

    return tokens
```

Meaning:

```text
response text -> tokenizer IDs -> decode each ID separately -> model-visible token strings
```

This is important because the answer-token locations must match the exact tokens used by Qwen, not just human-readable words.

### API key handling

The script supports one key or multiple keys:

```text
--api_key ONE_KEY
--api_keys KEY_1 KEY_2 KEY_3
```

For PowerShell environment variables, the correct multi-key form is:

```powershell
--api_keys $env:baby_key $env:phucbill_key $env:vkb_work_key
```

### Windows usage-file issue

A Windows `PermissionError` occurred when replacing the usage JSON file:

```text
PermissionError: [WinError 5] Access is denied
```

The fix was to fall back to direct writing if atomic replacement fails.

## 4. Consistent sample filtering and training subset

The workflow used a consistent-response file:

```text
data/small_subset/train_qwen_samples_consistent.jsonl
```

A balanced subset was selected:

```text
data/small_subset/train_qwen_samples_91_true_91_false.jsonl
```

Initial class balance:

```text
91 true
91 false
182 total
```

After answer-token extraction, usable rows became:

```text
91 true
82 false
173 total
```

A final balanced ID file was produced:

```text
data/small_subset/train_qwen_91_balanced_ids.json
```

Final balanced training IDs:

```text
82 true
82 false
164 total
```

## 5. Test set creation from remaining data

The test set was created from the remaining rows in:

```text
data/small_subset/train_qwen_samples_consistent.jsonl
```

while excluding all qids already used in:

```text
data/small_subset/train_qwen_samples_91_true_91_false.jsonl
```

The helper script retained for this is:

```text
new_scripts/create_test_set_from_remaining.py
```

Main behavior:

- read training qids,
- skip any qid already used for training,
- group remaining samples by label,
- try to sample up to 100 true and 100 false,
- fill remaining slots if one class is missing,
- save the held-out JSONL file.

Output file:

```text
data/small_subset/test_qwen_samples.jsonl
```

Result:

```text
Train qids excluded: 182
Remaining true: 0
Remaining false: 460
Saved total: 200
Saved true: 0
Saved false: 200
```

Important limitation:

```text
The held-out test set is false-only because all true samples had already been consumed by the training subset.
```

## 6. Test answer-token extraction and qid creation

Answer-token extraction was then run on:

```text
data/small_subset/test_qwen_samples.jsonl
```

Output:

```text
data/small_subset/test_answer_tokens_qwen.jsonl
```

The answer-token file contained:

```text
169 usable rows
0 true
169 false
```

A helper script was created and retained:

```text
new_scripts/create_test_qwen_ids.py
```

Purpose:

- read `test_answer_tokens_qwen.jsonl`,
- extract each qid,
- detect label from either `judge` or `judges`,
- write ids into the format expected by activation/classifier scripts.

Output:

```text
data/small_subset/test_qwen_ids.json
```

Final id counts:

```text
True ids: 0
False ids: 169
Skipped rows: 0
```

The final JSON structure is:

```json
{
  "t": [],
  "f": ["tc_448", "tc_964"]
}
```

## 7. Activation extraction

Activation extraction used:

```text
h_neuron_scripts/extract_activations.py
```

The script reads answer-token samples and qid files, runs the model forward, captures down-projection FFN activations, selects token regions, pools activations over tokens, and saves one activation matrix per qid.

Important flow:

```text
sample -> qid check -> tokenizer chat template -> model forward -> hooks capture activations -> select answer-token region -> mean/max over tokens -> save .npy
```

Core tensor shapes:

```text
[layers, selected_tokens, neurons]
```

means:

- for each layer,
- for each selected token,
- record each FFN neuron activation.

After pooling over selected tokens:

```text
[layers, neurons]
```

This fixed-size matrix is saved for classifier training/evaluation.

Test activation output root:

```text
data/small_subset/test_activations/
```

Important activation folders used later:

```text
data/small_subset/test_activations/answer_tokens/
data/small_subset/test_activations/output/
```

## 8. How activations support classifier training

Each saved activation matrix has shape:

```text
[layers, neurons]
```

The classifier flattens it into one long feature vector:

```text
[layer 0 neuron 0, layer 0 neuron 1, ..., layer 1 neuron 0, ...]
```

Training labels:

```text
false answer-token activations -> label 1 hallucination
true answer-token activations  -> label 0 non-hallucination
```

The classifier learns which layer/neuron activation patterns separate hallucinated answers from correct answers.

## 9. H-Neuron location detection

After training, the classifier weights indicate which activation features are associated with hallucination.

Positive weights are treated as hallucination-associated neurons.

Mapping from flat classifier feature index back to H-Neuron location:

```python
layer_idx = idx // intermediate_size
neuron_idx = idx % intermediate_size
```

Arrow summary:

```text
activation matrix [layers, neurons]
-> flatten into classifier features
-> train classifier true vs false
-> inspect positive classifier weights
-> map flat indices back to (layer, neuron)
-> these mapped neurons are candidate H-Neurons
```

## 10. False-only detector evaluation

Because the test set had only false samples, normal AUROC evaluation was not valid.

Reason:

```text
AUROC requires both positive and negative classes.
False-only data has only one class.
```

A helper script was created and retained:

```text
new_scripts/evaluate_false_only_detector.py
```

Purpose:

- load a trained classifier,
- load false-only test ids,
- load answer-token activations,
- predict hallucination/non-hallucination,
- report false-sample recall and probability statistics instead of AUROC.

Input defaults:

```text
classifier: data/small_subset/classifier/classifier.pkl
test ids: data/small_subset/test_qwen_ids.json
test activations: data/small_subset/test_activations/answer_tokens
output: data/small_subset/false_only_detector_results.json
```

Evaluation result:

```text
Input false ids: 169
Input true ids: 0
Tested false samples: 167
Missing activation files: 2
Predicted hallucination: 61
Predicted non-hallucination: 106
False-sample recall: 0.3653
Average hallucination probability: 0.4744
```

Interpretation:

```text
The detector caught 61 out of 167 available false samples.
False-only recall = 36.53%.
```

This is not a complete detector-quality score because there are no true samples in this test set.

## 11. Checkpoint result: First attempt, 29/5/2026

At this checkpoint, the pipeline reached:

```text
held-out false-only test set
-> answer-token extraction
-> qid JSON creation
-> activation extraction
-> false-only detector evaluation
```

Main checkpoint result:

```text
61 / 167 false samples detected as hallucination
false-only recall = 36.53%
average hallucination probability = 0.4744
```

Important warning encountered:

```text
InconsistentVersionWarning: Trying to unpickle estimator LogisticRegression from version 1.6.1 when using version 1.5.2.
```

Suggested environment fix:

```powershell
python -m pip install scikit-learn==1.6.1
```

## 12. Retained helper scripts

These Python helper scripts are retained.

### `new_scripts/create_test_set_from_remaining.py`

Creates a held-out test JSONL file from remaining consistent Qwen samples while excluding qids already used in the training subset.

Output:

```text
data/small_subset/test_qwen_samples.jsonl
```

### `new_scripts/create_test_qwen_ids.py`

Creates the test qid JSON file expected by activation/classifier scripts.

Input:

```text
data/small_subset/test_answer_tokens_qwen.jsonl
```

Output:

```text
data/small_subset/test_qwen_ids.json
```

It supports both label formats:

```text
judge: "false"
judges: ["false", "false"]
```

### `new_scripts/evaluate_false_only_detector.py`

Evaluates a trained hallucination classifier on a false-only test set.

It reports:

- tested false samples,
- missing activation files,
- predicted hallucination count,
- predicted non-hallucination count,
- false-sample recall,
- average/min/max hallucination probability.

## 13. Key limitations

Current limitations:

1. The test set is false-only.
2. AUROC, accuracy, precision, and balanced evaluation cannot be trusted without true samples.
3. The detector result only measures how many known-false samples are flagged.
4. Two test qids did not have matching answer-token activation files during false-only evaluation.
5. The classifier pickle was produced with a different scikit-learn version than the current environment.

## 14. Recommended next steps

Recommended next actions:

```text
Create or reserve a balanced true/false test set
-> extract answer tokens
-> create test ids
-> extract activations
-> run full classifier evaluation with AUROC
-> inspect positive classifier weights as H-Neurons
-> optionally run intervention experiments
```

Specific next improvements:

- reserve true samples before consuming all of them for training,
- build a balanced test set with both `t` and `f` ids,
- evaluate AUROC and threshold behavior,
- export H-Neuron layer/neuron indices to CSV or JSON,
- compare answer-token activations against output/all-token regions,
- check the `all_except_answer_tokens` region logic if that location is used later.

# vLLM / Colab CUDA Runtime Mismatch Debug Log

## Summary

When running `h_neuron_scripts/collect_responses.py` in Google Colab, the script fails before any project pipeline logic runs because `vllm` cannot load the CUDA runtime library it was built against.

Observed mismatch:

```text
Colab runtime CUDA: 12.8
vLLM binary requires: CUDA 13
Missing shared library: libcudart.so.13
```

This confirms the issue is a binary compatibility mismatch between the installed vLLM wheel and the Colab CUDA runtime, not a bug in `collect_responses.py` or a project import-path problem.

---

## Error

The failing script imports vLLM at module import time:

```python
from vllm import LLM, SamplingParams
```

The observed traceback ends with:

```text
ImportError: libcudart.so.13: cannot open shared object file: No such file or directory
```

The important part of the traceback is that the error occurs inside the installed `vllm` package while importing its CUDA extension:

```text
import vllm._C
ImportError: libcudart.so.13
```

Because this happens during import, none of the script's argument parsing, dataset loading, model path handling, or sampling logic has started yet.

---

## Diagnosis

The confirmed root cause is:

```text
Installed vLLM wheel expects CUDA runtime 13.
Current Colab runtime provides CUDA runtime 12.8.
Therefore libcudart.so.13 is unavailable.
```

This is not caused by:

- wrong repository path;
- wrong script path;
- missing `h_neuron_scripts` package;
- bad model path;
- bad data path;
- malformed command-line arguments;
- TriviaQA dataset loading;
- OpenAI judge configuration.

The failure happens earlier than all of those steps.

---

## Verification commands used/recommended

### 1. Check PyTorch CUDA runtime

```python
import torch

print('torch:', torch.__version__)
print('torch CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
```

Expected evidence in this case:

```text
torch CUDA build: 12.8
```

### 2. Check system CUDA / driver info

```python
!nvidia-smi
!nvcc --version || true
```

`nvidia-smi` may show driver-supported CUDA capability, while `nvcc` shows the installed toolkit if available. The key point is that the runtime is not CUDA 13.

### 3. Check available CUDA runtime libraries

```python
!find /usr/local -name 'libcudart.so*' 2>/dev/null
!find /usr/lib -name 'libcudart.so*' 2>/dev/null
```

Mismatch evidence:

```text
libcudart.so.12
libcudart.so.12.x.x
```

but no:

```text
libcudart.so.13
```

### 4. Check vLLM package version without importing `LLM`

```python
import importlib.metadata as md

print('vllm:', md.version('vllm'))
```

Avoid this during verification if the environment is already known broken:

```python
from vllm import LLM, SamplingParams
```

because it triggers the CUDA extension import and reproduces the failure.

### 5. Inspect vLLM shared-library dependencies

```python
import importlib.util
import pathlib
import subprocess

spec = importlib.util.find_spec('vllm')
vllm_dir = pathlib.Path(spec.origin).parent

for so in vllm_dir.rglob('*.so'):
    print('\n###', so)
    subprocess.run(['ldd', str(so)], check=False)
```

Strongest direct evidence:

```text
libcudart.so.13 => not found
```

This proves that the installed vLLM binary extension is linked against CUDA runtime 13, while the Colab runtime cannot provide it.

---

## Fix / Workaround chosen

For Colab, especially T4-style notebook workflows, the chosen workaround is to avoid vLLM for response collection and use the Transformers-only notebook path.

The project includes:

```text
new_scripts/disable_vllm_notebook.py
```

Running it updates:

```text
new_scripts/H_neurons_1.ipynb
```

The updated notebook:

- does not install `vllm`;
- does not reinstall or downgrade/upgrade the Colab PyTorch/CUDA stack;
- uses `transformers.AutoModelForCausalLM` and `transformers.AutoTokenizer` for sampling;
- preserves the response collection and judge/output JSONL workflow as much as possible.

Command used locally to apply the notebook workaround:

```text
python new_scripts/disable_vllm_notebook.py
```

After response collection, validate the output JSONL contract before running downstream steps:

```text
python h_neuron_scripts/validate_response_jsonl.py --input_path data/small_subset/test_qwen_samples.jsonl --max_rows 20
```

The validator checks that each JSONL row has exactly one top-level qid and that each sample contains `question`, `responses`, `judges`, and `ground_truth` fields with compatible lengths and judge labels.

---

## Recommended Colab action

Use the updated notebook:

```text
new_scripts/H_neurons_1.ipynb
```

Run it from the beginning in Colab. Do not run cells that install or reinstall vLLM unless the entire vLLM/PyTorch/CUDA stack is intentionally aligned.

Recommended fresh-runtime validation record:

```python
import torch

print('torch:', torch.__version__)
print('torch CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
```

Then run response collection with the Transformers backend and confirm no `ImportError: libcudart.so.13` occurs.

---

## Alternative if vLLM is required

If `h_neuron_scripts/collect_responses.py` must be run with vLLM, then one of these must be true:

1. Install a vLLM wheel compatible with the current CUDA 12.8 / PyTorch stack.
2. Use a runtime/container that provides CUDA 13 runtime libraries.
3. Build vLLM from source against the runtime actually available in the environment.

Do not fix this by symlinking:

```text
libcudart.so.12 -> libcudart.so.13
```

That can bypass the loader error but may cause undefined behavior, kernel crashes, or silent incorrect execution because CUDA runtime ABIs are not guaranteed to be interchangeable across major versions.

---

## Local repository validation

Local inspection confirmed that `new_scripts/disable_vllm_notebook.py` updates the Colab notebook to:

- install Transformers-related dependencies without installing vLLM;
- skip vLLM/CUDA reinstall steps;
- use `transformers.AutoModelForCausalLM` and `transformers.AutoTokenizer` for sampling;
- write one-qid-per-line JSONL records with `question`, `responses`, `judges`, and `ground_truth` fields.

A reusable smoke validator was added:

```text
h_neuron_scripts/validate_response_jsonl.py
```

Use it on Colab output before downstream processing:

```text
python h_neuron_scripts/validate_response_jsonl.py --input_path /content/H-Neuron-Implementation/data/small_subset/test_qwen_samples.jsonl --max_rows 20
```

Fresh Colab runtime validation is still environment-dependent and should record the observed `torch.__version__`, `torch.version.cuda`, GPU name, and whether response collection completed without `libcudart.so.13`.

---

## Final conclusion

The error is confirmed as:

```text
vLLM wheel/CUDA runtime mismatch
```

with:

```text
Colab CUDA runtime: 12.8
vLLM required runtime: 13
Missing library: libcudart.so.13
```

Use the Transformers-only notebook workaround for this project on Colab.

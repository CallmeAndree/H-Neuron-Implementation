# AGENTS.md

This file provides guidance to agents when working with code in this repository.

- Scripts live in `h_neuron_scripts/`, but README examples still use `scripts/`; run commands with `python h_neuron_scripts/<script>.py ...` from repo root.
- Install is `pip install -r requirements.txt`; there is no test/lint config or package metadata, so use targeted smoke commands such as `python h_neuron_scripts/filter_consistent_samples.py --input_path <jsonl> --output_path <jsonl>`.
- `collect_responses.py` imports vLLM at module import time; `ImportError: libcudart.so.13` means the installed vLLM wheel expects CUDA 13 runtime. Fix by matching vLLM/PyTorch/CUDA versions, or use the Transformers-only notebook path documented in `new_scripts/disable_vllm_notebook.py`.
- Pipeline records are JSONL with exactly one top-level qid per line; downstream scripts assume this shape and silently skip malformed/mixed rows.
- `extract_answer_tokens.py` only processes all-true/all-false rows, picks the most frequent response, and resumes from existing output rows when `--resume` is set.
- `extract_activations.py` hooks every module name containing `down_proj`; models using `c_proj`, `fc2`, or other MLP names need hook changes before CETT extraction.
- Activation files are saved as `<output_root>/<location>/act_<qid>.npy` with shape `[layers, neurons]`; `classifier.py` flattens them in NumPy C-order.
- H-Neuron mapping relies on `flat_idx // intermediate_size` and `flat_idx % intermediate_size`; keep activation shape `[num_layers, intermediate_size]` aligned with model config.
- `all_except_answer_tokens` currently means all tokens outside the answer span in the full chat sequence, not only assistant non-answer tokens.
- Intervention utilities scale `down_proj.weight[:, target_neurons]` for positive logistic-regression weights; `intervene_model.py` is import-only and has no CLI.

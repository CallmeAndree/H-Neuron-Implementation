# Qwen2.5-7B-Instruct Quantization Support

## Summary

This note documents the changes made to support running a quantized
Qwen2.5-7B-Instruct model in `notebook/kaggle_latest/h-neurons-1.ipynb`, so the
model fits and runs efficiently on a single T4 GPU (16 GB) without OOM-ing.

In fp16, Qwen2.5-7B-Instruct weights are ~14 GB, which leaves very little
headroom for KV cache and activations on a single T4. Quantization reduces the
weight footprint to ~5-6 GB (4-bit), leaving much more room for generation.

## Files changed

- `h_neuron_scripts/collect_responses.py`
- `requirements.txt`
- `notebook/kaggle_latest/h-neurons-1.ipynb`

## `collect_responses.py` changes

Added a new `--quantization` CLI argument with choices `none`, `awq`, `gptq`,
`bitsandbytes` (default `none`, preserving prior behavior).

```python
parser.add_argument(
    "--quantization",
    type=str,
    choices=["none", "awq", "gptq", "bitsandbytes"],
    default="none",
    help=(
        "Quantization method. Use 'awq' or 'gptq' when --model_path points to a "
        "pre-quantized checkpoint (e.g. Qwen/Qwen2.5-7B-Instruct-AWQ) so vLLM loads "
        "it with the matching kernels. Use 'bitsandbytes' to 4-bit quantize a "
        "full-precision checkpoint on the fly with the Transformers fallback backend."
    ),
)
```

### vLLM backend (`_init_vllm_backend`)

- `awq` / `gptq`: passes `quantization="awq"` or `quantization="gptq"` to
  `vllm.LLM(...)`. This is the correct path when `--model_path` already points
  to a pre-quantized checkpoint (e.g. `Qwen/Qwen2.5-7B-Instruct-AWQ`), since
  vLLM loads the checkpoint directly with the matching quantized kernels.
- `bitsandbytes`: passes `quantization="bitsandbytes"` and
  `load_format="bitsandbytes"` so vLLM quantizes a full-precision checkpoint
  to 4-bit on load.

### Transformers fallback backend (`_init_transformers_backend`)

- `bitsandbytes`: builds a `transformers.BitsAndBytesConfig` with
  `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`,
  `bnb_4bit_compute_dtype=torch.float16`, `bnb_4bit_use_double_quant=True`, and
  passes it as `quantization_config` to `AutoModelForCausalLM.from_pretrained`.
  Requires CUDA; raises `RuntimeError` otherwise.
- `awq` / `gptq` / `none`: no special handling needed. Pre-quantized AWQ/GPTQ
  checkpoints carry their own `quantization_config` in `config.json`, and
  Transformers auto-detects the right kernels. `dtype=dtype` is passed as
  before for the unquantized path.

## `requirements.txt` changes

Added the two packages needed for the new quantization paths:

```text
bitsandbytes
autoawq
```

## Notebook changes (`h-neurons-1.ipynb`)

The `MODEL_ID` cell was updated to default to a pre-quantized AWQ checkpoint
instead of the full-precision model, and a `QUANTIZATION` variable was added:

```python
# Fresh Qwen2.5-7B-Instruct rerun. In fp16 the weights are ~14 GB, so a single T4 (16 GB) will OOM.
# We use a pre-quantized AWQ checkpoint (4-bit) so the model fits comfortably on one T4 (~5-6 GB)
# and vLLM can serve it with AWQ kernels directly (no on-the-fly quantization needed).
# If you prefer to quantize the full-precision checkpoint yourself instead, keep
# MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct' and pass --quantization bitsandbytes below.
MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct-AWQ'
QUANTIZATION = 'awq'  # one of: none, awq, gptq, bitsandbytes
```

The `collect_responses.py` invocation cell now forwards `QUANTIZATION`:

```python
!python h_neuron_scripts/collect_responses.py \
    --model_path {MODEL_ID} \
    --data_path /kaggle/input/datasets/vkb0205/unprocessed-h-neurons/train_split_2.parquet \
    --output_path /kaggle/working/train_small_qwen7b_2.jsonl \
    --sample_num 10 \
    --max_samples 5000 \
    --judge_type llm \
    --api_key {api_key} \
    --base_url {BASE_URL_2} \
    --judge_model {model} \
    --gpu_util 0.7 \
    --quantization {QUANTIZATION} \
    --resume
```

## Why AWQ by default

- No on-the-fly quantization step is needed at load time (faster startup, no
  extra CPU/GPU conversion pass).
- vLLM has native AWQ kernels, so generation throughput stays close to the
  full-precision model, unlike naive bitsandbytes quantization which can be
  slower for batched generation.
- Weight footprint drops to ~5-6 GB, leaving substantial headroom on a single
  T4 (16 GB) for KV cache and activations across longer sequences.

## Alternative: on-the-fly bitsandbytes quantization

If a pre-quantized checkpoint is not desired (e.g. to quantize the exact same
full-precision weights being used elsewhere in the project), set:

```python
MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'
QUANTIZATION = 'bitsandbytes'
```

This quantizes the full-precision checkpoint to 4-bit NF4 at load time, on
either the vLLM or Transformers fallback backend.

## Verification performed

- `python3 -c "import json; json.load(open('h-neurons-1.ipynb'))"` confirmed
  the notebook JSON remains valid after editing.
- `python3 -m py_compile h_neuron_scripts/collect_responses.py` confirmed the
  script has no syntax errors.
- End-to-end execution was not performed, since that requires a GPU runtime
  (Kaggle/Colab T4) and network access to download the AWQ checkpoint weights.

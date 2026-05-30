import json
from pathlib import Path

NOTEBOOK_PATH = Path('new_scripts/H_neurons_1.ipynb')
nb = json.loads(NOTEBOOK_PATH.read_text(encoding='utf-8'))


def set_source(cell_idx, source):
    nb['cells'][cell_idx]['source'] = [line + '\n' for line in source.splitlines()]
    if source and not source.endswith('\n'):
        nb['cells'][cell_idx]['source'][-1] = nb['cells'][cell_idx]['source'][-1].rstrip('\n')
    nb['cells'][cell_idx]['outputs'] = []
    nb['cells'][cell_idx]['execution_count'] = None

set_source(5, """import subprocess
import sys

# Colab/Kaggle setup without vLLM. Keep the default Colab PyTorch stack to avoid CUDA/vLLM kernel issues on T4.
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q', '-U',
    'transformers', 'accelerate', 'datasets', 'openai', 'scikit-learn', 'joblib', 'tqdm'
], check=True)

print('Installed dependencies without vLLM.')""")

set_source(8, """# vLLM is intentionally disabled in this notebook.
# Do NOT uninstall/reinstall torch or install vLLM here; Colab T4 often hits vLLM/FlashInfer CUDA kernel errors.
print('Skipping vLLM and CUDA reinstall steps.')""")

set_source(11, """import os

# Make accidental vLLM imports use safer settings, but this notebook no longer calls vLLM.
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_ATTENTION_BACKEND'] = 'TORCH_SDPA'
print('vLLM disabled for response collection; using Hugging Face Transformers instead.')""")

set_source(12, r'''import json
import os
import re
import string
import time
from pathlib import Path

import torch
from datasets import load_dataset
from openai import OpenAI
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def handle_punc(text):
        exclude = set(string.punctuation + '‘’´`')
        return ''.join(ch if ch not in exclude else ' ' for ch in text)
    if not s:
        return ''
    return white_space_fix(remove_articles(handle_punc(str(s).lower().replace('_', ' ')))).strip()


def load_existing_qids(path):
    if not os.path.exists(path):
        return set()
    qids = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                qids.update(json.loads(line).keys())
            except Exception:
                pass
    return qids


def rule_judge(response, norm_gts):
    norm_res = normalize_answer(response)
    return 'true' if any(gt and gt in norm_res for gt in norm_gts) else 'false'


judge_client = OpenAI(api_key=api_key, base_url=BASE_URL_2)

def llm_judge(question, response, answer_list):
    prompt = (
        f'Question: {question}\n'
        f'Response: {response}\n'
        f'Correct Answers: {answer_list}\n'
        "Please judge whether the response is correct or not. "
        "Return 't' if the response is correct, and 'f' if the response is incorrect. "
        "Don't add any additional information."
    )
    for attempt in range(5):
        try:
            completion = judge_client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0,
            )
            res = completion.choices[0].message.content.strip().lower()
            if 't' in res:
                return 'true'
            if 'f' in res:
                return 'false'
            print(f'Invalid judge response: {res}; retrying')
        except Exception as e:
            print(f'Judge API failed, attempt {attempt + 1}/5: {e}')
            time.sleep(2)
    return 'error'


DATA_PATH = '/content/H-Neuron-Implementation/data/TriviaQA/rc.nocontext/train-00000-of-00001.parquet'
OUTPUT_PATH = '/content/H-Neuron-Implementation/data/small_subset/test_qwen_samples.jsonl'
SAMPLE_NUM = 5
MAX_SAMPLES = 200
MAX_NEW_TOKENS = 50

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

print('Loading tokenizer/model with Transformers, not vLLM...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model_lm = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
)
model_lm.eval()

terminators = []
if tokenizer.eos_token_id is not None:
    terminators.append(tokenizer.eos_token_id)
im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
if isinstance(im_end_id, int) and im_end_id >= 0:
    terminators.append(im_end_id)
terminators = list(dict.fromkeys(terminators)) or None

dataset = load_dataset('parquet', data_files=DATA_PATH, split='train')
if MAX_SAMPLES:
    dataset = dataset.select(range(MAX_SAMPLES))
processed_qids = load_existing_qids(OUTPUT_PATH)

all_correct_count = 0
all_incorrect_count = 0

with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
    for item in tqdm(dataset, desc='Sampling with Transformers + LLM judge'):
        qid = str(item.get('question_id', ''))
        if qid in processed_qids:
            continue

        question = item.get('question', '')
        if not question or 'answer' not in item:
            continue

        raw_aliases = []
        for col in ['aliases', 'normalized_aliases']:
            val = item['answer'].get(col)
            if val:
                raw_aliases.extend(val if isinstance(val, list) else [str(val)])
        norm_gts = [normalize_answer(a) for a in set(raw_aliases) if a]
        if not norm_gts:
            continue

        suffix = 'Respond with the answer only, without any explanation.'
        prompt_messages = [{'role': 'user', 'content': f'{question.strip()} {suffix}'}]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors='pt').to(model_lm.device)
        prompt_len = inputs.input_ids.shape[-1]

        responses = []
        judges = []
        judge_cache = {}

        for _ in range(SAMPLE_NUM):
            with torch.inference_mode():
                output_ids = model_lm.generate(
                    **inputs,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    top_k=50,
                    max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=terminators,
                    pad_token_id=tokenizer.eos_token_id,
                )
            ans = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
            responses.append(ans)

            uncertain_terms = ["don't know", 'cannot', 'not provided', 'no information']
            if any(term in ans.lower() for term in uncertain_terms):
                judges.append('uncertain')
                continue

            if ans not in judge_cache:
                judge_cache[ans] = llm_judge(question, ans, raw_aliases)
            judges.append(judge_cache[ans])

        true_count = judges.count('true')
        if true_count == SAMPLE_NUM:
            all_correct_count += 1
        elif true_count == 0:
            all_incorrect_count += 1

        result = {
            qid: {
                'question': f'{question.strip()} {suffix}',
                'responses': responses,
                'judges': judges,
                'ground_truth': list(set(raw_aliases)),
            }
        }
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
        f.flush()
        processed_qids.add(qid)

        if len(processed_qids) % 10 == 0:
            tqdm.write(f'Stats -> All-Correct: {all_correct_count}, All-Incorrect: {all_incorrect_count}')

print('Saved responses to', OUTPUT_PATH)''')

NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Updated {NOTEBOOK_PATH}')

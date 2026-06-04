import os
import json
import re
import string
import argparse
from typing import List, Set, Dict, Mapping

import torch
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI  # 用于 LLM Judge

def parse_args():
    parser = argparse.ArgumentParser(description="Consistency Filtering with Rule or LLM Judge.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model for sampling")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the TriviaQA parquet file")
    parser.add_argument("--output_path", type=str, default="data/consistency_samples.jsonl", help="Output path")
    
    parser.add_argument("--sample_num", type=int, default=10, help="Samples per question")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of questions to process")
    parser.add_argument("--backend", type=str, choices=["auto", "vllm", "transformers"], default="auto", help="Sampling backend. 'auto' tries vLLM first, then falls back to Transformers on import/runtime CUDA mismatch errors.")
    parser.add_argument("--gpu_util", type=float, default=0.7, help="vLLM GPU memory utilization")
    parser.add_argument("--tp_size", type=int, default=None, help="Tensor parallel size")

    parser.add_argument("--judge_type", type=str, choices=["rule", "llm"], default="rule", help="How to judge correctness")
    parser.add_argument("--api_key", type=str, default=None, help="Single API key for LLM Judge")
    parser.add_argument("--api_keys", nargs="+", default=None, help="Multiple API keys for LLM Judge rotation")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Model name for LLM Judge")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output_path by skipping already processed question IDs")
    
    return parser.parse_args()

# ==========================================
# Utilities
# ==========================================

def normalize_answer(s: str) -> str:
    """Standardize answer strings for Rule Judge."""
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def handle_punc(text):
        exclude = set(string.punctuation + "‘’´`")
        return ''.join(ch if ch not in exclude else ' ' for ch in text)
    if not s: return ""
    return white_space_fix(remove_articles(handle_punc(str(s).lower().replace('_', ' ')))).strip()

def load_existing_qids(path: str) -> Set[str]:
    if not os.path.exists(path): return set()
    qids = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                qids.update(data.keys())
            except: continue
    return qids

# ==========================================
# Consistency Sampler with Multi-Judge Support
# ==========================================

class ConsistencySampler:
    def __init__(self, args):
        self.args = args
        
        # 1. Init Sampling LLM. Keep vLLM import lazy so CUDA wheel/runtime
        # mismatches (for example libcudart.so.13 on CUDA 12.x Colab) do not
        # fail before the Transformers fallback can run.
        self.backend = None
        self.sampling_llm = None
        self.sampling_params = None
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_sampling_backend()

        # 2. Init Judge Client (if needed)
        self.judge_client = None
        if args.judge_type == "llm":
            self.api_keys = self._load_api_keys()
            self.current_key_idx = 0
            self.judge_client = self._make_client(self.current_key_idx)

    def _is_cuda_mismatch_error(self, error: Exception) -> bool:
        text = str(error).lower()
        return any(term in text for term in ["libcudart.so.13", "cuda", "cudart", "undefined symbol"])

    def _log_vllm_fallback(self, error: Exception):
        print("vLLM backend failed to initialize; falling back to Transformers backend.")
        print(f"vLLM error: {error}")
        if self._is_cuda_mismatch_error(error):
            print(
                "This looks like a vLLM wheel/CUDA runtime mismatch, not a project path, "
                "dataset, or model argument issue. Do not symlink libcudart.so.12 to "
                "libcudart.so.13; use --backend transformers or install a vLLM wheel that "
                "matches the runtime CUDA/PyTorch stack."
            )

    def _init_sampling_backend(self):
        if self.args.backend in ("auto", "vllm"):
            try:
                self._init_vllm_backend()
                return
            except Exception as e:
                if self.args.backend == "vllm":
                    raise RuntimeError(
                        "Failed to initialize vLLM backend. If this is a CUDA runtime "
                        "mismatch such as missing libcudart.so.13, rerun with "
                        "--backend transformers or align vLLM/PyTorch/CUDA versions."
                    ) from e
                self._log_vllm_fallback(e)

        self._init_transformers_backend()

    def _init_vllm_backend(self):
        from vllm import LLM, SamplingParams

        self.tp_size = self.args.tp_size or max(1, torch.cuda.device_count())
        self.sampling_llm = LLM(
            model=self.args.model_path,
            tensor_parallel_size=self.tp_size,
            gpu_memory_utilization=self.args.gpu_util,
            trust_remote_code=True
        )
        self.sampling_params = SamplingParams(
            temperature=1.0, top_p=0.9, top_k=50, max_tokens=50
        )
        self.backend = "vllm"
        print("Using vLLM sampling backend.")

    def _init_transformers_backend(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        if not torch.cuda.is_available():
            self.model.to(self.device)
        self.model.eval()
        self.backend = "transformers"
        print("Using Transformers sampling backend.")

    def _sample_answer(self, messages: List[Dict[str, str]]) -> str:
        if self.backend == "vllm":
            outputs = self.sampling_llm.chat(messages, self.sampling_params, use_tqdm=False)
            return outputs[0].outputs[0].text.strip()

        if hasattr(self.tokenizer, "apply_chat_template"):
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
            encoded = self.tokenizer(prompt, return_tensors="pt")

        if torch.is_tensor(encoded):
            input_ids = encoded
            attention_mask = torch.ones_like(input_ids)
        elif isinstance(encoded, Mapping) or hasattr(encoded, "keys"):
            if "input_ids" not in encoded:
                raise TypeError(
                    f"Tokenizer output is missing input_ids; got keys: {list(encoded.keys())}"
                )
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
        else:
            raise TypeError(
                f"Unsupported tokenizer output type for Transformers sampling: {type(encoded).__name__}"
            )

        if not torch.is_tensor(input_ids):
            raise TypeError(
                f"Tokenizer input_ids must be a Tensor, got {type(input_ids).__name__}"
            )
        if not torch.is_tensor(attention_mask):
            raise TypeError(
                f"Tokenizer attention_mask must be a Tensor, got {type(attention_mask).__name__}"
            )

        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                top_k=50,
                max_new_tokens=50,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _load_api_keys(self) -> List[str]:
        keys = []
        if self.args.api_keys:
            keys.extend(self.args.api_keys)
        if self.args.api_key:
            keys.append(self.args.api_key)

        keys = [key.strip() for key in keys if key and key.strip()]
        if not keys:
            raise ValueError("Provide at least one API key with --api_key or --api_keys for LLM Judge.")
        return list(dict.fromkeys(keys))

    def _make_client(self, key_idx: int) -> OpenAI:
        print(f"Using judge API key {key_idx + 1}/{len(self.api_keys)}")
        return OpenAI(api_key=self.api_keys[key_idx], base_url=self.args.base_url)

    def _switch_to_next_key(self):
        if len(self.api_keys) <= 1:
            return
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        self.judge_client = self._make_client(self.current_key_idx)

    def _is_quota_error(self, error: Exception) -> bool:
        text = str(error).lower()
        return any(term in text for term in ["quota", "rate limit", "rate_limit", "429", "resource_exhausted", "exceeded"])

    def _is_permanent_quota_error(self, error: Exception) -> bool:
        text = str(error).lower()
        return any(term in text for term in ["prepayment credits are depleted", "billing", "daily request", "requests per day"])

    def rule_judge(self, response: str, norm_gts: List[str]) -> str:
        """Simple string matching judge."""
        norm_res = normalize_answer(response)
        for gt in norm_gts:
            if gt and gt in norm_res:
                return "true"
        return "false"

    def llm_judge(self, question: str, response: str, answer_list: List[str]) -> str:
        """LLM based judge with retry logic."""
        prompt = (
            f"Question: {question}\n"
            f"Response: {response}\n"
            f"Correct Answers: {answer_list}\n"
            f"Please judge whether the response is correct or not. "
            f"Return 't' if the response is correct, and 'f' if the response is incorrect. "
            f"Don't add any additional information."
        )
        
        max_attempts = max(5, 5 * len(self.api_keys))
        for attempt in range(max_attempts):
            try:
                completion = self.judge_client.chat.completions.create(
                    model=self.args.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                res = completion.choices[0].message.content.strip().lower()
                if 't' in res: return "true"
                if 'f' in res: return "false"
                print(f"Judge API returned invalid answer with key {self.current_key_idx + 1}: {res}; retrying.")
            except Exception as e:
                print(f"Judge API failed (attempt {attempt+1}) with key {self.current_key_idx + 1}: {e}")
                if self._is_quota_error(e):
                    self._switch_to_next_key()
        return "error"

    def process_data(self):
        dataset = load_dataset("parquet", data_files=self.args.data_path, split="train")
        if self.args.max_samples:
            sample_count = min(self.args.max_samples, len(dataset))
            if sample_count < self.args.max_samples:
                print(
                    f"Requested --max_samples {self.args.max_samples}, but dataset only has "
                    f"{len(dataset)} examples. Processing {sample_count} examples instead."
                )
            dataset = dataset.select(range(sample_count))
        if self.args.resume:
            processed_qids = load_existing_qids(self.args.output_path)
            output_mode = 'a'
            print(f"Resume enabled: found {len(processed_qids)} already processed IDs in {self.args.output_path}")
        else:
            processed_qids = set()
            output_mode = 'w'
            if os.path.exists(self.args.output_path):
                print(f"Resume disabled: overwriting existing output file {self.args.output_path}")
        
        all_correct_count = 0
        all_incorrect_count = 0

        with open(self.args.output_path, output_mode, encoding='utf-8') as f:
            for item in tqdm(dataset, desc=f"Sampling ({self.args.judge_type} judge)"):
                qid = str(item.get('question_id', ''))
                if qid in processed_qids: continue

                question = item.get('question', '')
                if not question or 'answer' not in item: continue

                # Get ground truth
                raw_aliases = []
                for col in ['aliases', 'normalized_aliases']:
                    val = item['answer'].get(col)
                    if val: raw_aliases.extend(val) if isinstance(val, list) else raw_aliases.append(str(val))
                
                norm_gts = [normalize_answer(a) for a in set(raw_aliases) if a]
                if not norm_gts: continue

                suffix = "Respond with the answer only, without any explanation."
                # Sampling
                messages = [{"role": "user", "content": f"{question.strip()} {suffix}"}]
                responses = []
                judges = []
                
                # Cache for LLM judge to avoid redundant API calls for the same response in 10 samples
                judge_cache = {}

                for _ in range(self.args.sample_num):
                    try:
                        ans = self._sample_answer(messages)
                        responses.append(ans)

                        # 1. Uncertainty check (Rule-based pre-filter)
                        uncertain_terms = ["don't know", "cannot", "not provided", "no information"]
                        if any(term in ans.lower() for term in uncertain_terms):
                            judges.append("uncertain")
                            continue

                        # 2. Correctness check
                        if self.args.judge_type == "rule":
                            judges.append(self.rule_judge(ans, norm_gts))
                        else:
                            # Use cache to save tokens if model repeats the same answer
                            if ans not in judge_cache:
                                judge_cache[ans] = self.llm_judge(question, ans, raw_aliases)
                            judges.append(judge_cache[ans])

                    except Exception as e:
                        print(f"Sampling error at {qid}: {e}")
                        break

                if len(responses) < self.args.sample_num: continue

                # Stats update
                true_count = judges.count("true")
                if true_count == self.args.sample_num: all_correct_count += 1
                elif true_count == 0: all_incorrect_count += 1

                # Save record
                result = {
                    qid: {
                        "question": f"{question.strip()} {suffix}",
                        "responses": responses,
                        "judges": judges,
                        "ground_truth": list(set(raw_aliases))
                    }
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
                processed_qids.add(qid)
                
                if len(processed_qids) % 10 == 0:
                    tqdm.write(f"Stats -> All-Correct: {all_correct_count}, All-Incorrect: {all_incorrect_count}")

if __name__ == "__main__":
    args = parse_args()
    sampler = ConsistencySampler(args)
    sampler.process_data()
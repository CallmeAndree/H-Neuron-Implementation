import os
import json
import re
import string
import argparse
import time
from typing import List, Set, Dict, Optional

import torch
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from openai import OpenAI  # 用于 LLM Judge

def parse_args():
    parser = argparse.ArgumentParser(description="Consistency Filtering with Rule or LLM Judge.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model for sampling")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the TriviaQA parquet file")
    parser.add_argument("--output_path", type=str, default="data/consistency_samples.jsonl", help="Output path")
    
    parser.add_argument("--sample_num", type=int, default=10, help="Samples per question")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of questions to process")
    parser.add_argument("--gpu_util", type=float, default=0.7, help="vLLM GPU memory utilization")
    parser.add_argument("--tp_size", type=int, default=None, help="Tensor parallel size")

    parser.add_argument("--judge_type", type=str, choices=["rule", "llm"], default="rule", help="How to judge correctness")
    parser.add_argument("--api_key", type=str, default=None, help="Single API key for LLM Judge")
    parser.add_argument("--api_keys", nargs="+", default=None, help="Multiple API keys for LLM Judge rotation")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Model name for LLM Judge")
    parser.add_argument("--rpm_limit", type=int, default=15, help="Requests per minute limit per API key")
    parser.add_argument("--rpd_limit", type=int, default=500, help="Requests per day limit per API key")
    parser.add_argument("--usage_path", type=str, default=None, help="Path to save API usage state. Defaults to output_path + '.judge_usage.json'")
    parser.add_argument("--resume", action="store_true", help="Resume API usage state from usage_path instead of starting fresh")
    
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
        
        # 1. Init Sampling LLM (vLLM)
        self.tp_size = args.tp_size or torch.cuda.device_count()

        self.sampling_llm = LLM(
            model=args.model_path,
            tensor_parallel_size=self.tp_size,
            gpu_memory_utilization=args.gpu_util,
            trust_remote_code=True
        )

        self.sampling_params = SamplingParams(
            temperature=1.0, top_p=0.9, top_k=50, max_tokens=50
        )

        # 2. Init Judge Client (if needed)
        self.judge_client = None
        if args.judge_type == "llm":
            self.api_keys = self._load_api_keys()
            self.usage_path = args.usage_path or f"{args.output_path}.judge_usage.json"
            self.usage_state = self._load_usage_state() if args.resume else self._new_usage_state()
            self.current_key_idx = self._find_available_key_or_wait()
            self.judge_client = self._make_client(self.current_key_idx)

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

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _new_usage_state(self) -> Dict[str, object]:
        return {
            "date": self._today(),
            "current_key_idx": 0,
            "daily_counts": [0 for _ in self.api_keys],
            "minute_timestamps": [[] for _ in self.api_keys],
        }

    def _load_usage_state(self) -> Dict[str, object]:
        if os.path.exists(self.usage_path):
            try:
                with open(self.usage_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if len(state.get("daily_counts", [])) == len(self.api_keys):
                    if state.get("date") != self._today():
                        state["date"] = self._today()
                        state["daily_counts"] = [0 for _ in self.api_keys]
                        state["minute_timestamps"] = [[] for _ in self.api_keys]
                    return state
            except Exception as e:
                print(f"Could not load judge usage state from {self.usage_path}: {e}")
        return self._new_usage_state()

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
            print(f"Warning: could not atomically replace judge usage file ({e}); writing directly.")
            with open(self.usage_path, "w", encoding="utf-8") as f:
                json.dump(self.usage_state, f, ensure_ascii=False, indent=2)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except PermissionError:
                pass

    def _cleanup_minute_window(self, key_idx: int):
        now = time.time()
        self.usage_state["minute_timestamps"][key_idx] = [
            ts for ts in self.usage_state["minute_timestamps"][key_idx]
            if now - ts < 60
        ]

    def _key_has_daily_quota(self, key_idx: int) -> bool:
        return self.usage_state["daily_counts"][key_idx] < self.args.rpd_limit

    def _key_has_minute_quota(self, key_idx: int) -> bool:
        self._cleanup_minute_window(key_idx)
        return len(self.usage_state["minute_timestamps"][key_idx]) < self.args.rpm_limit

    def _find_available_key(self) -> Optional[int]:
        start_idx = int(self.usage_state.get("current_key_idx", 0)) % len(self.api_keys)
        for offset in range(len(self.api_keys)):
            idx = (start_idx + offset) % len(self.api_keys)
            if self._key_has_daily_quota(idx) and self._key_has_minute_quota(idx):
                self.usage_state["current_key_idx"] = idx
                return idx
        return None

    def _seconds_until_next_key_available(self) -> int:
        waits = []
        now = time.time()
        for idx in range(len(self.api_keys)):
            if not self._key_has_daily_quota(idx):
                continue
            self._cleanup_minute_window(idx)
            stamps = self.usage_state["minute_timestamps"][idx]
            if len(stamps) < self.args.rpm_limit:
                return 0
            waits.append(max(1, int(61 - (now - min(stamps)))))
        if waits:
            return min(waits)
        raise RuntimeError(f"All API keys reached the daily request limit of {self.args.rpd_limit}.")

    def _find_available_key_or_wait(self) -> int:
        while True:
            idx = self._find_available_key()
            if idx is not None:
                return idx
            delay = self._seconds_until_next_key_available()
            print(f"All judge keys are at RPM limit; waiting {delay}s before retrying.")
            time.sleep(delay)

    def _make_client(self, key_idx: int) -> OpenAI:
        print(
            f"Using judge API key {key_idx + 1}/{len(self.api_keys)}; "
            f"today={self.usage_state['daily_counts'][key_idx]}/{self.args.rpd_limit}, "
            f"minute={len(self.usage_state['minute_timestamps'][key_idx])}/{self.args.rpm_limit}"
        )
        return OpenAI(api_key=self.api_keys[key_idx], base_url=self.args.base_url)

    def _switch_to_available_key_or_wait(self):
        idx = self._find_available_key_or_wait()
        if idx != self.current_key_idx:
            self.current_key_idx = idx
            self.judge_client = self._make_client(idx)
        self._save_usage_state()

    def _record_request(self):
        now = time.time()
        self.usage_state["daily_counts"][self.current_key_idx] += 1
        self.usage_state["minute_timestamps"][self.current_key_idx].append(now)
        self._cleanup_minute_window(self.current_key_idx)
        self._save_usage_state()

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
            self._switch_to_available_key_or_wait()
            try:
                completion = self.judge_client.chat.completions.create(
                    model=self.args.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                self._record_request()

                res = completion.choices[0].message.content.strip().lower()
                if 't' in res: return "true"
                if 'f' in res: return "false"
                print(f"Judge API returned invalid answer with key {self.current_key_idx + 1}: {res}; retrying.")
            except Exception as e:
                print(f"Judge API failed (attempt {attempt+1}) with key {self.current_key_idx + 1}: {e}")
                if self._is_permanent_quota_error(e):
                    self.usage_state["daily_counts"][self.current_key_idx] = self.args.rpd_limit
                    self._save_usage_state()
                elif self._is_quota_error(e):
                    self.usage_state["minute_timestamps"][self.current_key_idx].append(time.time())
                    self._cleanup_minute_window(self.current_key_idx)
                    self._save_usage_state()
                time.sleep(1)
        return "error"

    def process_data(self):
        if self.args.judge_type == "llm":
            os.makedirs(os.path.dirname(self.usage_path) or ".", exist_ok=True)
            if not self.args.resume:
                self.usage_state = self._new_usage_state()
                self._save_usage_state()

        dataset = load_dataset("parquet", data_files=self.args.data_path, split="train")
        if self.args.max_samples:
            dataset = dataset.select(range(self.args.max_samples))
        processed_qids = load_existing_qids(self.args.output_path)
        
        all_correct_count = 0
        all_incorrect_count = 0

        with open(self.args.output_path, 'a', encoding='utf-8') as f:
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
                        outputs = self.sampling_llm.chat(messages, self.sampling_params, use_tqdm=False)
                        ans = outputs[0].outputs[0].text.strip()
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
                
                if len(processed_qids) % 10 == 0:
                    tqdm.write(f"Stats -> All-Correct: {all_correct_count}, All-Incorrect: {all_incorrect_count}")

if __name__ == "__main__":
    args = parse_args()
    sampler = ConsistencySampler(args)
    sampler.process_data()
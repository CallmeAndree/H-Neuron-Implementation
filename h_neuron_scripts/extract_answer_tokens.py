import os
import json
import argparse
import time
from typing import Dict, List, Optional, Set

from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Extract answer tokens from consistent responses.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to samples files")
    parser.add_argument("--output_path", type=str, default="data/answer_tokens.jsonl", help="Path to save processed results")
    parser.add_argument("--tokenizer_path", type=str, default="data/activations", help="Path to the target model tokenizer")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output_path and usage_path instead of starting fresh")

    # LLM Extractor Config
    parser.add_argument("--api_key", type=str, default=None, help="Single OpenAI-compatible API key")
    parser.add_argument("--api_keys", nargs="+", default=None, help="Multiple OpenAI-compatible API keys for rotation")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="API Base URL")
    parser.add_argument("--llm_model", type=str, default="gpt-4o", help="LLM for extraction")

    # Quota controls. Gemini 3.1 Flash Lite free tier is commonly 15 RPM and 500 RPD.
    parser.add_argument("--rpm_limit", type=int, default=15, help="Requests per minute limit per API key")
    parser.add_argument("--rpd_limit", type=int, default=500, help="Requests per day limit per API key")
    parser.add_argument("--usage_path", type=str, default=None, help="Path to save API usage state. Defaults to output_path + '.usage.json'")

    return parser.parse_args()


# ==========================================
# LLM Prompt Templates
# ==========================================

USER_INPUT_TEMPLATE = """Question: {question}
Response: {response}
Tokenized Response with indices: {indexed_response_tokens}
Please identify the answer span in the tokenized response. Return only a JSON array of integer token indices from the indexed tokenized response. The selected indices must be valid, in ascending order, and should form the minimal answer span with redundant context removed."""

EXAMPLE_MESSAGES = [
    {
        "role": "user",
        "content": "Question: What is the correct name for the \"Flying Lady\" ornament on a Rolls Royce radiator.\nResponse: The correct name for the \"Flying Lady\" ornament on a Rolls Royce radiator is the Spirit of Ecstasy.\nTokenized Response with indices: [(0, '▁The'), (1, '▁correct'), (2, '▁name'), (3, '▁for'), (4, '▁the'), (5, '▁\"'), (6, 'F'), (7, 'lying'), (8, '▁Lady'), (9, '\"'), (10, '▁or'), (11, 'nament'), (12, '▁on'), (13, '▁a'), (14, '▁Roll'), (15, 's'), (16, '▁Roy'), (17, 'ce'), (18, '▁radi'), (19, 'ator'), (20, '▁is'), (21, '▁the'), (22, '▁Spirit'), (23, '▁of'), (24, '▁Ec'), (25, 'st'), (26, 'asy'), (27, '.')]\nPlease identify the answer span in the tokenized response. Return only a JSON array of integer token indices from the indexed tokenized response."
    },
    {
        "role": "assistant",
        "content": "[21, 22, 23, 24, 25, 26]"
    },
]


# ==========================================
# Extraction Logic
# ==========================================

class AnswerTokenExtractor:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        self.api_keys = self._load_api_keys()
        self.usage_path = args.usage_path or f"{args.output_path}.usage.json"
        self.usage_state = self._load_usage_state() if args.resume else self._new_usage_state()
        self.current_key_idx = self._find_available_key_or_wait()
        self.client = self._make_client(self.current_key_idx)

    def _load_api_keys(self) -> List[str]:
        keys = []
        if self.args.api_keys:
            keys.extend(self.args.api_keys)
        if self.args.api_key:
            keys.append(self.args.api_key)

        keys = [key.strip() for key in keys if key and key.strip()]
        if not keys:
            raise ValueError("Provide at least one API key with --api_key or --api_keys.")
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
                print(f"Could not load usage state from {self.usage_path}: {e}")
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
            # On Windows, os.replace can fail if VS Code, antivirus, or file sync
            # briefly locks the usage file. Fall back to a direct non-atomic write
            # so extraction can continue instead of crashing.
            print(f"Warning: could not atomically replace usage file ({e}); writing directly.")
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
            print(f"All keys are at RPM limit; waiting {delay}s before retrying.")
            time.sleep(delay)

    def _make_client(self, key_idx: int) -> OpenAI:
        print(
            f"Using API key {key_idx + 1}/{len(self.api_keys)}; "
            f"today={self.usage_state['daily_counts'][key_idx]}/{self.args.rpd_limit}, "
            f"minute={len(self.usage_state['minute_timestamps'][key_idx])}/{self.args.rpm_limit}"
        )
        return OpenAI(api_key=self.api_keys[key_idx], base_url=self.args.base_url)

    def _switch_to_available_key_or_wait(self):
        idx = self._find_available_key_or_wait()
        if idx != self.current_key_idx:
            self.current_key_idx = idx
            self.client = self._make_client(idx)
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

    def get_tokenized_list(self, text: str) -> List[str]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        return tokens

    def extract_via_llm(self, question: str, response: str, tokens: List[str]) -> Optional[List[str]]:
        """Request LLM to select answer-token indices from the tokenized list."""
        indexed_tokens = list(enumerate(tokens))
        prompt = USER_INPUT_TEMPLATE.format(
            question=question,
            response=response,
            indexed_response_tokens=str(indexed_tokens)
        )

        max_attempts = max(6, 6 * len(self.api_keys))
        for attempt in range(max_attempts):
            self._switch_to_available_key_or_wait()
            try:
                completion = self.client.chat.completions.create(
                    model=self.args.llm_model,
                    messages=EXAMPLE_MESSAGES + [{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                self._record_request()

                reply = completion.choices[0].message.content.strip()
                if reply.startswith("```"):
                    reply = reply.strip("`").strip()
                    if reply.lower().startswith("json"):
                        reply = reply[4:].strip()
                extracted_indices = json.loads(reply)

                # Validation: selected indices must point to valid tokens.
                if (
                    isinstance(extracted_indices, list)
                    and all(isinstance(i, int) for i in extracted_indices)
                    and extracted_indices == sorted(extracted_indices)
                    and all(0 <= i < len(tokens) for i in extracted_indices)
                ):
                    return [tokens[i] for i in extracted_indices]
                print(
                    f"Invalid token-index extraction from key {self.current_key_idx + 1}; "
                    f"reply={reply}; retrying."
                )
            except Exception as e:
                print(f"Extraction failed (attempt {attempt + 1}) with key {self.current_key_idx + 1}: {e}")
                if self._is_permanent_quota_error(e):
                    self.usage_state["daily_counts"][self.current_key_idx] = self.args.rpd_limit
                    self._save_usage_state()
                elif self._is_quota_error(e):
                    # Treat 429/resource_exhausted as an RPM event unless the message says billing/daily quota.
                    self.usage_state["minute_timestamps"][self.current_key_idx].append(time.time())
                    self._cleanup_minute_window(self.current_key_idx)
                    self._save_usage_state()
                time.sleep(1)
        return None

    def load_processed_ids(self) -> Set[str]:
        """Resume from existing output file."""
        if not os.path.exists(self.args.output_path):
            return set()
        ids = set()
        with open(self.args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ids.update(json.loads(line).keys())
                except Exception:
                    continue
        return ids

    def run(self):
        os.makedirs(os.path.dirname(self.args.output_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.usage_path) or ".", exist_ok=True)

        if self.args.resume:
            processed_ids = self.load_processed_ids()
            output_mode = "a"
            print(f"Resume enabled: found {len(processed_ids)} already processed IDs in {self.args.output_path}")
        else:
            processed_ids = set()
            output_mode = "w"
            if os.path.exists(self.args.output_path):
                print(f"Resume disabled: overwriting existing output file {self.args.output_path}")
            self.usage_state = self._new_usage_state()
            self._save_usage_state()

        with open(self.args.input_path, "r", encoding="utf-8") as f_in, \
             open(self.args.output_path, output_mode, encoding="utf-8") as f_out:

            for line in tqdm(f_in, desc="Processing tokens"):
                data = json.loads(line)
                qid = list(data.keys())[0]
                content = data[qid]

                if qid in processed_ids:
                    continue

                # Ensure all responses have the same judge outcome (true/false)
                judges = content["judges"]
                if len(set(judges)) != 1 or "uncertain" in judges or "error" in judges:
                    continue

                # Take the most frequent response as representative
                responses = content["responses"]
                rep_response = max(set(responses), key=responses.count)

                # Tokenization
                tokenized_list = self.get_tokenized_list(rep_response)

                # LLM Extraction
                answer_tokens = self.extract_via_llm(content["question"], rep_response, tokenized_list)

                if answer_tokens:
                    result = {
                        qid: {
                            "question": content["question"],
                            "response": rep_response,
                            "tokenized_response": tokenized_list,
                            "answer_tokens": answer_tokens,
                            "judge": judges[0]  # Consistently correct or consistently hallucinated
                        }
                    }
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
                    os.fsync(f_out.fileno())
                    processed_ids.add(qid)
                    print(f"Saved {qid} to {self.args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    extractor = AnswerTokenExtractor(args)
    extractor.run()

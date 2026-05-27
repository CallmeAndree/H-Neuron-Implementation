import os
import json
import argparse
import time
from typing import Dict, List, Optional, Set

import torch
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
    parser.add_argument("--per_key_limit", type=int, default=500, help="Maximum successful API calls per key")
    parser.add_argument("--usage_path", type=str, default=None, help="Path to save API key usage state. Defaults to output_path + '.usage.json'")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="API Base URL")
    parser.add_argument("--llm_model", type=str, default="gpt-4o", help="LLM for extraction")

    return parser.parse_args()


# ==========================================
# LLM Prompt Templates
# ==========================================

USER_INPUT_TEMPLATE = """Question: {question}
Response: {response}
Tokenized Response: {response_tokens}
Please help extract the "answer tokens" from all tokens, removing all redundant information, and the tokens you return must be part of the input Tokenized Response list."""

EXAMPLE_MESSAGES = [
    {
        "role": "user",
        "content": "Question: What is the correct name for the \"Flying Lady\" ornament on a Rolls Royce radiator.\nResponse: The correct name for the \"Flying Lady\" ornament on a Rolls Royce radiator is the Spirit of Ecstasy.\nTokenized Response: ['▁The', '▁correct', '▁name', '▁for', '▁the', '▁\"', 'F', 'lying', '▁Lady', '\"', '▁or', 'nament', '▁on', '▁a', '▁Roll', 's', '▁Roy', 'ce', '▁radi', 'ator', '▁is', '▁the', '▁Spirit', '▁of', '▁Ec', 'st', 'asy', '.']\nPlease help extract the \"answer tokens\" from all tokens, removing all redundant information, and the tokens you return must form a continuous segment of the input Tokenized Response list."
    },
    {
        "role": "assistant",
        "content": "['▁the', '▁Spirit', '▁of', '▁Ec', 'st', 'asy']"
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
        self.usage_state = self._load_usage_state() if args.resume else {"current_key_idx": 0, "usage": [0 for _ in self.api_keys]}
        self.current_key_idx = self._find_available_key()
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

    def _load_usage_state(self) -> Dict[str, object]:
        if os.path.exists(self.usage_path):
            try:
                with open(self.usage_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if len(state.get("usage", [])) == len(self.api_keys):
                    return state
            except Exception as e:
                print(f"Could not load usage state from {self.usage_path}: {e}")

        return {"current_key_idx": 0, "usage": [0 for _ in self.api_keys]}

    def _save_usage_state(self):
        tmp_path = f"{self.usage_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.usage_state, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.usage_path)

    def _find_available_key(self) -> int:
        start_idx = int(self.usage_state.get("current_key_idx", 0)) % len(self.api_keys)
        usage = self.usage_state["usage"]
        for offset in range(len(self.api_keys)):
            idx = (start_idx + offset) % len(self.api_keys)
            if usage[idx] < self.args.per_key_limit:
                self.usage_state["current_key_idx"] = idx
                return idx
        raise RuntimeError(f"All API keys reached per-key limit {self.args.per_key_limit}.")

    def _make_client(self, key_idx: int) -> OpenAI:
        print(f"Using API key {key_idx + 1}/{len(self.api_keys)}; usage={self.usage_state['usage'][key_idx]}/{self.args.per_key_limit}")
        return OpenAI(api_key=self.api_keys[key_idx], base_url=self.args.base_url)

    def _rotate_key(self):
        usage = self.usage_state["usage"]
        for offset in range(1, len(self.api_keys) + 1):
            idx = (self.current_key_idx + offset) % len(self.api_keys)
            if usage[idx] < self.args.per_key_limit:
                self.current_key_idx = idx
                self.usage_state["current_key_idx"] = idx
                self._save_usage_state()
                self.client = self._make_client(idx)
                return
        self._save_usage_state()
        raise RuntimeError(f"All API keys reached per-key limit {self.args.per_key_limit}.")

    def _is_quota_error(self, error: Exception) -> bool:
        text = str(error).lower()
        return any(term in text for term in ["quota", "rate limit", "rate_limit", "429", "resource_exhausted", "exceeded"])

    def get_tokenized_list(self, text: str) -> List[str]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        return tokens

    def extract_via_llm(self, question: str, response: str, tokens: List[str]) -> Optional[List[str]]:
        """Request LLM to select tokens from the tokenized list."""
        prompt = USER_INPUT_TEMPLATE.format(
            question=question,
            response=response,
            response_tokens=str(tokens)
        )

        for attempt in range(3 * len(self.api_keys)):
            if self.usage_state["usage"][self.current_key_idx] >= self.args.per_key_limit:
                print(f"API key {self.current_key_idx + 1} reached limit; rotating.")
                self._rotate_key()

            try:
                completion = self.client.chat.completions.create(
                    model=self.args.llm_model,
                    messages=EXAMPLE_MESSAGES + [{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                self.usage_state["usage"][self.current_key_idx] += 1
                self._save_usage_state()

                reply = completion.choices[0].message.content.strip().replace("'", "\"")
                extracted = json.loads(reply)

                # Validation: selected tokens must exist in original sequence
                if all(t in tokens for t in extracted):
                    return extracted
                print(f"Invalid token extraction from key {self.current_key_idx + 1}; retrying.")
            except Exception as e:
                print(f"Extraction failed (attempt {attempt + 1}) with key {self.current_key_idx + 1}: {e}")
                if self._is_quota_error(e):
                    self.usage_state["usage"][self.current_key_idx] = self.args.per_key_limit
                    self._rotate_key()
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
            if os.path.exists(self.usage_path):
                os.remove(self.usage_path)
                print(f"Resume disabled: removed existing usage state {self.usage_path}")
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

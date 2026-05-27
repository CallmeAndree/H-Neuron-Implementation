import os
import json
import argparse
import time
from typing import List, Optional, Set

from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Extract answer tokens from consistent responses.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to samples files")
    parser.add_argument("--output_path", type=str, default="data/answer_tokens.jsonl", help="Path to save processed results")
    parser.add_argument("--tokenizer_path", type=str, default="data/activations", help="Path to the target model tokenizer")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output_path instead of starting fresh")

    # LLM Extractor Config
    parser.add_argument("--api_key", type=str, default=None, help="Single OpenAI-compatible API key")
    parser.add_argument("--api_keys", nargs="+", default=None, help="Multiple OpenAI-compatible API keys for rotation")
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
        self.exhausted_key_indices = set()
        self.current_key_idx = 0
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

    def _make_client(self, key_idx: int) -> OpenAI:
        print(f"Using API key {key_idx + 1}/{len(self.api_keys)}")
        return OpenAI(api_key=self.api_keys[key_idx], base_url=self.args.base_url)

    def _rotate_key(self):
        if len(self.exhausted_key_indices) >= len(self.api_keys):
            raise RuntimeError("All API keys failed with quota/rate-limit errors during this run.")

        for offset in range(1, len(self.api_keys) + 1):
            idx = (self.current_key_idx + offset) % len(self.api_keys)
            if idx not in self.exhausted_key_indices:
                self.current_key_idx = idx
                self.client = self._make_client(idx)
                return

        raise RuntimeError("No available API key remains after rotation.")

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

        max_attempts = max(3, 3 * len(self.api_keys))
        for attempt in range(max_attempts):
            try:
                completion = self.client.chat.completions.create(
                    model=self.args.llm_model,
                    messages=EXAMPLE_MESSAGES + [{"role": "user", "content": prompt}],
                    temperature=0.0
                )

                reply = completion.choices[0].message.content.strip().replace("'", "\"")
                extracted = json.loads(reply)

                # Validation: selected tokens must exist in original sequence
                if all(t in tokens for t in extracted):
                    return extracted
                print(f"Invalid token extraction from key {self.current_key_idx + 1}; retrying.")
            except Exception as e:
                print(f"Extraction failed (attempt {attempt + 1}) with key {self.current_key_idx + 1}: {e}")
                if self._is_quota_error(e):
                    self.exhausted_key_indices.add(self.current_key_idx)
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

        if self.args.resume:
            processed_ids = self.load_processed_ids()
            output_mode = "a"
            print(f"Resume enabled: found {len(processed_ids)} already processed IDs in {self.args.output_path}")
        else:
            processed_ids = set()
            output_mode = "w"
            if os.path.exists(self.args.output_path):
                print(f"Resume disabled: overwriting existing output file {self.args.output_path}")

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

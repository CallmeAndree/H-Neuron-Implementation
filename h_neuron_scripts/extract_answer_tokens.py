import os
import json
import argparse
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Extract answer tokens from consistent responses.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to samples files")
    parser.add_argument("--output_path", type=str, default="data/answer_tokens.jsonl", help="Path to save processed results")
    parser.add_argument("--tokenizer_path", type=str, default="data/activations", help="Path to the target model tokenizer")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output_path by skipping already processed question IDs")

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

    def _switch_to_next_key(self):
        if len(self.api_keys) <= 1:
            return
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        self.client = self._make_client(self.current_key_idx)

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
            try:
                completion = self.client.chat.completions.create(
                    model=self.args.llm_model,
                    messages=EXAMPLE_MESSAGES + [{"role": "user", "content": prompt}],
                    temperature=0.0
                )
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
                if self._is_quota_error(e):
                    self._switch_to_next_key()
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

    def get_single_sample(self, row: Dict[str, Any], line_num: int) -> Tuple[str, Dict[str, Any]]:
        if not isinstance(row, dict):
            raise ValueError(f"line {line_num}: JSONL row is not a JSON object.")

        # Supported input format 1: {"tc_1": {"question": ..., "responses": ..., "judges": ...}}
        if len(row) == 1:
            qid = next(iter(row))
            content = row[qid]
            if not isinstance(content, dict):
                raise ValueError(
                    f"line {line_num}, qid {qid}: expected sample content to be an object, "
                    f"got {type(content).__name__}."
                )
            return qid, content

        # Supported input format 2: {"qid": "tc_1", "question": ..., "responses": ..., "judges": ...}
        if "qid" in row:
            qid = row["qid"]
            if not isinstance(qid, str) or not qid.strip():
                raise ValueError(f"line {line_num}: qid must be a non-empty string.")
            content = {key: value for key, value in row.items() if key != "qid"}
            return qid, content

        raise ValueError(
            f"line {line_num}: expected either one wrapped top-level qid or a flat row "
            f"with a qid field; got keys {list(row.keys())}."
        )

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

            for line_num, line in enumerate(tqdm(f_in, desc="Processing tokens"), start=1):
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                qid, content = self.get_single_sample(data, line_num)

                if qid in processed_ids:
                    continue

                # Ensure all responses have the same judge outcome (true/false)
                judges = content.get("judges")
                if not isinstance(judges, list) or not judges:
                    raise ValueError(f"line {line_num}, qid {qid}: missing or empty judges list.")
                normalized_judges = [str(judge).strip().lower() for judge in judges]
                if len(set(normalized_judges)) != 1 or normalized_judges[0] not in {"true", "false"}:
                    continue

                # Take the most frequent response as representative
                responses = content.get("responses")
                if not isinstance(responses, list) or not responses:
                    raise ValueError(f"line {line_num}, qid {qid}: missing or empty responses list.")
                rep_response = max(set(responses), key=responses.count)

                # Tokenization
                tokenized_list = self.get_tokenized_list(rep_response)

                # LLM Extraction
                question = content.get("question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError(f"line {line_num}, qid {qid}: missing or empty question string.")
                answer_tokens = self.extract_via_llm(question, rep_response, tokenized_list)

                if answer_tokens:
                    result = {
                        qid: {
                            "question": question,
                            "response": rep_response,
                            "tokenized_response": tokenized_list,
                            "answer_tokens": answer_tokens,
                            "judge": normalized_judges[0]  # Consistently correct or consistently hallucinated
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

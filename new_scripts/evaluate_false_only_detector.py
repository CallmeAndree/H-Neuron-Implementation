import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a hallucination detector on a false-only test set.")
    parser.add_argument("--classifier_path", type=str, default="data/small_subset_qwen7b/classifier/classifier.pkl")
    parser.add_argument("--test_ids", type=str, default="data/small_subset_qwen7b/test_qwen_ids.json")
    parser.add_argument("--test_acts", type=str, default="data/small_subset_qwen7b/test_activations/answer_tokens")
    parser.add_argument("--output_path", type=str, default="data/small_subset_qwen7b/false_only_detector_results.json")
    return parser.parse_args()


def main():
    args = parse_args()

    classifier = joblib.load(args.classifier_path)

    with open(args.test_ids, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    false_ids = id_map.get("f", [])
    true_ids = id_map.get("t", [])

    X = []
    used_qids = []
    missing_qids = []

    for qid in false_ids:
        act_path = Path(args.test_acts) / f"act_{qid}.npy"
        if act_path.exists():
            X.append(np.load(act_path).flatten())
            used_qids.append(qid)
        else:
            missing_qids.append(qid)

    if not X:
        raise RuntimeError(f"No activation files found in {args.test_acts}")

    X = np.array(X)
    preds = classifier.predict(X)
    probs = classifier.predict_proba(X)[:, 1]

    predicted_hallucination = int((preds == 1).sum())
    predicted_non_hallucination = int((preds == 0).sum())
    false_sample_recall = float((preds == 1).mean())
    average_hallucination_probability = float(probs.mean())

    results = {
        "classifier_path": args.classifier_path,
        "test_ids": args.test_ids,
        "test_acts": args.test_acts,
        "input_false_ids": len(false_ids),
        "input_true_ids": len(true_ids),
        "tested_false_samples": len(used_qids),
        "missing_activation_files": len(missing_qids),
        "predicted_hallucination": predicted_hallucination,
        "predicted_non_hallucination": predicted_non_hallucination,
        "false_sample_recall": false_sample_recall,
        "average_hallucination_probability": average_hallucination_probability,
        "min_hallucination_probability": float(probs.min()),
        "max_hallucination_probability": float(probs.max()),
        "used_qids": used_qids,
        "missing_qids": missing_qids,
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("--- False-only hallucination detector evaluation ---")
    print(f"Classifier: {args.classifier_path}")
    print(f"Test ids: {args.test_ids}")
    print(f"Test activations: {args.test_acts}")
    print(f"Input false ids: {len(false_ids)}")
    print(f"Input true ids: {len(true_ids)}")
    print(f"Tested false samples: {len(used_qids)}")
    print(f"Missing activation files: {len(missing_qids)}")
    print(f"Predicted hallucination: {predicted_hallucination}")
    print(f"Predicted non-hallucination: {predicted_non_hallucination}")
    print(f"False-sample recall: {false_sample_recall:.4f}")
    print(f"Average hallucination probability: {average_hallucination_probability:.4f}")
    print(f"Saved results to: {args.output_path}")


if __name__ == "__main__":
    main()

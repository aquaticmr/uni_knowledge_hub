import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import evaluate
from datasets import load_dataset
from transformers import pipeline


TASK_FIELD_CANDIDATES: Dict[str, List[Tuple[str, str]]] = {
    "sst2": [("sentence", "")],
    "cola": [("sentence", "")],
    "mrpc": [("sentence1", "sentence2")],
    "qqp": [("question1", "question2")],
    "rte": [("sentence1", "sentence2")],
    "wnli": [("sentence1", "sentence2")],
    "stsb": [("sentence1", "sentence2")],
    "mnli": [("premise", "hypothesis")],
    "qnli": [("question", "sentence")],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GLUE evaluation for a Hugging Face model.")
    parser.add_argument("--task", default="sst2", help="GLUE task name, e.g. sst2, cola, mrpc, qqp.")
    parser.add_argument(
        "--model",
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Hugging Face model id for text classification/regression.",
    )
    parser.add_argument("--split", default="validation", help="Dataset split name.")
    parser.add_argument("--max-samples", type=int, default=500, help="Maximum samples to evaluate.")
    parser.add_argument("--batch-size", type=int, default=16, help="Pipeline batch size.")
    parser.add_argument("--save", default="", help="Optional path to save a JSON report.")
    return parser.parse_args()


def resolve_text_fields(task: str, dataset_features: List[str]) -> Tuple[str, str]:
    if task in TASK_FIELD_CANDIDATES:
        for first, second in TASK_FIELD_CANDIDATES[task]:
            if first in dataset_features and (not second or second in dataset_features):
                return first, second

    for first, second in TASK_FIELD_CANDIDATES.get("sst2", []):
        if first in dataset_features:
            return first, second

    raise ValueError(
        f"Unable to detect text fields for task '{task}'. Available fields: {dataset_features}"
    )


def label_to_int(raw_label, prediction_map: Dict[str, int]):
    if isinstance(raw_label, int):
        return raw_label

    if isinstance(raw_label, float):
        return raw_label

    normalized = str(raw_label).strip().upper()
    if normalized in prediction_map:
        return prediction_map[normalized]

    if normalized.startswith("LABEL_"):
        try:
            return int(normalized.split("_", 1)[1])
        except ValueError:
            pass

    if normalized in {"NEGATIVE", "ENTAILMENT"}:
        return 0

    if normalized in {"POSITIVE", "CONTRADICTION"}:
        return 1

    if normalized in {"NEUTRAL"}:
        return 2

    raise ValueError(f"Cannot map predicted label '{raw_label}' to numeric class id")


def main() -> None:
    args = parse_args()

    task = args.task.lower().strip()
    metric = evaluate.load("glue", task)
    dataset = load_dataset("glue", task, split=args.split)

    if args.max_samples and args.max_samples > 0:
        max_n = min(args.max_samples, len(dataset))
        dataset = dataset.select(range(max_n))

    feature_names = list(dataset.features.keys())
    text_key_1, text_key_2 = resolve_text_fields(task, feature_names)

    clf = pipeline("text-classification", model=args.model, tokenizer=args.model, truncation=True)

    predictions = []
    references = dataset["label"]

    label_name_to_id = {}
    label_feature = dataset.features.get("label")
    if hasattr(label_feature, "names") and label_feature.names:
        for idx, name in enumerate(label_feature.names):
            label_name_to_id[name.upper()] = idx

    for i in range(0, len(dataset), args.batch_size):
        batch = dataset[i : i + args.batch_size]
        if text_key_2:
            inputs = [f"{a} [SEP] {b}" for a, b in zip(batch[text_key_1], batch[text_key_2])]
        else:
            inputs = batch[text_key_1]

        outputs = clf(inputs)
        for out in outputs:
            predictions.append(label_to_int(out.get("label", ""), label_name_to_id))

    result = metric.compute(predictions=predictions, references=references)

    report = {
        "task": task,
        "model": args.model,
        "split": args.split,
        "num_samples": len(dataset),
        "metrics": result,
    }

    print(json.dumps(report, indent=2))

    if args.save:
        save_path = Path(args.save)
        save_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report: {save_path}")


if __name__ == "__main__":
    main()

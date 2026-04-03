import argparse
import json
import re
import statistics
from pathlib import Path

from bs4 import BeautifulSoup
import evaluate
import requests


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def fetch_answer(api_base: str, question: str, timeout: int = 60) -> str:
    url = f"{api_base.rstrip('/')}/chat"
    resp = requests.post(url, json={"question": question}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("answer") or "").strip()


def query_terms(question: str) -> set[str]:
    stopwords = {
        "the", "is", "are", "a", "an", "of", "for", "to", "in", "on", "and",
        "give", "tell", "me", "about", "what", "details", "complete",
    }
    words = re.findall(r"[a-zA-Z0-9]+", (question or "").lower())
    return {w for w in words if len(w) >= 3 and w not in stopwords}


def fetch_web_lines(url: str, timeout: int = 45) -> list[str]:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    nodes = soup.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li", "td", "th"])

    lines = []
    for node in nodes:
        line = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if len(line) < 20:
            continue
        if len(line) > 260:
            continue
        lines.append(line)

    seen = set()
    deduped = []
    for line in lines:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    return deduped


def build_reference_from_urls(question: str, urls: list[str], max_lines: int = 16) -> str:
    terms = query_terms(question)
    candidates = []

    for url in urls:
        try:
            lines = fetch_web_lines(url)
        except Exception:
            continue

        for line in lines:
            lower = line.lower()
            overlap = sum(1 for t in terms if t in lower)
            bonus = 2 if any(k in lower for k in ["fees", "fee", "program", "hostel", "tuition", "caution", "development"]) else 0
            score = overlap + bonus
            candidates.append((score, line))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = [line for score, line in candidates if score > 0][:max_lines]
    if not selected:
        selected = [line for _, line in candidates][:max_lines]
    return "\n".join(selected).strip()


def main():
    parser = argparse.ArgumentParser(description="Compute ROUGE for project chat responses.")
    parser.add_argument("--dataset", default="qa_testset.jsonl", help="Path to jsonl dataset with question and reference_urls")
    parser.add_argument("--api-base", default="http://127.0.0.1:8080", help="Backend API base URL")
    parser.add_argument("--save", default="rouge_report.json", help="Output report JSON file")
    parser.add_argument("--allow-fallback-reference", action="store_true", help="Allow fallback to static 'reference' when URL references are absent")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    rows = load_jsonl(dataset_path)
    if not rows:
        raise SystemExit("Dataset is empty.")

    predictions = []
    references = []
    examples = []

    for idx, row in enumerate(rows, start=1):
        q = (row.get("question") or "").strip()
        urls = row.get("reference_urls") or []
        manual_reference = (row.get("reference") or "").strip()
        if not q:
            continue

        if urls:
            r = build_reference_from_urls(q, urls)
        elif args.allow_fallback_reference and manual_reference:
            r = manual_reference
        else:
            continue

        if not r:
            continue

        pred = fetch_answer(args.api_base, q)
        predictions.append(pred)
        references.append(r)
        examples.append({
            "index": idx,
            "question": q,
            "reference_urls": urls,
            "reference": r,
            "prediction": pred,
            "prediction_length": len(pred.split()),
        })

    if not predictions:
        raise SystemExit("No valid question/reference pairs found.")

    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=predictions, references=references)

    report = {
        "api_base": args.api_base,
        "samples": len(predictions),
        "scores": scores,
        "avg_prediction_tokens": statistics.mean(e["prediction_length"] for e in examples),
        "examples": examples,
    }

    print("ROUGE scores:")
    for k, v in scores.items():
        print(f"  {k}: {v:.6f}")

    out_path = Path(args.save)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()

# Evaluation: ROUGE and GLUE

This folder provides two scripts:

- `compute_rouge.py`: evaluates your project responses from `/chat` using question-reference pairs.
- `compute_glue.py`: runs an official GLUE benchmark task for a selected model.

## 1) Install eval dependencies

```powershell
Set-Location d:\Projects\Mohit_Rudrakar\backend\evaluation
d:\Projects\Mohit_Rudrakar\.venv\Scripts\python.exe -m pip install -r requirements-eval.txt
```

## 2) ROUGE for your project

Prepare dataset in `qa_testset.jsonl` with one JSON object per line:

```json
{"question":"...", "reference":"..."}
```

Run:

```powershell
Set-Location d:\Projects\Mohit_Rudrakar\backend\evaluation
d:\Projects\Mohit_Rudrakar\.venv\Scripts\python.exe compute_rouge.py --dataset qa_testset.jsonl --api-base http://127.0.0.1:8080 --save rouge_report.json
```

Output:

- console scores: `rouge1`, `rouge2`, `rougeL`, `rougeLsum`
- saved report with per-example predictions: `rouge_report.json`

## 3) GLUE score

GLUE is a model benchmark on predefined NLP tasks. It is not a direct metric for your RAG chatbot output quality, but you can still run it to report model benchmark performance.

Run example (SST-2):

```powershell
Set-Location d:\Projects\Mohit_Rudrakar\backend\evaluation
d:\Projects\Mohit_Rudrakar\.venv\Scripts\python.exe compute_glue.py --task sst2 --model distilbert-base-uncased-finetuned-sst-2-english --split validation --max-samples 500 --save glue_report.json
```

Output:

- console task score (for SST-2 this is `accuracy`)
- saved report: `glue_report.json`

## 4) Recommended project reporting

For your RBU assistant, report:

1. ROUGE on your own QA test set (project-relevant).
2. Retrieval diagnostics (source-hit rate, answer completeness checks).
3. Optional GLUE task score as a separate model benchmark (not as chatbot quality metric).

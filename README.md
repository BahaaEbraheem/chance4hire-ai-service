# Chance4Hire AI Service — Bilingual (Arabic / English) CV Screening

The NLP engine behind **Chance4Hire**, an AI recruitment platform built as my MSc thesis
(MSc Web Science, Syrian Virtual University). It reads a CV — Arabic, English, or the
code-switched mix real candidates write (`خبرة في Python و ASP.NET Core`) — extracts skills,
education and experience, and scores how well the candidate fits a job.

> Keyword ATS tools miss Arabic CVs. On the gold set below, a naive keyword ATS recovers
> **26.5%** of the skills in Arabic CVs; this service recovers **100%**.

## Results (40-CV gold set, 142 labelled skills)

| Engine | Precision | Recall | F1 | Arabic recall |
|---|---|---|---|---|
| B1. Naive keyword ATS (English, substring) | 0.931 | 0.570 | 0.707 | 0.265 |
| B2. Tuned English ATS (forms + word boundary) | 0.989 | 0.627 | 0.767 | 0.327 |
| B3. Bilingual, no normalization (ablation) | 0.985 | 0.951 | 0.968 | 0.898 |
| **Chance4Hire (bilingual + Arabic normalization)** | **0.986** | **0.986** | **0.986** | **1.000** |

Dataset: 16 Arabic, 14 English and 10 code-switched CVs, including "trap" CVs written to
provoke false positives. Latency: keyword engine 0.3 ms, Transformer engine ~530 ms average
(CPU). Reproduce with `python evaluate.py` — it prints the tables and writes `eval/report.json`.

## How it works

| Stage | Technique |
|---|---|
| **PDF text extraction** | PyMuPDF, chosen by measurement: it kept 12 of 13 skills in a Word-exported Arabic CV versus 8 of 13 with pdfplumber. Detects scanned (image-only) PDFs. |
| **Skill extraction** | Deterministic and explainable: a bilingual skill gazetteer with Arabic normalization (alef/ta-marbuta/diacritics). It never invents a skill that is not in the text. |
| **Semantic matching** | A multilingual BERT sentence encoder (`paraphrase-multilingual-MiniLM-L12-v2`, Sentence-Transformers / PyTorch) embeds CV n-grams and job skills; cosine similarity gives a paraphrase-tolerant, cross-lingual matching score and a confidence. |
| **Education & experience** | Degree gazetteer (Arabic + English) with the nearby year; years of experience from digits, Arabic-Indic digits and Arabic number words (`خمس سنوات`). |
| **Graceful degradation** | If the ML stack or model weights are unavailable, the service falls back to the keyword engine automatically, so it always answers. The response says which engine ran. |

A design note from the evaluation: cosine similarity is used for the *score*, not as a
yes/no presence threshold — short tech tokens cluster too tightly (an absent "REST API"
out-scored a present "Docker").

## API (FastAPI)

| Endpoint | Purpose |
|---|---|
| `POST /analyze` `{cv_text, job_skills[]}` | `extracted_skills`, `matching_score`, `confidence`, `education`, `experience_years`, `mode` |
| `POST /extract` (multipart PDF) | `text`, `pages`, `char_count`, `is_scanned` |
| `GET /metrics` | Precision / recall / F1 against the gold set, latency, baseline comparison |
| `GET /health` | Engine in use and model name |

## How the platform uses it

In Chance4Hire the ASP.NET Core / ABP backend calls this service over HTTP through a
resilient analyzer that falls back to a local engine if the service fails. Analysis progress
streams live to the recruiter over SignalR, the recruiter keeps the final decision (notes,
decision, feedback), and an accuracy dashboard reads `/metrics`. The rest of the platform is
private; this repository is the AI part.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000      # docs at http://localhost:8000/docs
python evaluate.py                                # reproduce the results table
python -m unittest discover tests                 # tests
```

The first run downloads the sentence model (~470 MB) from Hugging Face.

| Env var | Default | Meaning |
|---|---|---|
| `AI_MODE` | `auto` | `auto` (Transformer, else keyword), `transformer` (force), `keyword` (force). |
| `EMBED_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Any sentence-transformers model, e.g. an Arabic-tuned one. |

## Files

- `main.py` — FastAPI app and HTTP contract
- `analyzer.py` — Transformer and keyword engines, Arabic normalization
- `skills.py` — bilingual skill gazetteer
- `cv_profile.py` — education and experience extraction
- `extractor.py` — Arabic-aware PDF extraction
- `evaluate.py`, `eval/` — evaluation harness, gold dataset, last report

## Author

**Bahaa Ebraheem** — Senior Full Stack & AI Engineer · [LinkedIn](https://linkedin.com/in/bahaa-ebraheem-992443354)

# AI Call Transcriber & Auditor

Call-quality audit system for car dealership outbound calls. It transcribes multilingual (Gujarati / Hindi / English) CRE calls and scores each one per call type, using a hybrid of a fine-tuned Qwen3-32B LoRA model, stock Qwen3-32B and deterministic CRM rules. Results are shown in a Flask dashboard with live batch runs and exported as Excel reports.

## Project structure

```
api/          Flask app (main dashboard)
core/         Transcription, recordings, CRM API/DB access, reviews
core/audit/   Audit pipeline: prompts, LLM/LoRA scoring, rules, Excel reports, importer
reviewer/     Human review dashboard (reviewer_wsgi.py)
web/          Dashboard templates
workers/      Batch processing engine
resources/    Report template, call-type parameters, CA cert
models/       Model files (weights not in git, see below)
docs/         Project analysis report
config.py     Settings loaded from .env
wsgi.py       Entry point for the main dashboard
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DB, CRM API, HF token and model settings
```

Needs [Ollama](https://ollama.com) with the Qwen3 models configured in `.env`, plus the ASR and diarizer `.nemo` models in `models/`. These are not committed.

## Not in this repo

Call recordings, transcripts, audit reports and human reviews are **customer data** and are kept out of git (see `.gitignore`). Model weights are also kept local.

# Chatbot Code Blueprints + QA

This repository contains a working prototype for a conversational chatbot and governance framework for agentic AI, MLOps, cloud, and generative AI blueprints.

## Status

This branch includes a functional FastAPI prototype with:
- a browser-based chat UI
- a mock AI mode for local development and tests
- a live AI mode compatible with OpenAI-style APIs
- health monitoring endpoints
- CI automation and Docker support

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000

## API

- GET /health
- POST /api/chat

## License

MIT

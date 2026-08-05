# Research Agent

An end-to-end agentic AI system built with LangGraph and LangChain.
Takes a research question, runs a 4-agent pipeline, and produces
a structured report — with a live REST API and web interface.

## Live Demo
**API:** https://research-agent-513976967636.us-central1.run.app
**Docs:** https://research-agent-513976967636.us-central1.run.app/docs
**Frontend:** Run `streamlit run frontend/app.py` after setup

---

## What this project demonstrates
- Multi-agent orchestration with LangGraph (supervisor pattern, shared state, quality control loop)
- RAG concepts — web retrieval, synthesis, grounded generation
- Production REST API with FastAPI (async endpoints, Pydantic validation, error handling)
- Containerised deployment to GCP Cloud Run (Docker, live public URL)
- Streamlit frontend connecting to the live API

---

## Architecture

User Input
↓
FastAPI REST API (deployed on GCP Cloud Run)
↓
Multi-Agent Pipeline (LangGraph)
↓
Planner → Researcher (DuckDuckGo) → Writer → Critic
↑ ↓
└─revise──┘ (max 2 revisions)
↓
Structured Research Report
↑
Streamlit Frontend (displays report with download option)
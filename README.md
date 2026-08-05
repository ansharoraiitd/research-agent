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

```
User Input
    ↓
Streamlit Frontend
    ↓
FastAPI REST API (GCP Cloud Run)
    ↓
LangGraph Multi-Agent Pipeline
    ↓
Planner → Researcher (DuckDuckGo) → Writer → Critic
                                         ↓         ↓
                                     APPROVED   REVISION NEEDED
                                         ↓         ↓
                                        END     Writer (revise)
                                                   ↓ (max 2 times)
                                                  END
    ↓
Structured Research Report
```

---

## Multi-Agent System

A 4-agent pipeline where each agent is a specialist with one focused job.

| Agent | Job | Tools |
|-------|-----|-------|
| Planner | Creates 3 focused search queries | None |
| Researcher | Searches web, synthesises findings | DuckDuckGo |
| Writer | Writes structured report | None |
| Critic | Reviews quality, requests revision if needed | None |

**Key design decisions:**
- Critic caps revisions at 2 — prevents infinite loops
- Researcher runs in two steps: search then synthesise
- Shared TypedDict state flows through all agents via LangGraph
- Conditional edge after critic — approve or route back to writer

---

## REST API (FastAPI)

Served via FastAPI, deployed on GCP Cloud Run.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/research` | POST | Run research pipeline |

**Example request:**
```bash
curl -X POST https://research-agent-513976967636.us-central1.run.app/research \
  -H "Content-Type: application/json" \
  -d '{"task": "How is LangGraph used in production AI?"}'
```

---

## Project Structure

```
research-agent/
├── research_agent.py           # Week 2: original research agent
├── multi_agent/
│   ├── multi_agent_system.py   # Complete 4-agent pipeline
│   ├── 01_why_multi_agent.py   # Single vs specialist comparison
│   ├── 02_shared_state.py      # Shared state via LangGraph
│   ├── 03_supervisor.py        # Supervisor orchestration pattern
│   └── 04_agents_with_tools.py # Researcher with web search
├── fastapi/
│   ├── 01_basics.py            # FastAPI fundamentals
│   └── 02_agent_api.py         # Agent served as REST API
├── deploy/
│   ├── main.py                 # Production FastAPI app
│   ├── Dockerfile              # Container definition
│   └── requirements.txt        # Pinned dependencies
├── frontend/
│   └── app.py                  # Streamlit web interface
└── .env.example                # Required environment variables
```

---

## Week 2 — LangChain Foundations

| File | What it covers |
|------|---------------|
| 01_langchain_basics.py | LCEL chains, prompt templates, pipe operator |
| 02_tool_use.py | @tool decorator, bind_tools(), agent loop |
| 03_memory.py | ChatMessageHistory, session management |
| 04_langgraph_basics.py | StateGraph, nodes, conditional edges |
| research_agent.py | Complete research agent |

---

## Tech Stack

Python · LangChain · LangGraph · FastAPI · Streamlit
Gemini API · DuckDuckGo Search · Docker · GCP Cloud Run

---

## Setup

```bash
# Clone and install
git clone https://github.com/ansharoraiitd/research-agent
cd research-agent
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run the multi-agent system
python multi_agent/multi_agent_system.py

# Run the API locally
cd fastapi && python 02_agent_api.py

# Run the frontend
cd frontend && streamlit run app.py
```
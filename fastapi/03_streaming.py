# WHAT THIS DOES:
# SSE (Server-Sent Events) streaming sends progress updates as the agent runs.
# Instead of waiting 30s for a response, client sees live status updates.
# StreamingResponse keeps the HTTP connection open and sends chunks as ready.
# This is how ChatGPT and every real AI product handles long operations.

import sys
import os
import json
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Streaming Research API",
    description="Multi-agent research with real-time SSE progress updates",
    version="3.0.0"
)

# ── Request model ────────────────────────────────────────────
class ResearchRequest(BaseModel):
    task: str

# ── SSE helper ───────────────────────────────────────────────
def format_sse(data: dict) -> str:
    """
    Format a dict as an SSE event string.
    SSE format requires: 'data: \n\n'
    The double newline signals end of one event.
    """
    return f"data: {json.dumps(data)}\n\n"

# ── The streaming agent runner ────────────────────────────────
async def run_agent_streaming(task: str) -> AsyncGenerator[str, None]:
    """
    Generator function that runs the agent step by step
    and yields SSE events as each step completes.

    AsyncGenerator means this function can yield multiple times
    keeping the connection open between yields.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.tools import DuckDuckGoSearchRun

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    # ── Event 1: started ─────────────────────────────────────
    yield format_sse({
        "type": "status",
        "agent": "system",
        "message": f"Starting research: {task[:50]}...",
        "step": 0,
        "total_steps": 4
    })

    # ── Event 2: planner ─────────────────────────────────────
    yield format_sse({
        "type": "status",
        "agent": "planner",
        "message": "Planner creating search queries...",
        "step": 1,
        "total_steps": 4
    })

    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", "Create exactly 3 numbered search queries for this topic. Nothing else."),
        ("human", "{task}")
    ])
    plan = (planner_prompt | llm | StrOutputParser()).invoke({"task": task})

    yield format_sse({
        "type": "result",
        "agent": "planner",
        "message": "Planner done",
        "data": plan[:200],
        "step": 1,
        "total_steps": 4
    })

    # ── Event 3: researcher ──────────────────────────────────
    yield format_sse({
        "type": "status",
        "agent": "researcher",
        "message": "Researcher searching the web...",
        "step": 2,
        "total_steps": 4
    })

    time.sleep(1)
    search_results = ""
    try:
        search = DuckDuckGoSearchRun()
        # Get first query
        first_query = [
            l.lstrip("0123456789. ").strip()
            for l in plan.splitlines()
            if l.strip() and l.strip()[0].isdigit()
        ]
        if first_query:
            search_results = search.run(first_query[0])
    except Exception as e:
        search_results = f"Search unavailable: {e}"

    time.sleep(1)
    researcher_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract and synthesise key facts from search results. Be specific."),
        ("human", "Topic: {task}\nSearch results: {results}\nKey findings:")
    ])
    findings = (researcher_prompt | llm | StrOutputParser()).invoke({
        "task": task,
        "results": search_results[:2000]
    })

    yield format_sse({
        "type": "result",
        "agent": "researcher",
        "message": "Researcher done",
        "data": findings[:200],
        "step": 2,
        "total_steps": 4
    })

    # ── Event 4: writer ──────────────────────────────────────
    yield format_sse({
        "type": "status",
        "agent": "writer",
        "message": "Writer drafting report...",
        "step": 3,
        "total_steps": 4
    })

    time.sleep(1)
    writer_prompt = ChatPromptTemplate.from_messages([
        ("system", """Write a structured research report.
Format: # Title, ## Summary (2-3 sentences), ## Key Findings (3 bullets), ## Conclusion"""),
        ("human", "Topic: {task}\nFindings: {findings}\nWrite report:")
    ])
    report = (writer_prompt | llm | StrOutputParser()).invoke({
        "task": task,
        "findings": findings
    })

    # ── Event 5: complete ─────────────────────────────────────
    yield format_sse({
        "type": "complete",
        "agent": "writer",
        "message": "Research complete",
        "report": report,
        "step": 4,
        "total_steps": 4
    })


# ── Routes ───────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Streaming Research API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/research/stream")
async def stream_research(request: ResearchRequest):
    """
    Stream research progress via Server-Sent Events.

    Returns a stream of JSON events:
    - type: 'status' — agent is starting a step
    - type: 'result' — agent completed a step with output preview
    - type: 'complete' — full report ready

    Each event: data: {"type": "...", "agent": "...", "message": "..."}
    """
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty")

    return StreamingResponse(
        run_agent_streaming(request.task),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # prevents nginx buffering the stream
        }
    )


@app.post("/research/stream/test")
async def stream_test(request: ResearchRequest):
    """
    Test streaming without real agent — instant fake events.
    Use this to verify SSE is working before testing the real agent.
    """
    async def fake_stream():
        events = [
            {"type": "status", "agent": "planner", "message": "Planner running..."},
            {"type": "status", "agent": "researcher", "message": "Researcher searching..."},
            {"type": "status", "agent": "writer", "message": "Writer drafting..."},
            {"type": "complete", "agent": "system",
             "message": "Done", "report": f"Test report for: {request.task}"},
        ]
        for event in events:
            yield format_sse(event)
            time.sleep(0.5)

    return StreamingResponse(
        fake_stream(),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    uvicorn.run("03_streaming:app", host="0.0.0.0", port=8000, reload=True)
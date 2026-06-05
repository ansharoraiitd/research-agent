# WHAT THIS DOES:
# Middleware runs on every request — logs timing and adds headers.
# Exception handlers catch errors globally — returns clean JSON errors.
# Rate limiting prevents abuse — max requests per minute.
# CORS headers allow browser frontends to call this API.
# These are the production-grade patterns that separate demos from real products.

import sys
import os
import time
import uuid
import logging
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# ── Logging setup ─────────────────────────────────────────────
# Use Python's built-in logging instead of print()
# In production logs go to a file or logging service
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────────────────
app = FastAPI(
    title="Production Research API",
    description="Multi-agent research API with production-grade error handling",
    version="4.0.0"
)

# ── CORS Middleware ──────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing
# Without this, browsers block requests from your frontend to this API
# allow_origins=["*"] means any frontend can call this API
# In production you'd restrict to your actual frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request timing middleware ─────────────────────────────────
# This runs on EVERY request automatically
# Logs how long each request took and adds a request ID
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """
    Runs before and after every request.
    Logs timing and adds a unique request ID to every response.
    """
    # Before request
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    logger.info(f"[{request_id}] {request.method} {request.url.path} - started")

    # Run the actual route function
    response = await call_next(request)

    # After request
    elapsed = round((time.time() - start_time) * 1000, 1)
    logger.info(f"[{request_id}] {request.method} {request.url.path} - {response.status_code} in {elapsed}ms")

    # Add headers to every response
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(elapsed)

    return response

# ── Simple rate limiter ──────────────────────────────────────
# Tracks requests per IP address
# In production use Redis for this instead of a dict
request_counts = defaultdict(list)
RATE_LIMIT = 10  # max requests per minute per IP

def check_rate_limit(client_ip: str):
    """Block IPs that exceed RATE_LIMIT requests per minute."""
    now = time.time()
    minute_ago = now - 60

    # Remove old requests outside the 1-minute window
    request_counts[client_ip] = [
        t for t in request_counts[client_ip]
        if t > minute_ago
    ]

    if len(request_counts[client_ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT} requests per minute."
        )

    request_counts[client_ip].append(now)

# ── Global exception handlers ─────────────────────────────────
# These catch errors that reach the top level
# Instead of a raw 500 error, clients get a clean JSON response

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle all HTTPExceptions with a consistent JSON format."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} | Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch any unexpected exception — never let a raw crash reach the client."""
    logger.error(f"Unhandled error: {exc} | Path: {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "detail": "Internal server error — the team has been notified",
            "path": str(request.url.path)
        }
    )

# ── Models ───────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    task: str
    mode: Optional[str] = "quick"  # "quick" or "full"

class ResearchResponse(BaseModel):
    task: str
    report: str
    mode: str
    duration_ms: float
    request_id: Optional[str] = None

# ── Routes ───────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Production Research API",
        "version": "4.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health(request: Request):
    client_ip = request.client.host
    logger.info(f"Health check from {client_ip}")
    return {
        "status": "healthy",
        "uptime": "running",
        "rate_limit": f"{RATE_LIMIT} req/min"
    }

@app.post("/research", response_model=ResearchResponse)
async def research(request: Request, body: ResearchRequest):
    """
    Run research with full error handling and rate limiting.
    mode='quick' uses model knowledge (fast)
    mode='full' uses web search (slow, more current)
    """
    # Rate limiting check
    client_ip = request.client.host
    check_rate_limit(client_ip)

    # Input validation
    task = body.task.strip()
    if not task:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Task cannot be empty"
        )
    if len(task) > 300:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task too long ({len(task)} chars). Maximum is 300."
        )
    if body.mode not in ["quick", "full"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mode must be 'quick' or 'full'"
        )

    start = time.time()
    logger.info(f"Research started | task={task[:50]} | mode={body.mode}")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

        if body.mode == "quick":
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Write a structured research report.
Format: # Title, ## Summary, ## Key Findings (3 bullets), ## Conclusion"""),
                ("human", "Research report on: {task}")
            ])
            report = (prompt | llm | StrOutputParser()).invoke({"task": task})

        else:  # full mode with web search
            from langchain_community.tools import DuckDuckGoSearchRun
            try:
                search = DuckDuckGoSearchRun()
                results = search.run(task)
            except Exception as search_err:
                logger.warning(f"Search failed, falling back to model: {search_err}")
                results = "Web search unavailable"

            prompt = ChatPromptTemplate.from_messages([
                ("system", """Write a structured research report using these search results.
Format: # Title, ## Summary, ## Key Findings (3 bullets), ## Conclusion"""),
                ("human", "Task: {task}\nSearch results: {results}\nWrite report:")
            ])
            report = (prompt | llm | StrOutputParser()).invoke({
                "task": task,
                "results": results[:3000]
            })

        duration_ms = round((time.time() - start) * 1000, 1)
        logger.info(f"Research complete | {duration_ms}ms")

        # Get request ID from response headers if available
        return ResearchResponse(
            task=task,
            report=report,
            mode=body.mode,
            duration_ms=duration_ms
        )

    except HTTPException:
        raise  # re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Research pipeline failed. Please try again."
        )


# ── Test error handling ──────────────────────────────────────
@app.get("/error/test")
def test_error():
    """Intentionally raises an error to test exception handling."""
    raise ValueError("This is a test error")

@app.get("/error/http")
def test_http_error():
    """Raises an HTTP 404 to test HTTP exception handler."""
    raise HTTPException(status_code=404, detail="This resource doesn't exist")


if __name__ == "__main__":
    uvicorn.run("04_production:app", host="0.0.0.0", port=8000, reload=True)
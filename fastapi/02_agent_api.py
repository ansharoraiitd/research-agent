#WHAT THIS DOES:
#FastAPI server with real multi-agent pipeline connected.
#POST /research triggers the full planner-researcher-writer-critic pipeline.
#async def means the server doesn't block while the agent is running.
#This is our agent accessible via HTTP for the first time.

import os 
import sys 
import time 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional 
import uvicorn

#Now we import the multi-agent pipeline from earlier 
from multi_agent.multi_agent_system import system as agent_pipeline 

app = FastAPI(
    title="Multi-Agent Research API",
    description="""A production REST API wrapping a 4-agent research system.
    
    Agents:
    -**Planner**: Creates focused search queries
    -**Researcher**: Searches web, synthesises findings
    -**Writer**: Produces structured report
    -**Critic**: Reviews quality, requests revisions if needed""",
    version="2.0.0"
)

#Request and Response models:
class ResearchRequest(BaseModel):
    task: str
    max_revisions: Optional[int]=2

    class Config:
        json_scheme_extra={
            "example": {
                "task": "How is LangGraph being used in production AI in 2025?",
                "max_revisions": 2
            }
        }


class ResearchResponse(BaseModel):
    task: str 
    report: str 
    approved: bool 
    revisions: int 
    status: str 


class ErrorResponse(BaseModel):
    error: str 
    detail: str 

#Simple in-memory job tracking: 
#In production, this would be Redis or a database.
#For now, we will use a dict to track running jobs.
jobs={}


#Routes: 
@app.get("/")
def root():
    return {
        "message": "Multi-Agent Research API",
        "docs": "/docs",
        "endpoints": ["/health", "/research", "/research/quick"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "agents": 4,
        "version": "2.0.0"
    }


@app.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    """
    Run the full 4-agent research pipeline.
    
    - Planner creates search queries
    - Researcher searches web and synthesises findings
    - Writer produces structured report
    - Critic reviews and may request revisions
    
    Takes 20-40 seconds depending on web search speed.
    """
    #Also we need to validate input:
    if not request.task.strip():
        raise HTTPException(
            status_code=400,
            detail="Task cannot be empty."
        )

    if len(request.task) > 500:
        raise HTTPException(
            status_code=400,
            detail="Task is too long - maximum 500 characters"
        )    

    
    try:
        print(f"\n[API] Starting research: {request.task[:50]}...")
        start_time = time.time()

        #Run the multi-agent pipeline
        #This is the agent we built earlier 
        result = agent_pipeline.invoke({
            "task": request.task,
            "plan": "",
            "findings": "",
            "report": "",
            "critique": "",
            "approved": False,
            "revision_count": 0
        })

        elapsed = round(time.time() - start_time, 1)
        print(f"[API] Completed in {elapsed}s")

        return ResearchResponse(
            task=result["task"],
            report=result["report"],
            approved=result["approved"],
            revisions=result["revision_count"],
            status=f"completed in {elapsed}s"
        )

    except Exception as e:
        print(f"[API] Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent pipeline failed: {str(e)}"
        )


@app.post("/research/quick")
async def quick_research(request: ResearchRequest):
    """
    Quick research — skips web search, uses model knowledge only.
    Faster (5-10s) but less current information.
    Good for testing the API without burning search requests.        
    """
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from dotenv import load_dotenv
    load_dotenv()

    llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
    prompt=ChatPromptTemplate.from_messages([
        ("system", """You are a research writer. Write a structured report.
        Format: # Title, ## Summary (2-3 sentences), ## Key Findings (3 bullets), ## Conclusion"""),
        ("human", "Write a research report on: {task}")
    ])
    chain = prompt | llm | StrOutputParser()
    report = chain.invoke({"task": request.task})

    return ResearchResponse(
        task=request.task,
        report=report,
        approved=True,
        revisions=0,
        status="quick mode - no web search"
    )


#Now we run the server 
if __name__ == "__main__":
    uvicorn.run("02_agent_api:app", host="0.0.0.0", port=8000, reload=True)





        

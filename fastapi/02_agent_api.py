#Section 1: imports
import sys
import os 
import time 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 
from typing import Optional 
import uvicorn 
from dotenv import load_dotenv

load_dotenv()

from multi_agent.multi_agent_system import system as agent_pipeline


#Section 2: creating the app and the models

app = FastAPI(
    title="Multi-Agent Research API",
    description="4-agent pipeline: Planner -> Researcher -> Writer -> Critic",
    version="2.0.0"
)

class ResearchRequest(BaseModel):
    task: str 
    mode: Optional[str] = "quick"

    class Config:
        json_scheme_extra = {
            "example": {
                "task": "How is LangGraph used in production AI in 2026?",
                "mode": "quick"
            }
        }

class ResearchResponse(BaseModel):
    task: str 
    report: str 
    approved: bool 
    revisions: int 
    duration_seconds: float


#Section 3 - routes 

@app.get("/")
def root():
    return {
        "message": "Multi-Agent Research API",
        "docs": "/docs",
        "endpoints": ["/health", "/research"]
    }

@app.get("/health") 
def health():
    return {"status": "healthy", "agents": 4, "version": "2.0.0"}


@app.post("/research", response_model=ResearchResponse)    
async def run_research(request: ResearchRequest):
    """
    Run the research pipeline.
    mode='quick' - fast, uses model knowledge (5-10s)
    mode='full' - slow, uses web search (30-40s)
    """
    #validate inputs 
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty")

    if len(request.task) > 300:
        raise HTTPException(
            status_code=400,
            detail=f"Task too long ({len(request.task)} chars). max 300."
        )

    if request.mode not in ["quick", "full"]:
        raise HTTPException(
            status_code=400,
            detail="mode must be 'quick' or 'full'"
        )        

    start = time.time()

    try:
        if request.mode == "quick":
            #Fast path - LLM answers from training knowledge
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.prompts import ChatPromptTemplate 
            from langchain_core.output_parsers import StrOutputParser

            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
            prompt=ChatPromptTemplate.from_messages([
                ("system", """Write a structured research report.
                Format: #Title, ##Summary (2-3 sentences), ###Key Findings (3 bullet points), ##Conclusion (1 sentence)"""),
                ("human", "Write a research report on: {task}")
            ])
            report = (prompt | llm | StrOutputParser()).invoke(
                {"task": request.task}
            )
            approved = True 
            revisions = 0

        else:
            #Full path - real 4-agent pipeline with web search 
            result = agent_pipeline.invoke({
                "task": request.task,
                "plan": "",
                "findings": "",
                "report": "",
                "critique": "",
                "approved": False,
                "revision_count": 0
            })    
            report = result["report"]
            approved = result["approved"]
            revisions = result["revision_count"]

        duration = round(time.time() - start, 1)

        return ResearchResponse(
            task=request.task,
            report=report,
            approved=approved,
            revisions=revisions,
            duration_seconds=duration
        )    

    except HTTPException:
        raise 
    except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline failed: {str(e)}"
            )    


if __name__ == "__main__":
    uvicorn.run("02_agent_api:app", host="0.0.0.0", port=8000, reload=True)            




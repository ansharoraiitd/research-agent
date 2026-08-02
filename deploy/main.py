import os
import time 
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel 
import uvicorn 
from dotenv import load_dotenv 

load_dotenv()

app = FastAPI(
    title="Research Agent API",
    description="Multi-agent AI research system",
    version="1.0.0"
)

class ResearchRequest(BaseModel):
    task: str 

class ResearchResponse(BaseModel):
    task: str 
    report: str 
    duration_seconds: float 

@app.get("/")
def root():
    return {"message": "Research Agent API is live", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty") 

    start = time.time()

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Write a structured research report.
        Format: # Title, ## Summary, ## Key Findings (3 bullets), ## Conclusion"""),
        ("human", "Research report on: {task}")
    ])

    report = (prompt | llm | StrOutputParser()).invoke(
        {"task": request.task}
    )
    duration = round(time.time() - start, 1)

    return ResearchResponse(task=request.task, report=report, duration_seconds=duration)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
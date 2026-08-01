#Section 1: imports 

from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
from typing import Optional 
import uvicorn 

#Section 2: creating the app

app = FastAPI(
    title="Research Agent API",
    description="My multi-agent research system as a REST API.",
    version="1.0.0"
)

#Section 3: data models 

class QuestionRequest(BaseModel):
    question: str
    max_length: Optional[int] = 500

class AnswerResponse(BaseModel):
    question: str 
    answer: str 
    word_count: int     

#Section  4: routes 

@app.get("/")
def root():
    return{"message": "API is running", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    #validate question not empty 
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    answer = f"You asked: '{request.question}'. Agent will answer this soon."

    return AnswerResponse(
        question=request.question,
        answer=answer,
        word_count=len(answer.split())
    )    

#Section 5: run the server 

if __name__ == "__main__":
    uvicorn.run("01_basics:app", host="0.0.0.0", port=8000, reload=True)    
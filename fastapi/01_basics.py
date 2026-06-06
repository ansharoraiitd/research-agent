#WHAT THIS DOES:
#FastAPI wraps Python functions as web API endpoints.
#A route is a URL + function pair - request comes in, function runs, reponse goes out.
#Pydantic models define the shape of request and response data.
#Uvicorn is the web server that runs FastAPI.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 
from typing import Optional 
import uvicorn 

#First we will create the FastAPI app - this would be our web server.
app = FastAPI(
    title="Research Agent API",
    description="Multi-agent research system exposed as a REST API",
    version="1.0.0"
)

#Next, we would create request and response models.
#These pydantic models define the shape of the data coming in and going out.
#FastAPI automatically validates incoming requests against these models.
#If a required field is missing, FastAPI returns a clear error - no code needed.

class QuestionRequest(BaseModel):
    question: str                      #required - user's question
    max_length: Optional[int] = 500    #optional - default 500


class AnswerResponse(BaseModel):
    question: str     #echo back the question
    answer: str       #the answer
    word_count: int   #words in the answer


class HealthResponse(BaseModel):
    status: str
    message: str



#Routes: 
#Each route is a URL + HTTP method + function.
#The decorator (@app.get, @app.post) registers the route.
#FastAPI reads the function's return type and validates the response.


@app.get("/")
def root():
    """Root endpoint - confirms API is running."""
    return {"message": "Research agent API is running", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check - used by monitoring systems to verify the API is up."""
    return HealthResponse(
        status="healthy",
        message="All systems operational"
    )


@app.get("/about")
def about():
    """Information about this API."""
    return {
        "name": "Research Agent API",
        "agents": ["planner", "researcher", "writer", "critic"],
        "capabilities": ["web_search", "report_generation"],
        "version": "1.0.0"
    }    


@app.post("/ask", response_model = AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Answer a question — placeholder for the real agent.
    Later, this will call the actual multi-agent pipeline.
    """
    #Validate the question isn't empty:
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    #For now, we will have a simple echo response.
    #Later we will call multi_agent_system.pipeline.ivnoke(...)    
    answer = f"You asked : '{request.question}'. The agent pipeline will answer this later."

    return AnswerResponse(
        question=request.question,
        answer=answer,
        word_count=len(answer.split())
    )


@app.post("/echo")
def echo(request: QuestionRequest):
    """
    Echo endpoint — useful for testing that requests are received correctly.
    Returns exactly what was sent.
    """
    return {
        "received": request.question,
        "length": len(request.question),
        "max_length_setting": request.max_length 
    }    


#Now we would run the server:
#Uvicorn is the ASGI web server that runs FastAPI.
#host="0.0.0.0" means accessible from any network interface.
#port = 8000 is the standard dev port. 
#reload = True means the server restarts when we save the file - great for dev.

if __name__ == "__main__":
    uvicorn.run("01_basics:app", host="0.0.0.0", port=8000, reload=True)    
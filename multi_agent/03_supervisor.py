#WHAT THIS DOES:
#Supervisor pattern - one agent decides which specialist to call next.
#Supervisor reads state after every agent, then routes to next agent or END.
#More flexible than linear pipeline - can skip, repeat or reorder agents.
#Most common pattern in production multi-agent systems.

import time
import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv 

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

#Defining STATE:
class SupervisedState(TypedDict):
    task: str
    plan: str
    findings: str
    report: str
    next_agent: str    # supervisor writes this to control routing
    iteration: int     # tracks how many times supervisor has run


#Supervisor agent: 
#The supervisor's job: read state, decide who to call next.
#It returns next_agent which the conditional edge reads.

def supervisor_node(state: SupervisedState) -> dict:
    """
    Supervisor reads the current state and decides what to do next.
    Returns next_agent: 'planner', 'researcher', 'writer', or 'FINISH'
    """
    iteration = state.get("iteration", 0) + 1
    print(f"\n[Supervisor] Iteration {iteration} - assessing state...")

    #Build a status report of what's been done so far
    status = f"""Current state:
    - Task: {state['task'][:50]}
    - Plan: {'Done' if state.get("plan") else 'NOT DONE'}
    - Findings: {'Done' if state.get("findings") else 'NOT DONE'}
    - Report: {'Done' if state.get("report") else 'NOT DONE'}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research supervisor managing a team.
        You must decide which agent to call next based on current progress.
        Rules:
        -If plan is not done: respond with exactly 'planner'
        -If plan is done but findings are not done: respond with exactly 'researcher'
        -If findings are done but report is not done: respond with exactly 'writer'
        -If report is done: respond with exactly 'FINISH'

        Respond with ONE word only: planner, researcher, writer, or FINISH
        """),
        ("human", status)
    ])

    chain = prompt | model | StrOutputParser()
    decision=chain.invoke({}).strip().lower()

    #Clean up the decision in case model adds extra test
    if "planner" in decision:
        next_agent = "planner"
    elif "researcher" in decision:
        next_agent = "researcher"
    elif "writer" in decision:
        next_agent = "writer"
    else:
        next_agent = "FINISH"

    print(f"[Supervisor] Decison: call '{next_agent}'") 
    return {"next_agent": next_agent, "iteration": iteration}


#Specialist agent nodes:
#Same as before, but now called BY the supervisor (NOT in fixed sequence)

def planner_node(state: SupervisedState) -> dict:
    print("[Planner] Creating research plan...")
    time.sleep(1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research planner. Create exactly 3 numbered research questions. Nothing else."),
        ("human", "Research questions for: {task}")
    ])
    chain = prompt | model | StrOutputParser()
    plan = chain.invoke({"task": state["task"]})
    print("[Planner] Done")
    return {"plan": plan}


def researcher_node(state: SupervisedState) -> dict:
    print("[Researcher] Answering research questions...")
    time.sleep(1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical researcher. Answer each question with specific facts and examples."),
        ("human", "Answer these questions:\n{plan}\n\nContext: {task}")
    ])
    chain = prompt | model | StrOutputParser()
    findings = chain.invoke({"plan": state["plan"], "task": state["task"]})
    print("[Researcher] Done")
    return {"findings": findings}


def writer_node(state: SupervisedState) -> dict:
    print("[Writer] Writing final report...")
    time.sleep(1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical writer. Write a structured report.
        Format: # Title, ## Summary (2 sentences), ## Key Findings (3 bullets), ## Conclusion (1 sentence)"""),
        ("human", "Write report from:\n{findings}\n\nTopic: {task}")
    ])
    chain = prompt | model | StrOutputParser()
    report = chain.invoke({"findings": state["findings"], "task": state["task"]})
    print("[Writer] Done")
    return {"report": report}


#Routing function:
#This is the conditional edge - reads next_agent from state and returns the next node to go to

def route_next(state: SupervisedState) -> str:
    """Read next_agent from state and return the node name."""
    return state["next_agent"]


#Build the graph:
builder = StateGraph(SupervisedState)

#Add all nodes:
builder.add_node("supervisor", supervisor_node)
builder.add_node("planner", planner_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)

#Start always goes to supervisor:
builder.add_edge(START, "supervisor")

#After every specialist agent, go back to supervisor
#Supervisor then decides what happens next
builder.add_edge("planner", "supervisor")
builder.add_edge("researcher", "supervisor")
builder.add_edge("writer", "supervisor")

#Supervisor has a conditional edge - routes based on next_agent
builder.add_conditional_edges(
    "supervisor",
    route_next,
    {
        "planner": "planner",
        "researcher": "researcher",
        "writer": "writer",
        "FINISH": END
    }
)

pipeline = builder.compile()

#Running this system:
def run(task: str):
    print("\n"+"="*70)
    print(f"TASK: {task}")
    print("="*70)

    result = pipeline.invoke({
        "task": task,
        "plan": "",
        "findings": "",
        "report": "",
        "next_agent": "",
        "iteration": 0
    })

    print("\n"+"="*70)
    print("FINAL REPORT")
    print("="*70)
    print(result["report"])
    print(f"\nSupervisor ran {result['iteration']} times")


run("How does ChromaDB enable semantic search in RAG systems?")    



    

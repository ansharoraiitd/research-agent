#WHAT THIS DOES:
#Three agents connected via LangGraph shared state.
#Each agent is a node - reads from state, adds to state.
#State flows through graph automatically - no manual passing.
#This is how production multi-agent systems are built.

import time
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

#The shared state:
#This TypedDict flows through every node in the graph.
#Each agent reads what it needs and writes what it produces.
#By the end, state contains everything - task, plan, findings, report.

class ResearchState (TypedDict):
    task: str          # the original research task — set at start
    plan: str          # planner's output — 3 research questions
    findings: str      # researcher's output — answers to questions
    report: str        # writer's output — final structured report

#Agent nodes:
#Each node is a function that takes state and returns state updates.
#The return dict only needs the fields this node changed.

def planner_node(state: ResearchState) -> dict:
    """
    Planner agent — reads task, writes plan.
    Job: break the task into 3 clear research questions.
    """
    print("[Planner] Creating research plan...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research planner. Your ONLY job is
        to create exactly 3 focused research questions for the given topic.
        Output format: numbered list, nothing else."""),
        ("human", "Create 3 research questions for: {task}")
    ])

    chain = prompt | model | StrOutputParser()
    plan = chain.invoke({"task": state['task']})

    print(f"[Planner] Done - {len(plan.splitlines())} questions created")
    return {"plan": plan}


def researcher_node(state: ResearchState) -> dict:
    """
    Researcher agent — reads plan, writes findings.
    Job: answer the research questions with specific facts.
    """
    print("[Researcher] Investigating questions...")
    time.sleep(1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical researcher. Your ONLY job
        is to provide detailed, factual answers. Be specific.
        Use real examples and concrete details."""),
        ("human", """Answer these research questions thoroughly:
        {plan}
        Context topic: {task}""")
    ])

    chain = prompt | model | StrOutputParser()
    findings=chain.invoke({
        "plan": state['plan'],
        "task": state['task']
    })
    print(f"[Researcher] Done - findings: {len(findings)} chars")
    return {"findings": findings}


def writer_node(state: ResearchState) -> dict:
    """
    Writer agent — reads findings, writes report.
    Job: turn research findings into a clean structured report.
    """
    print("[Writer] Writing final report...")
    time.sleep(1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical writer. Your ONLY job is
        to write a clean structured report from research findings.
        Use this exact format:
        # [Title]
        ## Executive Summary
        [2 sentences]
        ## Key Findings
        - [finding 1]
        - [finding 2]
        - [finding 3]
        ## Conclusion
        [1 sentence]"""),
        ("human", """Write a report using these findings:
        {findings}
        Original topic: {task}""")
    ])

    chain = prompt | model | StrOutputParser()
    report = chain.invoke({
        "findings": state['findings'],
        "task": state['task']
    })

    print(f"[Writer] Done - report : {len(report)} chars")
    return {"report": report}


#Building the graph: 
#Wire the 3 agent nodes together in sequence.
#State flows: START -> planner -> researcher -> writer -> END

builder = StateGraph(ResearchState)
builder.add_node("planner", planner_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", END)

pipeline = builder.compile()

#Running our system:
def run_pipeline(task: str):
    print("\n"+"="*70)
    print(f"Task: {task}")
    print("="*70)

    result = pipeline.invoke({"task": task})

    print("\n"+"="*70)
    print("FINAL REPORT")
    print("="*70)
    print(result['report'])

    print("\n"+"="*70)
    print("STATE AFTER COMPLETION")
    print("="*70)

    print(f"Task     : {result['task'][:60]}...")
    print(f"Plan     : {result['plan'][:60]}...")
    print(f"Findings : {result['findings'][:60]}...")
    print(f"Report   : {result['report'][:60]}...")
    print("\nAll four fields populated — state grew as agents ran")


run_pipeline("The role of RAG in enterprise AI applications")    




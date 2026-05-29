#WHAT THIS DOES:
#Multi-agent pipeline where the researcher has web search as a tool.
#Planner creates questions, Researcher searches web for answers, writer produces a structured report from real search results.
#This is real data flowing through a multi-agent system.

import time
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv 

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
search_tool = DuckDuckGoSearchRun()

#Defining State:
class PipelineState(TypedDict):
    task: str
    plan: str               # planner's research questions
    raw_search: str         # raw web search results
    findings: str           # researcher's synthesis of search results
    report: str             # writer's final structured report

'''
#Helper: 
def clean_text(content) -> str:
    """Extract clean text from Gemini response."""
    if isinstance(content, list):
        return " ".join(
            p["text"] for p in content
            if isinstance(p, dict) and "text" in p
        )
    return str(content)
'''

#Node 1: Planner 
def planner_node(state: PipelineState) -> dict:
    """Create 3 specific search queries for the task."""
    print("[1/3 Planner] Creating search queries...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research planner. Create exactly 3
        specific web search queries for the given topic.
        Each query should find different useful information.
        Output: numbered list of 3 queries only."""),
        ("human", "Create 3 web search queries for: {task}")
    ])

    chain = prompt | model | StrOutputParser()
    plan = chain.invoke({"task": state['task']})
    print(f"[Planner] Created {len(plan.splitlines())} queries")
    return {"plan": plan}


#Node 2: Researcher with web search
def researcher_node(state: PipelineState) -> dict:
    """
    Search the web using the planner's queries.
    Then synthesise the results into clean findings.
    Two steps: search → synthesise.
    """
    print("[2/3 Researcher] Searching the web...")
    time.sleep(1)

    #Step A: extracting the query and searching
    # In a more advanced version, we would search all 3 queries
    queries = [
        line.strip()
        for line in state['plan'].splitlines()
        if line.strip() and line.strip()[0].isdigit()
    ]

    all_results = []
    for i, query in enumerate(queries[:2]):    #searching first 2 queries
        #Removing the number prefix (e.g. "1. What is the....?")
        clean_query = query.lstrip("0123456789. ").strip()
        if clean_query:
            print(f"[Researcher] Searching: {clean_query[:50]}...")
            try:
                result = search_tool.run(clean_query)
                all_results.append(f"Query: {clean_query}\nResults: {result}")
                time.sleep(1)
            except Exception as e:
                all_results.append(f"Query: {clean_query}\nResults: Search failed - {e}")  

    raw_search = "\n\n---\n\n".join(all_results) 
    #Step B: synthesising the raw search results into clean findings
    print("[Researcher] Synthesising earch results...")   
    time.sleep(1)

    synthesise_prompt = ChatPromptTemplate([
        ("system", """You are a research analyst. Extract and synthesise
        the most important factual information from these search results.
        Be specific. Include real facts, numbers, and examples where available.
        Organise by the research questions asked."""),
        ("human", """Research task: {task}
        Search results: {raw_search}
        Synthesise the key findings:""")
    ])       

    synth_chain = synthesise_prompt | model | StrOutputParser()
    findings = synth_chain.invoke({
        "task": state['task'],
        "raw_search": raw_search
    })

    print(f"[Researcher] Findings: {len(findings)} chars")
    return {"raw_search": raw_search, "findings": findings}


#Node 3: Writer
def writer_node(state: PipelineState) -> dict:
    """Write a structured report from the researcher's findings."""
    print("[3/3 Writer] Writing final report...")
    time.sleep(1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical writer. Write a clear structured report.
        Use this exact format:
        # [Title]
        ## Summary
        [2-3 sentences overview]
        ## Key Findings
        - [specific finding with detail]
        - [specific finding with detail]
        - [specific finding with detail]
        ## Conclusion
        [1-2 sentences]"""),
        ("human", """Write a report on: {task}
        Based on these research findings: {findings}""")
    ])

    chain = prompt | model | StrOutputParser()
    report = chain.invoke({
        "task": state["task"],
        "findings": state["findings"]
    })

    print(f"[Writer] Report: {len(report)} chars")
    return {"report": report}


#Building the graph:

builder = StateGraph(PipelineState)

builder.add_node("planner", planner_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", END)

pipeline = builder.compile()


#Running this system:
def run(task: str):
    print("\n"+"="*70)
    print(f"Task: {task}")
    print("="*70)

    result = pipeline.invoke({
        "task": task,
        "plan": "",
        "raw_search": "",
        "findings": "",
        "report": ""
    })

    print("\n"+"="*70)
    print("FINAL REPORT")
    print("="*70)

    print(result['report'])

    print("\n"+"="*70)
    print("RAW SEARCH PREVIEW")
    print("="*70)
    print(result['raw_search'][:300] + "...")
    print("\nReport was grounded in real web search results above")


run("Latest developments in LangGraph for production AI agents in 2025")






    

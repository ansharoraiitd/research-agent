#WHAT THIS DOES:
#Complete multi agent research system with 4 specialist agents.
#Planner -> Researcher (web search) -> Writer -> Critic (quality check)
#Critic can send report back to writer for revision - quality control loop.
#This is the most complete agentic system built by me so far.

import time
import sys 
import os 
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
class SystemState(TypedDict):
    task: str
    plan: str
    findings: str 
    report: str 
    critique: str          # critic's feedback on the report
    approved: bool         # True = report passed quality check
    revision_count: int     # how many times writer has revised


#Node 1: Planner 
def planner_node(state: SystemState) -> dict:
    print("\n[1/4 Planner] Creating research plan...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research planner. Create exactly 3
        specific, focused search queries. Output numbered list only."""),
        ("human", "Create 3 search queries for: {task}")
    ])
    chain = prompt | model | StrOutputParser()
    plan = chain.invoke({"task": state["task"]})
    print(f"[Planner] Done")
    return {"plan": plan}    

#Node 2: Researcher 
def researcher_node(state: SystemState) -> dict:
    print("[2/4 Researcher] Searching web + synthesising...")
    time.sleep(1)

    # Extract and run first search query
    lines = [l.strip() for l in state["plan"].splitlines()
             if l.strip() and l.strip()[0].isdigit()]

    search_results = []
    for line in lines[:2]:
        query = line.lstrip("0123456789. ").strip()
        if query:
            try:
                result = search_tool.run(query)
                search_results.append(f"Search: {query}\n{result}")
                time.sleep(1)
            except Exception:
                pass

    raw = "\n\n".join(search_results) if search_results else "No search results"

    # Synthesise findings from search results
    time.sleep(1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract key factual findings from search results. Be specific."),
        ("human", "Topic: {task}\n\nSearch results:\n{raw}\n\nKey findings:")
    ])
    chain = prompt | model | StrOutputParser()
    findings = chain.invoke({"task": state["task"], "raw": raw})
    print(f"[Researcher] Done")
    return {"findings": findings}


# ── Node 3: Writer ────────────────────────────────────────────
def writer_node(state: SystemState) -> dict:
    revision = state.get("revision_count", 0)
    if revision > 0:
        print(f"[3/4 Writer] Revising report (revision {revision})...")
    else:
        print("[3/4 Writer] Writing initial report...")
    time.sleep(1)

    # Include critique in prompt if this is a revision
    critique_context = ""
    if state.get("critique"):
        critique_context = f"\n\nPrevious critique to address:\n{state['critique']}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical writer. Write a structured report.
        Format:
        # [Title]
        ## Summary
        [2-3 sentences]
        ## Key Findings
        - [finding with specific detail]
        - [finding with specific detail]
        - [finding with specific detail]
        ## Conclusion
        [1-2 sentences]"""),  
        ("human", """Write report on: {task}
        Research findings:
        {findings}{critique_context}""")
    ])

    chain = prompt | model | StrOutputParser()
    report = chain.invoke({
        "task": state["task"],
        "findings": state["findings"],
        "critique_context": critique_context
    })
    print(f"[Writer] Done")
    return {"report": report, "revision_count": revision + 1}


# ── Node 4: Critic ────────────────────────────────────────────
def critic_node(state: SystemState) -> dict:
    print("[4/4 Critic] Reviewing report quality...")
    time.sleep(1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a quality reviewer. Review this report strictly.
        Check: Does it have a title, summary, 3+ key findings, and conclusion?
        Are the findings specific with real details (not vague)?

        Respond with exactly:
        APPROVED - if the report meets all requirements
        REVISION NEEDED: [specific issue] - if it needs improvement"""),
        ("human", """Review this report:
        {report}
        Verdict:""")
    ])

    chain = prompt | model | StrOutputParser()
    verdict = chain.invoke({"report": state["report"]})
    print(f"[Critic] Verdict: {verdict[:60]}")

    approved = verdict.upper().startswith("APPROVED")
    return {"critique": verdict, "approved": approved}


# ── Routing function ──────────────────────────────────────────
def route_after_critic(state: SystemState) -> str:
    """
    If approved or too many revisions — finish.
    Otherwise send back to writer for revision.
    """
    if state.get("approved"):
        print("[Critic] Report APPROVED — finishing")
        return "END"
    if state.get("revision_count", 0) >= 2:
        print("[Critic] Max revisions reached — finishing anyway")
        return "END"
    print("[Critic] Sending back to writer for revision...")
    return "writer"


#Build the graph
builder = StateGraph(SystemState)

builder.add_node("planner", planner_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "critic")

# Conditional edge after critic — approve or revise
builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {"END": END, "writer": "writer"}
)

system = builder.compile()

#Running our system:
def run(task: str):
    print(f"\n{'='*55}")
    print(f"MULTI-AGENT RESEARCH SYSTEM")
    print(f"Task: {task}")
    print("="*55)

    result = system.invoke({
        "task": task,
        "plan": "",
        "findings": "",
        "report": "",
        "critique": "",
        "approved": False,
        "revision_count": 0
    })

    print(f"\n{'='*55}")
    print("FINAL APPROVED REPORT")
    print("="*55)
    print(result["report"])
    print(f"\nRevisions: {result['revision_count']} | "
          f"Approved: {result['approved']}")


run("How is LangGraph being used in production AI systems in 2025?")
#What this does:
#Full research agent - takes a question, searches the web, analyses findings, and produces a structured report.
#Combines LangGraph (orchestration) + tools (web search) + LangChain (prompts and model calls), all the things used earlier.

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
search_tool = DuckDuckGoSearchRun()

# State:

class ResearchState(TypedDict):
    question: str
    search_results: str
    key_finding: str
    report: str

# Helper to extract clean text from Gemini
def extract_text(content) -> str:
    if isinstance(content, list):
        return " ".join(
            p["text"] for p in content
            if isinstance(p, dict) and "text" in p
        )
    return content    


#Node 1: search the web
def search(state: ResearchState) -> dict:
    """Search the web for the research question."""
    print(f"\n[1/3] Searching the web...")
    results = search_tool.run(state["question"])
    return {"search_results": results}


#Node 2: analyse the results
def analyse(state: ResearchState) -> dict:
    """Pull out the key findings from the search results."""
    print(f"\n[2/3] Analyse findings...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research analyst. Extract the
        3-5 most important findings from the search results.
        Be factual and concise. Use bullet points."""),
        ("human", """Research question: {question}
        Search results: {search_results}
        Extract the key findings: """)
    ])

    chain = prompt | model | StrOutputParser()
    findings = chain.invoke({
        "question": state["question"],
        "search_results": state["search_results"]
    })
    return {"key_finding": findings}


#Node 3: format report
def format_report(state: ResearchState) -> dict:
    """Write a clean structured research report."""
    print(f"\n[3/3] Writing report...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional research writer.
        Write a clear, structured research report.
        Format:
        # [Title]
        ## Summary
        [2-3 sentence overview]
        ## Key Findings
        [bullet points]
        ## Conclusion
        [1-2 sentences]"""),
        ("human", """Question: {question}
        Findings: {key_finding}
        Write the report:""")
        ])    

    chain = prompt | model | StrOutputParser()
    report = chain.invoke({
        "question": state["question"],
        "key_finding": state["key_finding"]
        })
    return {"report": report}


#Build the graph: 
builder = StateGraph(ResearchState)

builder.add_node("search", search)
builder.add_node("analyse", analyse)
builder.add_node("format_report", format_report)

builder.add_edge(START, "search")
builder.add_edge("search", "analyse")
builder.add_edge("analyse", "format_report")
builder.add_edge("format_report", END)

agent = builder.compile()

# Run this agent: 

def research(question: str):
    print("\n" + "="*50)
    print(f"Research Question: {question}")
    print("="*50)

    result = agent.invoke({"question": question})

    print("="*50)
    print("RESEARCH REPORT")
    print("="*50)
    print(result['report'])


#Testing with 2 different research questions
research("What is LangGraph and why is it used for AI agents?")
research("What are the main use cases for RAG in enterprise AI?")







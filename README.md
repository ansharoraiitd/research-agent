# Research Agent

An agentic AI research assistant built with LangGraph and LangChain.
Takes a research question, searches the web, analyses findings,
and produces a structured report — autonomously.

## How it works
1. User provides a research question
2. Agent searches the web using DuckDuckGo
3. Model analyses search results and extracts key findings  
4. Model writes a structured research report

The three steps are modelled as nodes in a LangGraph state graph.
State flows through each node — each step reads what the previous
step found and adds its own output to the state.

## Files
| File | What it does |
|------|-------------|
| research_agent.py | Main agent — search → analyse → report |
| 01_langchain_basics.py | LCEL chains and prompt templates |
| 02_tool_use.py | Agent with web search and calculator tools |
| 03_memory.py | Conversation memory with session management |
| 04_langgraph_basics.py | LangGraph conditional routing |

## Tech stack
Python · LangChain · LangGraph · Gemini API · DuckDuckGo Search

## Setup
pip install -r requirements.txt
Add GEMINI_API_KEY to .env file
python research_agent.py

## Example output
Question: What is LangGraph and why is it used for AI agents?

# LangGraph: The Framework for Stateful AI Agents
## Summary
LangGraph is a library for building stateful, multi-actor
applications with LLMs...
## Key Findings
- LangGraph 1.0 released as stable in 2025...
- Used by Uber, LinkedIn, JP Morgan in production...
## Conclusion
LangGraph has become the industry standard...

## Multi-Agent Research System

A production-grade multi-agent pipeline with 4 specialist agents
and a quality control loop.

### Architecture
Planner → Researcher (web search) → Writer → Critic
                                         ↑         |
                                         └─revise──┘ (if needed)

### Agents
| Agent | Job | Tools |
|-------|-----|-------|
| Planner | Creates 3 focused search queries | None |
| Researcher | Searches web, synthesises findings | DuckDuckGo |
| Writer | Writes structured report | None |
| Critic | Reviews quality, approves or requests revision | None |

### Key design decisions
- Critic caps revisions at 2 to prevent infinite loops
- Researcher does two steps: search then synthesise
- State flows through all agents — each reads what previous agents wrote
- Conditional edge after critic — approve or route back to writer

### Files
| File | What it builds |
|------|---------------|
| multi_agent_system.py | Complete 4-agent pipeline |
| 01_why_multi_agent.py | Single vs specialist comparison |
| 02_shared_state.py | Three agents with shared LangGraph state |
| 03_supervisor.py | Supervisor orchestration pattern |
| 04_agents_with_tools.py | Researcher with web search tool |
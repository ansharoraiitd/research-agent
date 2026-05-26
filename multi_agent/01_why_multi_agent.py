#WHAT THIS DOES:
#Shows why multi-agent systems beat single agents for complex tasks.
#One general agent vs 3 specialists doing the same job.
#Each specialist has a focused system prompt for their role only.

import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

TASK = "Analyse the impact of LangGraph on modern AI agent development."

#Approach 1: single general agent
#One agent, one prompt, tries to do everything at once.
print("="*70)
print("APPROACH 1: Single General Agent")
print("="*70)

single_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a general AI assistant.
    Do everything asked of you as best you can."""),
    ("human", """Please do ALL of the following for this topic:
    1. Plan what aspects to cover
    2. Research the key facts
    3. Write a structured analysis

    Topic: {task}""")
])

single_chain = single_agent_prompt | model | StrOutputParser()
single_output = single_chain.invoke({"task": TASK})
print(single_output)

time.sleep(2)

#APPROACH 2: 3 specialist agents
#Each agent has ONE job and a system prompt built for that job only.
print("\n" + "="*70)
print("APPROACH 2: Three specialist agents")
print("="*70)

#Agent 1 - Planner: decides what to research
#System prompt focused entirely on planning and structure
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research planner. Your ONLY job is to
    create a clear research plan. Output exactly 3 research questions
    that need to be answered. Nothing else — just 3 questions."""),
    ("human", "Create a research plan for: {task}")
])

planner_chain = planner_prompt | model | StrOutputParser()
print("\n[Agent 1 - Planner]")
research_plan = planner_chain.invoke({"task": TASK})
print(research_plan)

time.sleep(2)

#Agent 2 - Researcher: answers the research questions
# System prompt focused entirely on factual, detailed research
researcher_prompt = ChatPromptTemplate([
    ("system", """You are a technical researcher. Your ONLY job is to
    provide detailed, factual answers to research questions.
    Be specific. Use real examples. No fluff."""),
    ("human", """Research these questions thoroughly:
    {research_plan}

    Topic context: {task}""")
])

researcher_chain = researcher_prompt | model | StrOutputParser()
print("\n[Agent 2 — Researcher]")
research_findings = researcher_chain.invoke({
    "research_plan": research_plan,
    "task": TASK
})
print(research_findings)

time.sleep(2)

# Agent 3: Writer — turns research into a polished output
# System prompt focused entirely on clear, structured writing
writer_prompt = ChatPromptTemplate([
    ("system", """You are a technical writer. Your ONLY job is to
    write clear, structured analysis from research findings.
    Format: Title, Executive Summary (2 sentences), Key Findings
    (3 bullets), Conclusion (1 sentence). Nothing else."""),
    ("human", """Write a structured analysis using these findings:
    {findings}

    Topic: {task}""")
])

writer_chain = writer_prompt | model | StrOutputParser()
print("\n[Agent 3 — Writer]")
final_output = writer_chain.invoke({
    "findings": research_findings,
    "task": TASK 
})
print(final_output)

print("\n" + "=" * 70)
print("Compare the two approaches above.")
print("The specialist agents produce more focused, higher quality output.")
print("That quality difference is why multi-agent systems exist.")
print("=" * 70)
#What this does:
#LangGraph models agents as graphs- nodes do work, edges connect them.
#State flows through the graph, each node reads it and updates it.
#Conditional edges let the agent decide which path to take.
#This is how production agents handle multi-step reasoning.

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import operator

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

#Step 1: to define a state
#State is a TypedDict - a typed dictionary that flows through every node in the graph.
#Every node can read it and update it.
#This is what makes the agent "stateful"- it remembers what happened in previous nodes as it moves through the graph.

class AgentState(TypedDict):
    question: str          #the user's original question
    needs_search: bool     #does this need web search?
    search_results: str    #results from web search (if used)
    answer: str            #the final str


#Step 2: to define the nodes
#Each node is a python function that takes the current state as input,
#does some work, and then returns a dict of state updates.

def classify_question(state: AgentState) -> dict:
    """
    Node 1: Decide if the question needs web search.
    Updates state with needs_search = True or False.
    """
    question = state["question"]

    response=model.invoke(
        f"""Does this question require searching the web for
        current/recent information? Answer with only YES or 
        NO.

        Question: {question}""" 
    )

    answer_text=response.content
    if isinstance(answer_text, list):
        answer_text= " ".join(
            p["text"] for p in answer_text
            if isinstance(p, dict) and "text" in p
        )

    needs_search = "YES" in answer_text.upper()
    print(f"[classify] needs web search: {needs_search}")

    return {"needs_search": needs_search}


def web_search_node(state: AgentState) -> dict:
    """
    Node 2a: Search the web and store results in state.
    Only runs if needs_search is True.
    """
    print(f"[web_search] Searching for: {state['question']}")
    search = DuckDuckGoSearchRun()
    results = search.run(state["question"])
    return {"search_results": results}


def direct_answer_node(state: AgentState) -> dict:
    """
    Node 2b: Answer directly without web search.
    Only runs if needs_search is False.
    """
    print("[direct_answer] Answering from knowledge...")
    return {"search_results": ""}


def format_response(state: AgentState) -> dict:
    """
    Node 3: Generate the final answer using all state so far.
    This is the last node before END.
    """
    question = state['question']
    search_results = state.get("search_results", "")

    if search_results:
        prompt = f"""Answer this question using the search results below.
        Be concise and clear. 

        Question: {question}
        Search results: {search_results}
        """

    else: 
        prompt = f"""Answer this question concisely and clearly.
        Question: {question}  """


    response = model.invoke(prompt)
    answer = response.content
    if isinstance(answer, list):
        answer = " ".join(
            p["text"] for p in answer
            if isinstance(p, dict) and "text" in p
        )

    return {"answer": answer}


#Step 3: to define routing logic
# this function decided which node to go after classify_question
# it reads the state and returns the name of the next node
# this is a conditional edge - the graph branches based on state

def route_after_classify(state: AgentState) -> str:
    """Return the name of the next node based on state."""
    if state["needs_search"]:
        return "web_search_node"

    else:
        return "direct_answer_node"


#Step 4: to build the graph
#This is where we wire everything together.
#Add nodes first, then add edges to connect them.

graph_builder = StateGraph(AgentState)

#Add all nodes
graph_builder.add_node("classify_question", classify_question)
graph_builder.add_node("web_search_node", web_search_node)
graph_builder.add_node("direct_answer_node", direct_answer_node)
graph_builder.add_node("format_response", format_response)

#Add all edges
graph_builder.add_edge(START, "classify_question")
graph_builder.add_edge("web_search_node", "format_response")
graph_builder.add_edge("direct_answer_node", "format_response")
graph_builder.add_edge("format_response", END)

#Add conditional edge - branches based on route_after_classify
graph_builder.add_conditional_edges(
    "classify_question",       # from this node
    route_after_classify,      # call this function to decide
    {                          # map return values to node names
        "web_search_node": "web_search_node",
        "direct_answer_node": "direct_answer_node"
    }
)

#Compile the graph - make it runnable
graph = graph_builder.compile()


#Step 5: to run the graph
def ask(question: str):
    print("\n" + "="*50)
    print(f"Question: {question}")
    print("="*50)

    # invoke() runs the graph with initial state
    result = graph.invoke({"question": question})
    print(f"\nAnswer:\n{result['answer']}")        # result is also a dict, so we choose ['answer']

#Testing with questions that need different paths
ask("What are the latest AI news from this week?")
ask("What is the difference between supervised and unsupervised learning?")    



  







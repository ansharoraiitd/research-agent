#WHAT THIS DOES:
#Tools are python functions that the model can call to take actions.
#The model decides which tool to use based on the user's question.
#This is what makes the difference b/w a chatbot and an agent.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite")

#Step 1 : defining my tools
#The @tool decorator turns a normal python function into a tool the model can call.
#The docstring is critical - the model reads it to decide when to use this tool. So we have to write clear docstrings.

@tool
def web_search(query: str) -> str:
    """Search the web for current information about a topic.
    Use this when you need up-to-date facts, recent news, or
    information you don't already know."""
    search = DuckDuckGoSearchRun()
    result = search.run(query)
    return result


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    Use this for any arithmetic, percentage calculations, or 
    numerical computations. Input must be a valid Python math 
    expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Could not calculate: {e}"

        
@tool
def summarise_text(text: str) -> str:
    """Summarise a long piece of text into 3 bullet points.
    Use this when the user provides text that needs to be 
    condensed."""
    #This tool itself calls the LLM.(tools can do anything)
    from langchain_google_genai import ChatGoogleGenerativeAI
    summariser = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite")
    result = summariser.invoke(
        f"Summarise this in exactly 3 bullet points: \n\n{text}"
    )
    return result.content


#Step 2 : giving tools to the model
#bind_tools() attach our toolbox to the model
#Now, when the model responds, it can choose to call a tool instead of (or before) giving a text answer
tools = [web_search, calculate, summarise_text]
model_with_tools = model.bind_tools(tools)

#Step 3 : a simple agent loop
# This is the core of how agents work:
# 1. User sends message
# 2. Model responds — either with text OR with a tool call
# 3. If tool call: run the tool, send result back to model
# 4. Model gives final answer using the tool result
# 5. Repeat

def run_agent(user_question: str):
    """
    Simple agent loop:
    user -> model -> maybe tool call -> tool result -> model -> answer
    """
    print(f"\nQuestion: {user_question}")
    print("="*50)

    messages = [HumanMessage(content=user_question)]

    # First model call — may return tool calls or direct answer
    response = model_with_tools.invoke(messages)

    #Check if the model wants to use a tool: 
    if response.tool_calls:
        print(f"Agent decided to use tool: {response.tool_calls[0]['name']}")

        #Execute each tool the model requested.
        tool_results=[]
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            #Find and run the right tool.
            tool_map = {t.name: t for t in tools}
            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
                print(f"Tool result preview: {str(result)[:150]}...")
                tool_results.append({
                    "role": "user",
                    "content": str(result),
                    "tool_call_id": tool_call["id"]
                })

        #Send tool results back to model for final answer.
        from langchain_core.messages import ToolMessage
        tool_message = [
            ToolMessage(
                content=r["content"],
                tool_call_id=r["tool_call_id"]
            )
            for r in tool_results
        ]

        #Final model call with tool results as context
        final_response = model_with_tools.invoke(
            messages + [response] + tool_message
        )

        # extract clean text from Gemini's response
        content = final_response.content
        if isinstance(content, list):
            clean = " ".join(
                part["text"] for part in content
                if isinstance(part, dict) and "text" in part
            )
        else:
            clean = content
        print(f"Final Answer:\n{clean}")

    else:
        #Model answered directly without needing a tool.
        print(f"Agent answered directly (no tool needed)")
        content = response.content
        if isinstance(content, list):
            clean = " ".join(
                part["text"] for part in content
                if isinstance(part, dict) and "text" in part
            )
        else:
            clean = content
        print(f"\nAnswer:\n{clean}")

#Step 4 : testing the agent with different questions
#Watch which tool the model picks for each question

print("="*50)
print("Test 1: Question needing web search")
print("="*50)
run_agent("What are the latest developments in LangGraph in 2025?")

print("\n" + "=" * 55)
print("TEST 2: Question needing calculation")
print("=" * 55)
run_agent("If I make 100 API calls per hour and each costs $0.002, what is my daily cost?")

print("\n" + "=" * 55)
print("TEST 3: Question the model can answer directly")
print("=" * 55)
run_agent("What is the difference between RAG and fine-tuning?")











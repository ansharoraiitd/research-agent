#What this does:
#LangChain has built-in memory so we don't manage history manually 
#ChatMessageHistory stores messages.
#RunnableWithMessageHistory wraps any chain and injects history automatically every call.
#Same concept as the first week but LangChain handles the plumbing.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite")

#Section 1 : understanding ChatMessageHistory
#This is just a smarter list of storing messages.
#We can add messages, retrieve them and clear them.
print("="*50)
print("Section 1: ChatMessageHistory basics")
print("="*50)

history = ChatMessageHistory()

#Adding messages manually to understand the structure
history.add_user_message("My name is Ansh and I am learning LangChain.")
history.add_ai_message("Great to meet you Ansh! LangChain is a powerful framework.")
history.add_user_message("What framework am I learning?")

#Printing all stored messages.
print("Stored messages:")
for msg in history.messages:
    print(f"{msg.type.upper()}: {msg.content}")

print(f"Total messages stored: {len(history.messages)}")   

#Secton 2: chain with automatic memory
#MessagesPlaceholder is a slot in the prompt that gets automatically filled with the conversation history.
print("\n" + "="*50)
print("Section 2: Chain with automatic memory")
print("="*50)

#The prompt has a placeholder for history.
#{history} gets filled with past messages automatically
#{input} is the current user message

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI tutor teaching agentic AI.
    You remember everything from the conversation so far."""),
    MessagesPlaceholder(variable_name="history"),
    ("human","{input}")
])

#Building the chain
chain = prompt | model | StrOutputParser()

#Store for multiple sessions - each session_id gets its own history.
#This lets us run multiple separate conversations at once.
store={}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return history for a session, creating it if it doesn't exist."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

#Wrap the chain with automatic history management.
#Now every call automatically saves and injects history.
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='history'
)    


#Section 3 : testing if the memory works
print("\n" + "="*50)
print("SECTION 3: Testing memory across turns")
print("="*50)

#session_id identifies which conversation it belongs to 
# same session_id = same conversation_history

config = {"configurable": {"session_id": "ansh_session_1"}}

# Turn 1:
print("Turn 1:")
reply1=chain_with_memory.invoke(
    {"input": "My name is Ansh and I want to build a research agent."},
    config=config
)
print(f"Bot: {reply1}\n")

#Turn 2 - does it remember the name?
print("Turn 2:")
reply2=chain_with_memory.invoke(
    {"input": "What is my name and what do I want to build?"},
    config=config 
)
print(f"Bot: {reply2}\n")

#Turn 3 - does it remember anything?
print("Turn 3:")
reply3=chain_with_memory.invoke(
    {"input": "Based on what I told you, what should I learn first?"},
    config=config 
)
print(f"Bot: {reply3}\n")

#Showing the stored history to prove its all there:
print("="*50)
print("RAW HISTORY (what's stored after 3 turns):")
print("="*50)

session_history = get_session_history("ansh_session_1")
for msg in session_history.messages:
    preview = msg.content[:80]
    print(f"{msg.type.upper()}: {preview}...")


#Section 4: multiple sessions
#different session_id = completely separate conversation
print("\n" + "="*50)
print("Section 4: Two separate sessions")
print("="*50)

config_session2 = {"configurable": {"session_id": "other_user_session"}}

reply_s2=chain_with_memory.invoke(
    {"input": "What is my name?"},
    config=config_session2
)
print(f"New session (should not know my name): {reply_s2} ")





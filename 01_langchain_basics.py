# WHAT THIS DOES:
# LangChain wraps prompt + model + output into a chain using | operator.
# prompt | model | parser means: fill prompt → send to model → parse output.
# Same as week 1 but cleaner and composable.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# ── Step 1: the model ────────────────────────────────────────
# This wraps Gemini in a LangChain-compatible object.
# Same model you used last week, just wrapped differently.
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# ── Step 2: the prompt template ──────────────────────────────
# Instead of f-strings, LangChain uses template objects.
# {topic} is a variable — you fill it in when you call the chain.
# The advantage: the template is reusable and validatable.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise technical tutor. Answer in 3 bullet points maximum."),
    ("human", "Explain {topic} simply.")
])

# ── Step 3: the output parser ────────────────────────────────
# StrOutputParser just extracts the text string from the model's response.
# Without it you'd get a full AIMessage object back, not a clean string.
parser = StrOutputParser()

# ── Step 4: the chain ────────────────────────────────────────
# This is LCEL — pipe operator connects components left to right.
# Input flows: prompt gets filled → goes to model → model output gets parsed.
chain = prompt | model | parser

# ── Step 5: run it ───────────────────────────────────────────
print("=" * 50)
print("CHAIN 1: Explain a concept")
print("=" * 50)

result = chain.invoke({"topic": "RAG in AI agents"})
print(result)

# Run the same chain with different input — no rewriting needed
print("\n" + "=" * 50)
print("CHAIN 2: Same chain, different topic")
print("=" * 50)

result2 = chain.invoke({"topic": "vector databases"})
print(result2)

# ── Step 6: multiple inputs with batch ───────────────────────
# batch() runs the chain on a list of inputs — useful for processing
# multiple things without writing a for loop
print("\n" + "=" * 50)
print("CHAIN 3: Batch — run on multiple topics at once")
print("=" * 50)

topics = [
    {"topic": "LangChain"},
    {"topic": "embeddings"},
    {"topic": "FastAPI"}
]

results = chain.batch(topics)
for topic, result in zip(topics, results):
    print(f"\n{topic['topic']}:")
    print(result)
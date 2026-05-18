#What this does:
#Langchain wraps promt + model + output into a chain using | operator - the "pipe" operator
#prompt | model | parser means: fill prompt -> send to model -> parse output
#same as week 1 but cleaner and composable

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

#Step 1 : THE MODEL
# This wraps Gemini in a LangChain-compatible object
#Same model I used in week 1, just wrapped differently
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#Step 2 : THE PROMPT TEMPLATE
#Instead of f-strings, LangChaim uses template objects.
#{topic} is a variable - we fill it when we call a chain.
#The advantage: the template is reusable and validatable.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise technical tutor. Answer in 3 bullet points maximum."),
    ("human", "Explain {topic} simply.")
])

#Step 3 : THE OUTPUT PARSER
#StrOutputParser just extracts the text string from the model's response.
#Without it, we will get a full AIMessage object back, not a clean string
parser = StrOutputParser()

#Step 4 : THE CHAIN
#This is LCEL - LangChain Expression Language - pipe operator connects left to right.
#Input flows: prompt gets filled -> goes to model -> model output gets parsed.
chain = prompt | model | parser

#Step 5 : RUN IT 
print("="*50)
print("CHAIN 1: Explain a concept")
print("="*50)

result = chain.invoke({"topic": "RAG in AI Agents"})
print(result)

print("="*50)
print("CHAIN 2: Same chain, different topic")
print("="*50)

result2 = chain.invoke({"topic": "vector databases"})
print(result2)


#Multiple inputs: 
#batch() runs the chain on a list of inputs - useful for processing.
#multiple things without writing a for loop
print("\n" + "="*50)
print("CHAIN 3: Batch - run on multiple topics at once")
print("="*50)

topics = [
    {"topic": "Langchain"},
    {"topic": "embeddings"},
    {"topic": "FastAPI"}
]

results = chain.batch(topics)

for topic, result in zip(topics, results):
    print(f"\n{topic['topic']}:")
    print(result)

#Second chain : different purpose, reusing model and parser
#Notice that the model and parser would stay the same. Only the prompt changes.
#This is the composability benefit of LCEL.

comparison_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a technical analyst.
    Compare the two technologies given.
    Format your answer as:
    SIMILARITY: one sentence
    DIFFERENCE: one sentence  
    USE WHEN: one sentence each"""),
    ("human", "Compare {tech_a} vs {tech_b}")
])

comparison_chain = comparison_prompt | model | parser

print("\n" + "=" * 50)
print("CHAIN 4: Comparison chain")
print("=" * 50)

comparison = comparison_chain.invoke({
    "tech_a": "LangChain",
    "tech_b": "calling the LLM API directly"
})

print(comparison)


import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")


llm = Ollama(model="llama2")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chatbot that answers cosmology questions."),
    ("user", "{query}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# response = chain.invoke({"query" : "What is gravitational lensing in 5 sentences?"})
response = chain.invoke({"query" : "Can you elaborate for a few more lines?"})

print(response)
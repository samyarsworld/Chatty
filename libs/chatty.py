import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")

#### INDEXING ####

# Load Documents
print("Loading documents...")
loader = TextLoader("../cosmos.txt")
docs = loader.load()
print(f"{len(docs)} documents loaded.")

# Split
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"{len(splits)} splits done.")

# Embed
print(f"Uploading {len(splits)} vectors to the vector database...")
vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings())
retriever = vectorstore.as_retriever()
print("Uploading to vector database done.")

#### RETRIEVAL and GENERATION ####

# Prompt
template = """Based solely on the information given in the following context, answer the following question.
            If the information isn’t available in the context, simply reply with ‘Sorry, can't provide an answer.’.
            Please do not provide additional explanations or information.

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = Ollama(model="llama3")

ollama_emb = OllamaEmbeddings(model="llama3")

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
question = "Who went to the new planet and what was the name of the planet?"

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
print("Chatty is thinking about your question...")
response = chain.invoke({"question" : question})

print(response)
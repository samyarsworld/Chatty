import logging
import os
import json
import pathlib
from dotenv import load_dotenv
import requests
from langsam.libs.constants import FILE_LOADERS, ALLOWED_FILE_TYPES
from langsam.libs.constants.prompt_templates import *

import streamlit as st

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings


load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")

logger = logging.getLogger(__name__)


@st.cache_resource
def load_model(model_name, device):
    with st.spinner(f"Downloading the {model_name} embeddings model..."):
        embedding_model=HuggingFaceInstructEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device}
        )
    return embedding_model


class Chatty:
    """
    Chatbot
    """
    def __init__(
            self,
            file_path: str,
            file_type: str,
            embedding_model: str="hkunlp/instructor-large",
            device: str="cpu"
    ) -> None:
        """
        Perform initial parsing of the uploaded file and initialize the
        chat instance.

        Args:
            file_path: Full path and name of uploaded file
            file_type: File extension determined after upload

        Returns:


        """
        self.embedding_model = load_model(embedding_model, device)
        self.vectordb = None
        loader = FILE_LOADERS[file_type](file_path=file_path)
        pages = loader.load_and_split()
        docs = self.__split_into_chunks(pages)
        self.__upload_to_db(docs)

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.llm = Ollama(model=st.session_state['selected_model'], base_url="http://ollama:11434")

        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.vectordb.as_retriever(search_kwargs={"k": 10}),
            memory=self.memory
        )

    def __upload_to_db(self, docs):
        """
        Upload to vector database.
        Args:

        Returns:

        """
        pass

    def __split_into_chunks(self, pages):
        """
        Split pages into chunks.
        Args:

        Returns:
        
        """
        pass
    
    def chat(self) -> str:
        pass
        


class StandAloneBot:

    def __init__(self):
        pass

    def run(self):
        #### INDEXING ####

        # Load Documents
        print("Loading documents...")
        loader = FILE_LOADERS["txt"]("../cosmos.txt")
        docs = loader.load()
        print(f"{len(docs)} documents loaded.")

        # Split
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
        splits = text_splitter.split_documents(docs)
        print(f"{len(splits)} splits done.")

        # Embed
        print(f"Uploading {len(splits)} vectors to the vector database...")
        vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings())
        retriever = vectorstore.as_retriever()
        print("Uploading to vector database done.")

        #### RETRIEVAL and GENERATION ####

        # Prompt
        prompt = ChatPromptTemplate.from_template(LLAMA3_GENERATE_TEMPLATE)

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


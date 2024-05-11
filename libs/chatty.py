"""
Main application
"""
import logging
import os
import json
import pathlib
from dotenv import load_dotenv
import requests
from constants import FILE_LOADERS, ALLOWED_FILE_TYPES

import streamlit as st

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.text_splitter import SemanticChunker
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

class Message:
    """
    Base message class
    """
    def __init__(self, message):
        self.message = message

class UserMessage(Message):
    """
    Represents a message from the user.
    """

class ChatbotMessage(Message):
    """
    Represents a message from the chatbot.
    """


@st.cache_resource
def load_model(modelName, device):
    with st.spinner(f"Downloading the {modelName} embeddings model..."):
        embeddingModel=HuggingFaceInstructEmbeddings(
            model_name=modelName,
            model_kwargs={"device": device}
        )
    return embeddingModel


class Chatty:
    """
    Chatbot
    """
    def __init__(
            self,
            filePath: str,
            fileType: str,
            embeddingModel: str="hkunlp/instructor-large",
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
        self.embeddingModel = load_model(embeddingModel, device)
        self.vectordb = None
        loader = FILE_LOADERS[fileType](file_path=filePath)
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
        


chatbot = Chatty('"../cosmos.txt"', 'txt', embeddingModel="hkunlp/instructor-large", device='cuda')

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


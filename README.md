﻿# Chatty

UNDONE REPOSITIRY. 20% COMPLETE.

## Summary

Chatty is an advanced AI chatbot that enables users to upload and interact with various document formats including PowerPoint, PDF, Excel, Word, and text files (XLSX, PPTX, DOCX, PDF, CSV, TXT). Leveraging a suite of open-source technologies such as Weaviate, Ollama, Llama-3, Docker, Streamlit, and Ollama Web UI, Chatty allows users to pose questions directly from their documents without the need for private APIs. This README outlines the methods, technologies, and metrics used to create Chatty, offering a detailed view of its functionality and advantages.

Note 1: This is a generalized version of this app. Adding Metadata to the embeddings is highly important and requires lots of considerations for different applications which outside of scope of this project.

Note 2: Latency and handling throughput become a driving factor for a production-level app. Refactoring to incorporate parallelization to reduce latency, building a multi-deployable model to handle throughput, and building inference clusters on the cloud are crucial to handle the publicly available app. These are outside of the scope of this project.

## Models

3 models will be built:

1. A naive RAG model using Chromadb, Langchain, and Llama-2 8B.
2. A semi-advanced RAG model using Chromadb, Langchain, and Llama-3 8B.
3. An advanced RAG model using Weaviate, Langchain, and Llama-3 8B.

Note: Initially LangSmith will be used for development.
Note: Initially a virtual environment is used for development. Later we move towards dockerizing the application for production.

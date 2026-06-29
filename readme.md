#  RAG Pipeline with LangChain, ChromaDB, Groq & Streamlit

## Overview

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about PDF documents. The system retrieves relevant document chunks from ChromaDB and generates answers using Groq's Llama 3.3 70B model.

## Tech Stack

* LangChain
* ChromaDB
* HuggingFace Embeddings (`all-mpnet-base-v2`)
* Groq (`llama-3.3-70b-versatile`)
* Streamlit

## Workflow

```text
PDF Documents
      |
Document Loader
      |
Text Splitter
      |
Embeddings
      |
ChromaDB
      |
Retriever
      |
Relevant Chunks
      |
Groq LLM
      |
Generated Answer
```

## Features

* PDF document ingestion
* Text chunking
* Vector embeddings
* Similarity search
* Context-aware answer generation
* Source document retrieval

## Author

Built using LangChain, ChromaDB, Groq, and Streamlit.

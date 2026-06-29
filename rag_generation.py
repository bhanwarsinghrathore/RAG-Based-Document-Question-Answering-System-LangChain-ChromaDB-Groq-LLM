# -------- IMPORTS --------

from pathlib import Path

from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv

load_dotenv()

# -------- PATH CONFIG --------
# Local directory where Chroma stores vectors, metadata, and index files
BASE_DIR = Path(__file__).parent
vector_db_path = str(BASE_DIR / "vector_db")
collection_name = "documents_collection"

# -------- EMBEDDING MODEL --------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"                  # "BAAI/bge-base-en-v1.5"
)

# -------- LLM (Groq) --------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=512
)

# -------- LOAD VECTOR STORE --------
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embedding,
    persist_directory=vector_db_path
)
print("Vector Count:", vector_store._collection.count())

# Create retriever
retriever = vector_store.as_retriever(                           # Retriever performs similarity search on stored vectors
    search_type="similarity",                                    # "mmr" / similarity_score_threshold
    search_kwargs={
        "k": 1,
    }
)         

# -------- RAG PROMPT --------                                  # Prompt template used to inject retrieved context into the LLM
template = """
You are a helpful assistant.

Use only the provided context.

If the answer is not present in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# -------- BUILD RAG CHAIN/PIPELINE --------
# Convert retrieved documents into a single context string
def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('source')}\n"
        f"{doc.page_content}"
        for doc in docs
    )

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -------- USER QUERY --------                                    
def ask_question(query):
    answer = rag_chain.invoke(query)
    docs = retriever.invoke(query)

    return answer, docs




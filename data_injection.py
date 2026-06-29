# -------- IMPORTS --------

from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from pathlib import Path

# -------- PATH CONFIG --------
BASE_DIR = Path(__file__).parent

docs_dir_path = BASE_DIR / "documents"
vector_db_path = str(BASE_DIR / "vector_db")
collection_name = "documents_collection"

# -------- EMBEDDING MODEL --------
# HuggingFace embedding model for converting text into vectors
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"                     # BAAI/bge-base-en-v1.5
)

# -------- LOAD DOCUMENTS --------
# Loads all PDF files from the given directory path
loader = PyPDFDirectoryLoader(docs_dir_path)

documents = loader.load()

# -------- TEXT SPLITTING --------
# splitting a large PDF into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # Max length of each chunk
    chunk_overlap=200    # overlap between two consecutive chunks
)

# Split large PDF documents into smaller parts/chunks for efficient embedding and retrieval
text_chunks = text_splitter.split_documents(documents)

# -------- VECTOR STORE CREATION/ CHROMA DB --------
# Generate embeddings for all text chunks and store them in Chroma Database
vector_store = Chroma.from_documents(
    documents=text_chunks,       # Input text chunks
    embedding=embedding,         # Embedding model
    persist_directory=vector_db_path,  # DB save location
    collection_name=collection_name    # Logical collection name inside Chroma used to organize vector records
)



 
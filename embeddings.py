from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Splits documents into smaller chunks for better processing and embedding.
    
    Args:
        documents (list): List of documents to be chunked.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.
        
    Returns:
        list: List of chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)
    
    return chunks

def embed_documents(chunks, embedding_model="text-embedding-ada-002", persist_directory="chroma_db"):
    """
    Embeds the chunked documents using OpenAI's embedding model and stores them in a Chroma vector store.
    
    Args:
        chunks (list): List of chunked documents.
        embedding_model (str): The OpenAI embedding model to use.
        persist_directory (str): Directory to persist the Chroma vector store.
        
    Returns:
        Chroma: The Chroma vector store containing the embedded documents.
    """
    # Ensure the API key is available in the session state
    if "api_key" not in st.session_state or not st.session_state.api_key:
        raise ValueError("OpenAI API key is missing. Please provide it in the sidebar.")

    # Pass the API key explicitly to OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=st.session_state.api_key)
    
    # Create or load the Chroma vector store
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    
    # Persist the database
    db.persist()
    
    return db


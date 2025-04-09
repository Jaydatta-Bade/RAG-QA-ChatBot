import streamlit as st
import tiktoken

def calculate_embedding_cost(texts):
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004  # $0.0004 per 1k tokens

def clear_history():
    """
    Clears the session state history.
    """
    if "history" in st.session_state:
        del st.session_state["history"]
    if "messages" in st.session_state:
        del st.session_state["messages"]
    if "docs" in st.session_state:
        del st.session_state["docs"]
    if "embeddings" in st.session_state:
        del st.session_state["embeddings"]
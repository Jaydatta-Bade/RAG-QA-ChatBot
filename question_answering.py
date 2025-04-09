from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st

def ask_and_get_answer(vectorstore, q, k=3):
    """
    Ask a question and get an answer using the vector store.

    Args:
        vectorstore (Chroma): The Chroma vector store containing the embedded documents.
        q (str): The question to ask.
        k (int): The number of documents to retrieve.

    Returns:
        str: The answer to the question.
    """
    # Ensure the API key is available in the session state
    if "api_key" not in st.session_state or not st.session_state.api_key:
        raise ValueError("OpenAI API key is missing. Please provide it in the sidebar.")

    # Pass the API key explicitly to ChatOpenAI
    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": k})
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=st.session_state.api_key)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.invoke(q)

    return answer["result"]
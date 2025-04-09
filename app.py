import os
import streamlit as st
from document_loader import load_document
from embeddings import chunk_documents, embed_documents
from utils import clear_history, calculate_embedding_cost
from question_answering import ask_and_get_answer
import random
import time

def main():
    st.subheader('RAG Question-Answering ChatBot 🤖')
    st.write("This is a conversational chatbot for your private documents powered by OpenAI, ChromaDB, LangChain, and Streamlit.")


    with st.sidebar:
        # Input box for OpenAI API key
        api_key = st.text_input('Enter OpenAI API Key:', type='password')
        if api_key:
            st.session_state.api_key = api_key

        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            if 'api_key' not in st.session_state or not st.session_state.api_key:
                st.error("Please enter your OpenAI API key to proceed.")
            else:
                with st.spinner('Reading, chunking and embedding file ...'):
                    file_name = os.path.join('./', uploaded_file.name)
                    with open(file_name, 'wb') as f:
                        f.write(uploaded_file.getvalue())

                    data = load_document(file_name)
                    chunks = chunk_documents(data, chunk_size=chunk_size)
                    tokens, embedding_cost = calculate_embedding_cost(chunks)
                    vector_store = embed_documents(chunks)
                    st.session_state.vs = vector_store
                    st.success('File uploaded, chunked and embedded successfully.')

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! Upload a file and ask questions. 👇"}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            if 'vs' in st.session_state:
                vector_store = st.session_state.vs
                answer = ask_and_get_answer(vector_store, prompt, k)
                assistant_response = answer
            else:
                assistant_response = random.choice(
                    [
                        "Please upload a file first.",
                        "I need a file to answer your questions.",
                        "Upload a file to get started.",
                    ]
                )

            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()


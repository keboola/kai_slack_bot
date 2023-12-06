# docsbot.py

import streamlit as st
import pinecone 
import openai
import os
import logging

from llama_index import VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore

from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="DocsBot", layout="centered")
st.title("📑 DocsBot")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

pinecone.init(
    api_key=api_key,
    environment=pinecone_env
)

if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, ask me a question about your docs!"}
    ]
    
vector_store = PineconeVectorStore(pinecone.Index("zendesk"))
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)

if prompt := st.chat_input("Your question"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
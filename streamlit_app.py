import os
import logging
from decouple import config
from llama_index.chat_engine import CondenseQuestionChatEngine
from llama_index.chat_engine.condense_question import ChatMessage
from langchain.callbacks import StreamlitCallbackHandler

import streamlit as st

import openai
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index import VectorStoreIndex
from llama_index.prompts import Prompt

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

pinecone.init(api_key=api_key, environment=pinecone_env)
index = pinecone.Index("zendesk")

custom_prompt = Prompt("""\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation. 

<Chat History> 
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
""")

    
if "messages" not in st.session_state:
    st.session_state.messages = []
    ai_intro = "Hello, I'm Kai, your AI SQL Bot. I'm here to assist you with SQL queries.What can I do for you?"
    
    st.session_state.messages.append({"role":"assistant", "content" : ai_intro})

vector_store = PineconeVectorStore(index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine()
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    condense_question_prompt=custom_prompt,
    #chat_history=st.session_state.messages,
    verbose=True
)


user_input = st.chat_input("ask_a_question")


if user_input:
    # Add user message to the chat
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display "Kai is typing..."
    with st.chat_message("Kai"):
        st.markdown("typing")
    st_callback = StreamlitCallbackHandler(st.container())
    response = chat_engine.chat(user_input)

    # Add Kai's message to session state
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Display Kai's message
    with st.chat_message("Kai"):
        st.markdown(response)
        # Display source nodes for Kai's response
        st.write(response.source_nodes)

with st.container():    
    last_output_message = []
    last_user_message = []

    for message in reversed(st.session_state.messages):
        if message["role"] == "Kai":
            last_output_message = message
            break
    for message in reversed(st.session_state.messages):
        if message["role"] =="user":
            last_user_message = message
            break  

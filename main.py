"""Main entrypoint for the app."""
import os

import langsmith
from src.models import ChatRequest
from src.chain import create_chain, get_pinecone_retriever_with_index
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from langchain_cohere import ChatCohere, CohereEmbeddings
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename='.env'))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
# PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
# PINECONE_INDEX_NAME = 'confluence'

APP_PORT = int(os.environ.get("APP_PORT"))

client = langsmith.Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r-plus",
    temperature=0
)

retriever = get_pinecone_retriever_with_index(
    pinecone_api_key=PINECONE_API_KEY,
    index_name='serverless-kai-dev',
    embedding_model=CohereEmbeddings(cohere_api_key=COHERE_API_KEY,
                                     model="embed-english-v3.0"),
    k=5
)

rag_chain = create_chain(llm, retriever)

add_routes(
    app,
    rag_chain,
    path="/rag-chain",
    input_type=ChatRequest,
    config_keys=["metadata", "configurable", "tags"],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)

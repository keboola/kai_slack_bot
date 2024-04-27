"""Main entrypoint for the app."""
import os

from src.models import ChatRequest
from src.chain import create_chain
from src.client import get_pinecone_selfquery_retriever_with_index
from src.metadata_fields import (
    confluence_metadata_fields,
    keboola_dev_tools_metadata_fields
)

import langsmith
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename='.env'))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

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

# llm = ChatOpenAI(
#     openai_api_key=OPENAI_API_KEY,
#     model="gpt-4-turbo-2024-04-09"
# )

embedding_model = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

# TODO: Complete Self-query retriever
# document_content_description = """\
# Developer documentation for developers who are working with Keboola programmatically
# """
#
# retriever = get_pinecone_selfquery_retriever_with_index(
#     pinecone_api_key=PINECONE_API_KEY,
#     index_name='kai-knowledge-base',
#     llm=llm,
#     embedding_model=embedding_model,
#     document_content_description=document_content_description,
#     metadata_field_info=keboola_dev_tools_metadata_fields,
#     return_k=5
# )

vectorstore = PineconeVectorStore(
    pinecone_api_key=PINECONE_API_KEY,
    embedding=embedding_model,
    index_name=PINECONE_INDEX_NAME
)

retriever = vectorstore.as_retriever(k=5)

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

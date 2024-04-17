"""Main entrypoint for the app."""
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import ChatCohere
from langchain_community.vectorstores import Chroma

import langsmith
from src.models import ChatRequest
from src.chain import create_chain
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename='.env'))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = 'confluence'

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
APP_PORT = int(os.environ.get("APP_PORT"))

# llm = ChatOpenAI(
#     openai_api_key=OPENAI_API_KEY,
#     model_name="gpt-3.5-turbo-0125",
#     temperature=0
# )


# retriever = get_pinecone_retriever_with_index(
#     pinecone_api_key=PINECONE_API_KEY,
#     index_name=PINECONE_INDEX_NAME,
#     embedding_model=OpenAIEmbeddings()
# )

llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r-plus",
    temperature=0
)

loader = WebBaseLoader("https://arxiv.org/html/2305.10601v2")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=256
)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

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



import os
import re
import langsmith
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from typing import List, Sequence
from langchain_core.language_models import LanguageModelLike
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain.chains import RetrievalQA

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
import os

# available at app.pinecone.io
os.environ['PINECONE_API_KEY'] = os.environ.get('PINECONE_API_KEY') or "your Pinecone API key"
# available at platform.openai.com/api-keys
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY') or "your OpenAI API key"

index_name = "serverless-confluence-dev"

model_name = 'text-embedding-3-small'

embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.environ['OPENAI_API_KEY']
)

vectorstore = PineconeVectorStore(
    pinecone_api_key=os.environ['PINECONE_API_KEY'],
    embedding=embeddings,
    index_name=index_name
)
retriever = vectorstore.as_retriever(k=5)

llm = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    model_name='gpt-4o',
    temperature=0.0
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)




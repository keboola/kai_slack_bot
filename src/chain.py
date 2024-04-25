import os
import re
import langsmith
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from typing import List, Sequence
from langchain_core.language_models import LanguageModelLike
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)

from langchain_cohere import ChatCohere
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.client import get_cohere_retriever_with_reranker


load_dotenv(find_dotenv(filename='.env'))

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

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


SYSTEM_MULTI_QUERY_TEMPLATE = """\
You are an AI language model assistant tasked with understanding \
the context of a conversation and generating standalone versions of a follow-up \
question to facilitate a comprehensive document search in a vector database. \
If a query contains only one subject or aspect, generate a single standalone \
question. For queries with multiple subjects or aspects, create multiple \
questions per each aspect. Each reformulated question should be standalone \
and crafted to address potential limitations in distance-based similarity \
search. Return these alternative questions separated by a newline.
"""

# SYSTEM_MULTI_QUERY_TEMPLATE = """\
# You are an AI language model assistant tasked with understanding \
# the context of a conversation and generating multiple versions of a follow-up \
# question to facilitate a comprehensive document search in a vector database. \
# Use the chat history and the provided follow-up question to create only one \
# distinct query. Each reformulated question should be standalone and crafted \
# to address potential limitations in distance-based similarity search.
# """

# Possibly add chat history from Slack thread
HUMAN_MULTI_QUERY_TEMPLATE = """\
Follow-up question: {question}
Alternative Question:
"""

SYSTEM_RESPONSE_TEMPLATE = """
You are an AI assistant with the capability to retrieve relevant documents \
to aid in answering user queries. First, retrieve pertinent information from a \
specified document collection. Construct a detailed and accurate response \
based on the user's question and the retrieved documents. If there is \
no relevant information within the context, respond with "Hmm, I'm not sure." \
Generate a comprehensive answer of 80 words or less, using an unbiased and \
journalistic tone. Combine information from different sources into a coherent \
answer without repeating text. 
"""
# Cite the sources in your answer using [number] \
# notation, where the count starts from 1. Only cite the most relevant results \
# that accurately answer the question. Place these citations at the end of the \
# sentence that reference them - do not put them all at the end. ALWAYS include \
# list of cited source URLs at the end of your answer, formatted as \
# "Sources:\n[number] URL".
# """

HUMAN_RESPONSE_TEMPLATE = """
Document collection is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question.
Question: {question}
Answer:
"""

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def unique_documents(documents_lists: List[List[Document]]) -> List[Document]:
    unique_docs = []
    seen_contents = set()  # Set to track seen page contents for uniqueness
    for documents in documents_lists:
        for document in documents:
            if document.page_content not in seen_contents:
                seen_contents.add(document.page_content)
                unique_docs.append(document)
    return unique_docs


def create_retriever_chain(
        llm: LanguageModelLike,
        retriever: BaseRetriever
) -> Runnable:
    MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_MULTI_QUERY_TEMPLATE),
            # MessagesPlaceholder(variable_name="chat_history"),
            ("human", HUMAN_MULTI_QUERY_TEMPLATE),
        ]
    )

    multi_query_chain = (
            MULTI_QUERY_PROMPT
            | ChatCohere(model="command-r-plus", temperature=0)
            | StrOutputParser()
            | (lambda x: re.sub(r'\n+', '\n', x))
            | (lambda x: x.split("\n"))
    ).with_config(run_name="MultiQuery")

    # Cohere reranker
    compression_retriever = get_cohere_retriever_with_reranker(
        cohere_api_key=COHERE_API_KEY,
        base_retriever=retriever,
        model="rerank-english-v3.0",
        top_n=3
    ).with_config(run_name="RetrieverRerank")

    return (
            multi_query_chain
            | compression_retriever.map()
            | RunnableLambda(unique_documents)
            .with_config(run_name="FlattenUnique")
    ).with_config(run_name="RetrievalChainWithReranker")


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n\n".join(formatted_docs)


def parse_sources(docs: Sequence[Document]) -> List[str]:
    urls = list(set(doc.metadata['source_url'] for doc in docs))
    if urls:
        return [f"{i+1}. {url}" for i, url in enumerate(urls)]


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")

    source_urls = []
    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=lambda x: format_docs(x["docs"]))
        .assign(sources=lambda x: source_urls.append(parse_sources(x["docs"])))
        .with_config(run_name="RetrieveDocs")
    )

    response_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_RESPONSE_TEMPLATE),
            ("human", HUMAN_RESPONSE_TEMPLATE),
        ]
    )

    rag_chain = (
            RunnablePassthrough()
            | context
            | response_prompt
            | llm
            | StrOutputParser()
            | (lambda x: f"{x}\n\nSources:\n" + "\n".join(source_urls[0]) if source_urls[0] else "")
    )

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

# TODO: Tailor FewShotPrompt examples for self-query
# TODO: add multiple index retrievement ability (Cohere with connectors)
# TODO: add web search
# TODO: add slack search
# TODO: ingest code base
# TODO: ingest zendesk

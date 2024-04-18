import os
import re
from operator import itemgetter
from typing import Dict, List, Optional, Sequence
from src.models import ChatRequest

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import langsmith

from langchain_pinecone import PineconeVectorStore

from langchain_cohere import ChatCohere, CohereEmbeddings, CohereRagRetriever, \
    CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableSequence,
    chain, RunnableSerializable,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv, find_dotenv
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
# PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
# PINECONE_INDEX_NAME = 'confluence'

SYSTEM_MULTI_QUERY_TEMPLATE = """\
You are an AI language model assistant tasked with understanding \
the context of a conversation and generating multiple versions of a follow-up \
question to facilitate a comprehensive document search in a vector database. \
Use the chat history and the provided follow-up question to create five \
distinct queries. Each reformulated question should be standalone and crafted \
to address potential limitations in distance-based similarity search. \
Return these alternative questions separated by a newline.
"""

HUMAN_MULTI_QUERY_TEMPLATE = """\
Follow-up question: {question}
Alternative Questions:
"""

SYSTEM_RESPONSE_TEMPLATE = """
You are an AI assistant with the capability to retrieve relevant documents \
to aid in answering user queries. First, retrieve pertinent information from a \
specified document collection. Construct a detailed and accurate response \
based on the user's question and the retrieved documents. If there is \
no relevant information within the context, respond with "Hmm, I'm not sure." \
Generate a comprehensive answer of 80 words or less, using an unbiased and \
journalistic tone. Combine information from different sources into a coherent \
answer without repeating text. Cite the sources in your answer using [number] \
notation, where the count starts from 1. Only cite the most relevant results \
that accurately answer the question. Place these citations at the end of the \
sentence that reference them - do not put them all at the end. ALWAYS include \
list of cited source URLs at the end of your answer, formatted as \
"Sources:\n[number] URL".
"""

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


def get_pinecone_retriever_with_index(
        pinecone_api_key: str,
        index_name: str,
        embedding_model: Embeddings,
        k: int = 5
) -> BaseRetriever:
    pinecone_client = PineconeVectorStore(
        pinecone_api_key=pinecone_api_key,
        embedding=embedding_model,
        index_name=index_name
        # environment=PINECONE_ENVIRONMENT  # for pod-based only
    )

    vectorstore = pinecone_client.from_existing_index(
        index_name=index_name,
        embedding=embedding_model
    )

    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_cohere_retriever_with_reranker(
        base_retriever: BaseRetriever,
        model: str,
        top_n: int = 3
) -> ContextualCompressionRetriever:
    cohere_rerank = CohereRerank(
        cohere_api_key=COHERE_API_KEY,
        model=model,
        top_n=top_n
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=cohere_rerank,
        base_retriever=base_retriever
    )

    return compression_retriever


def create_retriever_chain(
        llm: LanguageModelLike,
        retriever: BaseRetriever
) -> Runnable:
    MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_MULTI_QUERY_TEMPLATE),
            MessagesPlaceholder(variable_name="messages"),
            ("human", HUMAN_MULTI_QUERY_TEMPLATE),
        ]
    )

    multi_query_chain = (
            MULTI_QUERY_PROMPT
            | llm
            | StrOutputParser()
            | (lambda x: re.sub(r'\n+', '\n', x))
            | (lambda x: x.split("\n"))
    ).with_config(run_name="MultiQuery")

    # Cohere reranker
    compression_retriever = get_cohere_retriever_with_reranker(
        base_retriever=retriever,
        model="rerank-english-v3.0",
        top_n=3
    ).with_config(run_name="RetrieverRerank")

    # rag_retriever = CohereRagRetriever(llm=llm)

    return (
            multi_query_chain
            | compression_retriever.map()
            # | rag_retriever.map().with_config(run_name="RagRetriever")
            | RunnableLambda(unique_documents)
            .with_config(run_name="FlattenUnique")
    ).with_config(run_name="RetrievalChainWithReranker")


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    # print(docs)
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}' source='{doc.metadata['source']}'>\
        {doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n\n".join(formatted_docs)


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")

    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=lambda x: format_docs(x["docs"]))
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
    )

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )


# TODO: add multiple index retrievement ability (Cohere with connectors)
# TODO: add web search
# TODO: add slack search
# TODO: pinecone serverless, ingest confluence
# TODO: ingest code base
# TODO: ingest zendesk

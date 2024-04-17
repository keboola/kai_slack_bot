import os
import re
from operator import itemgetter
from typing import Dict, List, Optional, Sequence
from src.models import ChatRequest

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from langchain_anthropic import ChatAnthropic

import langsmith

from langchain_pinecone import PineconeVectorStore

from langchain_cohere import ChatCohere, CohereEmbeddings, CohereRagRetriever, \
    CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_community.vectorstores import Chroma
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

from langchain.load import dumps, loads

# from langchain_fireworks import ChatFireworks
# from langchain_google_genai import ChatGoogleGenerativeAI

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

# COHERE_RERANK_API_KEY = os.environ.get("COHERE_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = 'confluence'

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
Chat History:
{chat_history}
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
sentence or paragraph that reference them - do not put them all at the end. \
Include source URLs at the end of your answer, formatted as "[number]: [URL]".
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
        embedding_model: Embeddings
) -> BaseRetriever:
    pinecone_client = PineconeVectorStore(
        pinecone_api_key=pinecone_api_key,
        embedding=embedding_model,
        index_name=index_name
        # environment=PINECONE_ENVIRONMENT
    )

    vectorstore = pinecone_client.from_existing_index(
        index_name=index_name,
        embedding=embedding_model
    )

    return vectorstore.as_retriever()


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


# def reciprocal_rank_fusion(results: List[List[Document]], k=5) -> List[Document]:
#     fused_scores = {}
#     for docs in results:
#         # Assumes the docs are returned in sorted order of relevance
#         for rank, doc in enumerate(docs):
#             doc_str = dumps(doc)
#             if doc_str not in fused_scores:
#                 fused_scores[doc_str] = 0
#             previous_score = fused_scores[doc_str]
#             fused_scores[doc_str] += 1 / (rank + k)
#
#     reranked_results = [
#         loads(doc)
#         for doc, score in
#         sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
#     ]
#     print(reranked_results)
#     return reranked_results


def create_retriever_chain(
        llm: LanguageModelLike,
        retriever: BaseRetriever
) -> Runnable:
    MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_MULTI_QUERY_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
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


def serialize_history(request: ChatRequest) -> List[BaseMessage]:
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(
                HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


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

    default_response_synthesizer = response_prompt | llm

    @chain
    def cohere_response_synthesizer(input: dict) -> RunnableSerializable:
        return response_prompt | llm.bind(source_documents=input["docs"])

    # response_synthesizer = (
    #         default_response_synthesizer.configurable_alternatives(
    #             ConfigurableField("llm"),
    #             default_key="openai_gpt_3_5_turbo",
    #             anthropic_claude_3_sonnet=default_response_synthesizer,
    #             fireworks_mixtral=default_response_synthesizer,
    #             google_gemini_pro=default_response_synthesizer,
    #             cohere_command=cohere_response_synthesizer,
    #         )
    #         | StrOutputParser()
    # ).with_config(run_name="GenerateResponse")

    rag_chain = (
        RunnablePassthrough()
        | context
        # | response_synthesizer
        | default_response_synthesizer
        | StrOutputParser()
    )

    chain_with_message_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_message_history


import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load, chunk and index the contents of the blog.
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )

loader = WebBaseLoader("https://arxiv.org/html/2305.10601v2")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

retriever = vectorstore.as_retriever(k=5)

# llm = ChatOpenAI(
#     openai_api_key=OPENAI_API_KEY,
#     model_name="gpt-3.5-turbo-0125",
#     temperature=0
# )

llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r-plus",
    temperature=0
)

# retriever = get_pinecone_retriever_with_index(
#     pinecone_api_key=PINECONE_API_KEY,
#     index_name=PINECONE_INDEX_NAME,
#     embedding_model=OpenAIEmbeddings()
# )

# TODO: add chat memory
# TODO: add multiple index retrievement ability (Cohere with connectors)
# TODO: add web search
# TODO: add slack search
# TODO: pinecone serverless, ingest confluence


answer_chain = create_chain(llm, retriever)

if __name__ == "__main__":
    # Test run
    answer = answer_chain.invoke(
        {
            'question': "What's LLM agent?",
            'chat_history': []
        }
    )
    print(answer)

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

from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer

# from src.client import get_pinecone_selfquery_retriever_with_index
from src.client import get_cohere_retriever_with_reranker
from src.prompts import (
    SYSTEM_MULTI_QUERY_TEMPLATE,
    HUMAN_MULTI_QUERY_TEMPLATE,
    SYSTEM_RESPONSE_TEMPLATE,
    HUMAN_RESPONSE_TEMPLATE
)

load_dotenv(find_dotenv(filename='.env'))

#client = langsmith.Client()
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
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")


def unique_documents(documents_lists: List[List[Document]]) -> List[Document]:
    unique_docs = []
    seen_contents = set()  # Set to track seen page contents for uniqueness
    for documents in documents_lists:
        for document in documents:
            if document.page_content not in seen_contents:
                seen_contents.add(document.page_content)
                unique_docs.append(document)
    return unique_docs


def retrieve_full_page(docs: List[Document]) -> List[Document]:
    urls = list(set(doc.metadata['source_url'] for doc in docs))
    loader = AsyncHtmlLoader(urls, encoding='utf-8')
    html_docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(html_docs)

    cleaned_docs = []
    for doc in docs_transformed:
        if doc.metadata['source'].startswith('https://developers.keboola.com/'):
            # Clean the page content and update the document's metadata
            cleaned_content = re.sub(
                pattern=r'(?s)^DEVELOPERS DOCS.*?dbt\n\n',
                repl='',
                string=doc.page_content
            )
            doc.page_content = cleaned_content
        cleaned_docs.append(doc)

    return cleaned_docs


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n\n".join(formatted_docs)


def parse_sources(docs: Sequence[Document]) -> List[str]:
    # source in case of using AsyncHtmlLoader, source_url otherwise
    urls = list(set(doc.metadata['source_url'] for doc in docs))
    if urls:
        return [f"{i + 1}. {url}" for i, url in enumerate(urls)]


def create_retriever_chain(retriever: BaseRetriever) -> Runnable:
    multi_query_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_MULTI_QUERY_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", HUMAN_MULTI_QUERY_TEMPLATE),
        ]
    )

    multi_query_chain = (
            multi_query_prompt
            | ChatCohere(model="command-r-plus", temperature=0)
            | StrOutputParser()
            | (lambda x: re.sub(r'\n+', '\n', x))
            | (lambda x: x.split("\n"))
    ).with_config(run_name="MultiQuery")

    cohere_retriever_with_reranker = get_cohere_retriever_with_reranker(
        cohere_api_key=COHERE_API_KEY,
        base_retriever=retriever,
        model="rerank-english-v3.0",
        top_n=3
    ).with_config(run_name="RetrieverRerank")

    return (
            multi_query_chain
            | cohere_retriever_with_reranker.map()
            | RunnableLambda(unique_documents)
            .with_config(run_name="FlattenUnique")
            # | RunnableLambda(retrieve_full_page)
            # .with_config(run_name="RetrieveFullPage")
    ).with_config(run_name="RetrievalChainWithReranker")


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    retriever_chain = create_retriever_chain(
        retriever
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

    return (
            RunnablePassthrough()
            | context
            | response_prompt
            | llm
            | StrOutputParser()
            | (lambda x: f"{x}\n\nSources:\n" + "\n".join(source_urls[0])
                if source_urls[0] else "")  # attach sources
    ).with_config(run_name="RagFullChain")


# Configure language models and vector store
llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r-plus",
    temperature=0
)
embedding_model = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)
vectorstore = PineconeVectorStore(
    pinecone_api_key=PINECONE_API_KEY,
    embedding=embedding_model,
    index_name=PINECONE_INDEX_NAME
)
retriever = vectorstore.as_retriever(k=5)

# Initialise rag chain
rag_chain = create_chain(llm, retriever)


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

# TODO: Come up with a list of examples: input query – output structured_request
# TODO: Tailor FewShotPrompt examples for self-query
# TODO: add multiple index retrievement ability
# TODO: add web search
# TODO: add slack search
# TODO: ingest code base
# TODO: ingest zendesk
# TODO: retrieve confluence full pagse

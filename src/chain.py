import os
from operator import itemgetter
from typing import Dict, List, Optional, Sequence
from src.models import ChatRequest

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from langchain_anthropic import ChatAnthropic

import langsmith
from langsmith import Client

from langchain_pinecone import PineconeVectorStore

from langchain_cohere import ChatCohere, CohereEmbeddings, CohereRagRetriever, CohereRerank
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
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.load import dumps, loads

# from langchain_fireworks import ChatFireworks
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langsmith import Client

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename='.env'))

client = Client()

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

COHERE_RERANK_API_KEY = os.environ.get("COHERE_API_KEY")
COHERE_COMMAND_R_PLUS_API_KEY = os.environ.get("COHERE_API_KEY")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = 'confluence'

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question. Do NOT answer the question, just reformulate \
it if needed, otherwise return as it is.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

MULTI_QUERY_TEMPLATE = """\
You are an AI language model assistant. Your task is to generate five \
different versions of the given user question to retrieve relevant documents from a vector \
database. By generating multiple perspectives on the user question, your goal is to help \
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by a single newline sign "\n". User question: {question}"""

RESPONSE_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Keboola.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation.
At the end of your answer:
1. Create a list of source URLs of the cited documents, use format "[document id]: URL". Only cite the most \
relevant results that answer the question accurately. 
You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user."""



store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def unique_documents(documents_list: List[Document]) -> List[Document]:
    unique_documents = []
    seen_contents = set()  # Set to track seen page contents for uniqueness
    for document in documents_list:
        if document.page_content not in seen_contents:
            seen_contents.add(document.page_content)
            unique_documents.append(document)
    return unique_documents


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
        retriever: BaseRetriever,
        cohere_api_key: str,
        model: str,
        pick_top_n: int = 3
) -> ContextualCompressionRetriever:
    cohere_rerank = CohereRerank(
        cohere_api_key=cohere_api_key,
        model=model,
        top_n=pick_top_n
    )

    # Create a compression retriever that uses the Cohere reranker and the base retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=cohere_rerank,
        base_retriever=retriever
    )

    return compression_retriever


def reciprocal_rank_fusion(results: List[List[Document]], k=5) -> List[Document]:
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        loads(doc)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    print(reranked_results)
    return reranked_results


def create_retriever_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
            CONDENSE_QUESTION_PROMPT
            | llm
            | StrOutputParser()
    ).with_config(run_name="CondenseQuestion")

    with_message_history = RunnableWithMessageHistory(
        condense_question_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    MULTI_QUERY_PROMPT = PromptTemplate.from_template(MULTI_QUERY_TEMPLATE)
    listof_questions_chain = (
            # condense_question_chain
            with_message_history
            | MULTI_QUERY_PROMPT
            | llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
    ).with_config(run_name="ListofQuestions")

    # Cohere reranker
    # compression_retriever = get_cohere_retriever_with_reranker(
    #     retriever=retriever,
    #     cohere_api_key=COHERE_RERANK_API_KEY,
    #     model="rerank-english-v3.0",
    #     pick_top_n=3
    # )

    return (
            listof_questions_chain
            | retriever.map()
            | RunnableLambda(reciprocal_rank_fusion).with_config(run_name="FusionRerank")
            | RunnableLambda(unique_documents).with_config(run_name="FlattenUnique")
    ).with_config(run_name="RetrievalChainWithReranker")


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    # print(docs)
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}' source='{doc.metadata['source']}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n\n".join(formatted_docs)


def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    default_response_synthesizer = prompt | llm

    @chain
    def cohere_response_synthesizer(input: dict) -> RunnableSerializable:
        return prompt | llm.bind(source_documents=input["docs"])

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

    return (
            RunnablePassthrough()
            | context
            # | response_synthesizer
            | default_response_synthesizer
            | StrOutputParser()
    )





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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
retriever = vectorstore.as_retriever(k=4)

# llm = ChatOpenAI(
#     openai_api_key=OPENAI_API_KEY,
#     model_name="gpt-3.5-turbo-0125",
#     temperature=0
# )

llm = ChatCohere(
    cohere_api_key=COHERE_COMMAND_R_PLUS_API_KEY,
    model="command-r-plus",
    temperature=0,
)

# retriever = get_pinecone_retriever_with_index(
#     pinecone_api_key=PINECONE_API_KEY,
#     index_name=PINECONE_INDEX_NAME,
#     embedding_model=OpenAIEmbeddings()
# )

# TODO: add chat memory
# TODO: optimise query transformation
# TODO: add multiple index retrievement ability
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

import os
from operator import itemgetter
from typing import Dict, List, Optional, Sequence

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from langchain_anthropic import ChatAnthropic


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

# from langchain_fireworks import ChatFireworks
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langsmith import Client


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

COHERE_RERANK_API_KEY = os.environ.get("COHERE_API_KEY")
COHERE_COMMAND_R_PLUS_API_KEY = os.environ.get("COHERE_API_KEY")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = 'confluence'

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

MULTI_QUERY_TEMPLATE = """\
You are an AI language model assistant. Your task is to generate five \
different versions of the given user question to retrieve relevant documents from a vector \
database. By generating multiple perspectives on the user question, your goal is to help \
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. User question: {question}"""

RESPONSE_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Keboola.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user."""


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


def unique_documents(documents: Sequence[Document]) -> List[Document]:
    return [doc for i, doc in enumerate(documents) if doc not in documents[:i]]


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


def create_retriever_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
            CONDENSE_QUESTION_PROMPT
            | llm
            | StrOutputParser()
    ).with_config(run_name="CondenseQuestion")

    MULTI_QUERY_PROMPT = PromptTemplate.from_template(MULTI_QUERY_TEMPLATE)
    listof_questions_chain = (
            condense_question_chain
            | MULTI_QUERY_PROMPT
            | llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
    ).with_config(run_name="ListofQuestions")

    compression_retriever = get_cohere_retriever_with_reranker(
        retriever=retriever,
        cohere_api_key=COHERE_RERANK_API_KEY,
        model="rerank-english-v2.0",
        pick_top_n=3
    )

    conversation_chain = (
            listof_questions_chain
            | compression_retriever.map()
            | (lambda document_lists: [doc for docs in document_lists for doc in docs])
            | (lambda documents: [doc for i, doc in enumerate(documents) if doc not in documents[:i]])
    )

    return conversation_chain.with_config(run_name="RetrievalChainWithReranker")


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    print(docs)
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


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
            RunnablePassthrough.assign(chat_history=serialize_history)
            | context
            # | response_synthesizer
            | default_response_synthesizer
            | StrOutputParser()
    )


if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(filename='.env'))

    import bs4
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

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

    answer_chain = create_chain(llm, retriever)
    answer = answer_chain.invoke(
        {
            'question': "What's LLM agent?",
            'chat_history': []
        }
    )
    print(answer)

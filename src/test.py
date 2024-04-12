import os

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename='.env'))

if os.getenv("PINECONE_API_KEY", None) is None:
    raise Exception("Missing `PINECONE_API_KEY` environment variable.")

if os.getenv("PINECONE_ENV", None) is None:
    raise Exception("Missing `PINECONE_ENV` environment variable.")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "confluence")


# Set up index with multi query retriever
vectorstore = PineconeVectorStore.from_existing_index(
    'confluence', OpenAIEmbeddings()
)

model = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-3.5-turbo-0125",
    temperature=0
)

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=model
)

# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
)

print(chain.invoke("What is HealthCheck Lite?"))
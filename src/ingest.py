import os
import bs4
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename='.env'))

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if __name__ == '__main__':
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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    vectorstore = PineconeVectorStore(
        pinecone_api_key=PINECONE_API_KEY,
        index_name='serverless-kai-dev',
        embedding=CohereEmbeddings(cohere_api_key=COHERE_API_KEY,
                                   model="embed-english-v3.0")
    )
    vectorstore.add_documents(splits)
    print("DONE")

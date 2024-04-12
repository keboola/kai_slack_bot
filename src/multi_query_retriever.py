import asyncio
import logging
from typing import List, Sequence

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _unique_documents(documents: Sequence[Document]) -> List[Document]:
    return [doc for i, doc in enumerate(documents) if doc not in documents[:i]]


class KaiPineconeRetriever:
    """Custom version of Langchain's MultiQueryRetriever"""

    """Retrieve docs for each query. Return the unique union of all retrieved docs."""

    def __init__(self):
        # Initialize the model with zero temperature for deterministic results
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            logger.error("OPENAI_API_KEY is not set in the environment variables.")
            raise ValueError("Missing OPENAI_API_KEY")

        # self.embed = OpenAIEmbeddings(
        #     openai_api_key=openai_api_key,
        #     model="text-embedding-ada-002"
        # )

        self.retriever = PineconeVectorStore

        # self.vectorstore = self.retriever.from_existing_index(
        #     index_name='confluence',
        #     embedding=self.embed
        # )

    def _get_relevant_documents(self, queries: List[str]) -> List[Document]:
        """Get relevant documents given a user query.

        Args:
            queries: list of queries

        Returns:
            Unique union of relevant documents from all generated queries
        """

        documents = self.retrieve_documents(queries)
        return self.unique_union(documents)

    def retrieve_documents(self, queries: List[str]) -> List[Document]:
        """Run all LLM generated queries.

        Args:
            queries: query list

        Returns:
            List of retrieved Documents
        """
        documents = []
        for query in queries:
            docs = self.retriever.get_relevant_documents(query)
            documents.extend(docs)
        return documents

    def unique_union(self, documents: List[Document]) -> List[Document]:
        """Get unique Documents.

        Args:
            documents: List of retrieved Documents

        Returns:
            List of unique retrieved Documents
        """
        return _unique_documents(documents)


if __name__ == "__main__":
    # Test run
    import os
    from langchain.retrievers import MultiQueryRetriever
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_pinecone import PineconeVectorStore
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(filename='.env'))

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name='confluence',
        embedding=OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
    )

    model = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-3.5-turbo-0125",
        temperature=0
    )

    retriever = KaiMultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever()
    )

    unique_docs = retriever.get_relevant_documents(
        query="Keboola paid leave for fathers"
    )
    print(unique_docs)

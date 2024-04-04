import asyncio
import logging
from typing import List, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


def _unique_documents(documents: Sequence[Document]) -> List[Document]:
    return [doc for i, doc in enumerate(documents) if doc not in documents[:i]]


class KaiMultiQueryRetriever():
    """Custom version of Langchain's MultiQueryRetriever"""

    """Retrieve docs for each query. Return the unique union of all retrieved docs."""


    def get_relevant_documents(
            self,
            queries: List[str],
            *,
    ) -> List[Document]:
        """Get relevant documents given a user query.

        Args:
            queries: list of queries

        Returns:
            Unique union of relevant documents from all generated queries
        """

        documents = self.retrieve_documents(queries)
        return self.unique_union(documents)

    def retrieve_documents(
            self, queries: List[str], run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Run all LLM generated queries.

        Args:
            queries: query list

        Returns:
            List of retrieved Documents
        """
        documents = []
        for query in queries:
            docs = self.retriever.get_relevant_documents(
                query, callbacks=run_manager.get_child()
            )
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
        'confluence', OpenAIEmbeddings()
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
        query="What is Keboola?"
    )
    print(unique_docs)

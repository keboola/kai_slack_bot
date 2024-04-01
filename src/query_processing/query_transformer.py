import os
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

import dotenv
dotenv.load_dotenv(dotenv.find_dotenv())


# Ensure the necessary environment variables are set
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise Exception("Missing `PINECONE_API_KEY` environment variable.")

PINECONE_ENV = os.getenv("PINECONE_ENV")
if not PINECONE_ENV:
    raise Exception("Missing `PINECONE_ENV` environment variable.")


class QueryTransformer:
    def __init__(self, pinecone_index_name='confluence'):
        # Initialize vector store with the given Pinecone index name and embeddings
        self.vectorstore = PineconeVectorStore.from_existing_index(
            pinecone_index_name, OpenAIEmbeddings()
        )

        # Initialize the model with zero temperature for deterministic results
        self.model = ChatOpenAI(temperature=0)

        # Create the retriever using the vector store and model
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(), llm=self.model
        )

        # Define the RAG prompt template
        self.template = """
        Answer the question based only on the following context: {context}
        Question: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)

        # Create the RAG chain combining components into a processing pipeline
        self.chain = (
                RunnableParallel({"context": self.retriever, "question": RunnablePassthrough()})
                | self.prompt
                | self.model
                | StrOutputParser()
        )

    def generate_multi_queries(self, question_text):
        # Process the question through the chain to generate multiple queries
        return self.chain.invoke(question_text)


if __name__ == "__main__":
    qt = QueryTransformer()
    multi_queries = qt.generate_multi_queries("What is HealthCheck Lite?")
    print(multi_queries)

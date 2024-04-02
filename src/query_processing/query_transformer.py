"""Module for processing transforming user queries into multiple versions"""
import os
import logging
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv, find_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv(filename='.env'))


class QueryTransformer:
    def __init__(self):
        # Initialize the model with zero temperature for deterministic results
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            logger.error("OPENAI_API_KEY is not set in the environment variables.")
            raise ValueError("Missing OPENAI_API_KEY")

        self.model = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo-0125",
            temperature=0
        )
        logger.info("ChatOpenAI model initialized with zero temperature.")

        # Multi Query prompt template
        template = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {question}"""

        self.prompt = ChatPromptTemplate.from_template(template)
        logger.info("Multi Query prompt template initialized.")

        # Create the chain combining components into a processing pipeline
        self.chain = (
                self.prompt
                | self.model
                | StrOutputParser()
                | (lambda x: x.split("\n"))
        )
        logger.info("Query processing chain created.")

    # Generates 5 queries
    def generate_multi_queries(self, question_text: str) -> List[str]:
        logger.info(f"Generating multiple queries for the question: '{question_text}'")
        # Process the question through the chain to generate multiple queries
        return self.chain.invoke(question_text)


if __name__ == "__main__":
    # Test run
    qt = QueryTransformer()
    multi_queries = qt.generate_multi_queries("What is HealthCheck Lite?")
    logger.info(f"Generated queries: {multi_queries}")

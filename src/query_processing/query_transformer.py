from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv, find_dotenv
# OPENAI_API_KEY loads from environment automatically when ChatOpenAI() initialised
load_dotenv(find_dotenv())


class QueryTransformer:
    def __init__(self):
        # Initialize the model with zero temperature for deterministic results
        self.model = ChatOpenAI(temperature=0)

        # Multi Query prompt template
        template = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {question}"""

        self.prompt = ChatPromptTemplate.from_template(template)

        # Create the chain combining components into a processing pipeline
        self.chain = (
                self.prompt
                | self.model
                | StrOutputParser()
                | (lambda x: x.split("\n"))
        )

    def generate_multi_queries(self, question_text):
        # Process the question through the chain to generate multiple queries
        return self.chain.invoke(question_text)


if __name__ == "__main__":
    # Test run
    qt = QueryTransformer()
    multi_queries = qt.generate_multi_queries("What is HealthCheck Lite?")
    print(multi_queries)
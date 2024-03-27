import os
import logging
from decouple import config

import openai
import pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
# from llama_index.chat_engine.condense_question import CondenseQuestionChatEngine
from llama_index.llms import ChatMessage
from llama_index.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)

# Initialize environment variables and APIs
os.environ["OPENAI_API_KEY"] = openai.api_key = config("OPENAI_API_KEY")
api_key = config("PINECONE_API_KEY")
pinecone_env = config("PINECONE_ENV")
pinecone.init(api_key=api_key, environment=pinecone_env)

# Custom prompt for question condensation
custom_prompt = PromptTemplate("""\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
""")


# Main CLI application class
class CLIApp:
    def __init__(self):
        vector_store = PineconeVectorStore(pinecone.Index("kaidev"))
        self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    def run(self):
        print("Welcome to the CLI Chatbot. Type 'quit' to exit.")
        chat_history = []
        while True:
            user_message = input("You: ")
            if user_message.lower() == 'quit':
                break

            # Here you might add logic to manage and store the chat history as needed
            # For simplicity, this example doesn't accumulate history between questions

            # Prepare the message history for processing
            # This is simplified and should be adapted based on your chat history management logic
            message_history = [
                ChatMessage(role="user", content=user_message)
                # You can append previous chat messages here
            ]

            # Initialize chat engine with the message history
            query_engine = self.index.as_query_engine()
            chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine=query_engine,
                condense_question_prompt=custom_prompt,
                chat_history=message_history,
                verbose=True
            )

            # Generate and display response
            response = chat_engine.chat(user_message)
            response_str = str(response)
            response_source = response.sources
            response_message = f"Bot: {response_str}\n\nSources: {response_source}"
            print(response_message)


if __name__ == "__main__":
    cli_app = CLIApp()
    cli_app.run()

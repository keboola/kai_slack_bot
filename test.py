"""Main entrypoint for the app."""
import os
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler

from src.models import ChatRequest
from src.chain import create_chain
# from src.client import get_pinecone_selfquery_retriever_with_index

import langsmith
from langserve import add_routes

from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List


load_dotenv(find_dotenv(filename='.env'))

# configurations
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
APP_PORT = int(os.environ.get("APP_PORT"))

client = langsmith.Client()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Configure language models and vector store
llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r-plus",
    temperature=0
)
embedding_model = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)
vectorstore = PineconeVectorStore(
    pinecone_api_key=PINECONE_API_KEY,
    embedding=embedding_model,
    index_name=PINECONE_INDEX_NAME
)
retriever = vectorstore.as_retriever(k=5)
rag_chain = create_chain(llm, retriever)

# Add routes for RAG chain
add_routes(
    app,
    rag_chain,
    path="/rag-chain",
    input_type=ChatRequest,
    config_keys=["metadata", "configurable", "tags"],
)

# Define Slack App
slack_app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
slack_handler = SlackRequestHandler(slack_app)


def serialize_thread_history(thread_history) -> List[BaseMessage]:
    if thread_history:
        converted_thread_history = []
        for msg in thread_history:
            if msg.get("human", False):
                converted_thread_history.append(
                    HumanMessage(content=msg.get["text", ""])
                )
            if msg.get("bot_id", False):
                converted_thread_history.append(
                    AIMessage(content=msg.get["text", ""])
                )
        return converted_thread_history


@app.post("/slack/events")
async def slack_events_endpoint(req: Request):
    return await slack_handler.handle(req)


# Handling Slack Events
@slack_app.event("message")
def handle_message_events(body, say, context, client):
    # Extract thread and channel info
    event = body['event']
    channel_id = event['channel']
    thread_ts = event.get('thread_ts') or event['ts']

    # visual feedback
    client.reactions_add(
        channel=event["channel"],
        name="eyes",
        timestamp=event["ts"]
    )

    thread_history = client.conversations_replies(
        channel=event["channel"],
        ts=thread_ts
    )

    # Invoke RAG chain with message as input
    chat_request = ChatRequest(
        question=event['text'],
        chat_history=serialize_thread_history(thread_history))
    response = rag_chain.invoke(chat_request)

    # Post message response back to Slack
    client.chat_postMessage(channel=channel_id, thread_ts=thread_ts,
                            text=f"Response: {response}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)

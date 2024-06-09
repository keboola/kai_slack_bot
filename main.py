"""Main entrypoint for the app."""
import os
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from slack_bolt.adapter.socket_mode import SocketModeHandler

from src.chain import rag_chain
from src.models import ChatRequest
from src.slack import SlackApp

import langsmith
from langserve import add_routes

load_dotenv(find_dotenv(filename='.env'))

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SIGNING_SECRET")
APP_PORT = int(os.environ.get("APP_PORT"))

#client = langsmith.Client()  
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add routes for RAG chain
add_routes(
    app,
    rag_chain,
    path="/rag-chain",
    input_type=ChatRequest,
    config_keys=["metadata", "configurable", "tags"],
)

# Define Slack App
slack_app = SlackApp(SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET)


@app.post("/slack/events")
async def slack_events_endpoint(req: Request):
    return await slack_app.app_handler.handle(req)


if __name__ == "__main__":
    # Socket mode for local testing
    # SocketModeHandler(slack_app.app, os.environ.get("SLACK_APP_TOKEN")).start()

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)

"""Main entrypoint for the app."""
import asyncio
from typing import Optional, Union
from uuid import UUID

import langsmith
from langsmith import Client
from src.chain import ChatRequest, answer_chain
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename='.env'))

client = Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


add_routes(
    app,
    answer_chain,
    path="/chat",
    input_type=ChatRequest,
    config_keys=["metadata", "configurable", "tags"],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)

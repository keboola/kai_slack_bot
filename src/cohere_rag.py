import os
from pprint import pprint
from langchain_cohere import ChatCohere, CohereRagRetriever
import langsmith
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename='.env'))

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

# User query we will use for the generation
user_query = "What's the current price of a TON coin?"

# Use Cohere's RAG retriever with Cohere Connectors to generate an answer.
# Cohere provides exact citations for the sources it used.
llm = ChatCohere(model="command-r-plus", temperature=0)
rag = CohereRagRetriever(llm=llm, connectors=[{"id": "web-search"}])
docs = rag.get_relevant_documents(user_query)
answer = docs.pop()

pprint("Relevant documents:")
pprint(docs)

pprint(f"Question: {user_query}")
pprint("Answer:")
pprint(answer.page_content)
pprint(answer.metadata["citations"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)

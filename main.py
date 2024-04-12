import dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from src.query_transformer import QueryTransformer

dotenv.load_dotenv()

# query_transformer = QueryTransformer()
# multi_queries = query_transformer.generate_multi_queries("What is HealthCheck Lite?")
os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
embed = OpenAIEmbeddings(
    openai_api_key=openai_api_key,
    model="text-embedding-ada-002"
)

vectorstore = PineconeVectorStore.from_existing_index(
    index_name='confluence',
    embedding=embed
)

retriever = vectorstore.as_retriever()

# docs = {}
# for query in queries:
#     docs += retriever.get_relevant_documents(query)
# print(docs)

# rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
# )

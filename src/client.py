from typing import Dict, Sequence
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.language_models import BaseLanguageModel

from langchain_pinecone import PineconeVectorStore
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.self_query.pinecone import PineconeTranslator

from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)


def get_pinecone_selfquery_retriever_with_index(
        pinecone_api_key: str,
        index_name: str,
        llm: BaseLanguageModel,
        embedding_model: Embeddings,
        document_content_description: str,
        metadata_field_info: Sequence[AttributeInfo],
        return_k: int = 5
) -> BaseRetriever:
    vectorstore = PineconeVectorStore(
        pinecone_api_key=pinecone_api_key,
        embedding=embedding_model,
        index_name=index_name
    )
    pinecone_translator = PineconeTranslator()
    prompt = get_query_constructor_prompt(
        document_content_description,
        metadata_field_info,
        examples=None,
        allowed_comparators=pinecone_translator.allowed_comparators,
        allowed_operators=pinecone_translator.allowed_operators,
        schema_prompt=None,
    )
    output_parser = StructuredQueryOutputParser.from_components()

    query_constructor = (
            prompt
            # | (lambda x: x.text.replace(
            #     "eq | ne | gt | gte | lt | lte | contain | like | in | nin",
            #     'ne | gt | gte | lt | lte | in | nin'))
            | llm
            | output_parser
    ).with_config(run_name="SelfQueryConstructor")

    return SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vectorstore,
        structured_query_translator=PineconeTranslator(),
        search_kwargs={"k": return_k},
        enable_limit=False,
        verbose=True
    )


def get_cohere_retriever_with_reranker(
        cohere_api_key: str,
        base_retriever: BaseRetriever,
        model: str,
        top_n: int = 3
) -> ContextualCompressionRetriever:
    cohere_rerank = CohereRerank(
        cohere_api_key=cohere_api_key,
        model=model,
        top_n=top_n
    )

    return ContextualCompressionRetriever(
        base_compressor=cohere_rerank,
        base_retriever=base_retriever
    )

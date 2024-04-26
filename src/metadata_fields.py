from langchain.chains.query_constructor.base import AttributeInfo

confluence_metadata_fields = {
    AttributeInfo(
        name="confluence_space",
        description="Name of a Confluence space",
        type="string",
    ),
    AttributeInfo(
        name="created_date",
        description="The date of creation of the document",
        type="date",
    ),
    AttributeInfo(
        name="creator",
        description="Person who created of the document",
        type="string",
    ),
    AttributeInfo(
        name="data_source",
        description="Data source of the document",
        type="string",
    ),
    AttributeInfo(
        name="doc_id",
        description="Document ID",
        type="string",
    ),
    AttributeInfo(
        name="header_1",
        description="Header 1 of the document",
        type="string",
    ),
    AttributeInfo(
        name="last_modifier",
        description="The last person who modified the document",
        type="string",
    ),
    AttributeInfo(
        name="last_updated_date",
        description="The date of the last modification of the document",
        type="date",
    ),
    AttributeInfo(
        name="page_title",
        description="Title of the document",
        type="string",
    ),
    AttributeInfo(
        name="source_url",
        description="Source URL of the document",
        type="string",
    )
}

keboola_dev_tools_metadata_fields = {
    AttributeInfo(
        name="data_source",
        description="Database name of the document: confluence, keboola_dev_docs",
        type="string",
    ),
    AttributeInfo(
        name="header_1",
        description="Header of the document",
        type="string",
    ),
    AttributeInfo(
        name="page_title",
        description="Title of the document",
        type="string",
    ),
    AttributeInfo(
        name="source_url",
        description="Source URL of the document",
        type="string",
    )
}

# Template
template_metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]


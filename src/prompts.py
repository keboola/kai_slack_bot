# flake8: noqa
from langchain_core.prompts import PromptTemplate

# ---------------------------- CHAIN PROMPTS -----------------------------------
SYSTEM_MULTI_QUERY_TEMPLATE = """\
You are an AI language model assistant tasked with understanding \
the context of a conversation and generating standalone versions of a follow-up \
question to facilitate a comprehensive document search in a vector database. \
If a query contains only one subject or aspect, generate a single standalone \
question. For queries with multiple subjects or aspects, create multiple \
questions per each aspect. Each reformulated question should be standalone \
and crafted to address potential limitations in distance-based similarity \
search. Return these alternative questions separated by a newline.
"""

HUMAN_MULTI_QUERY_TEMPLATE = """\
Follow-up question: {question}
Alternative Question:
"""

SYSTEM_RESPONSE_TEMPLATE = """
You are an AI assistant with the capability to retrieve relevant documents \
to aid in answering user queries. First, retrieve pertinent information from a \
specified document collection. Construct a detailed and accurate response \
based on the user's question and the retrieved documents. If there is \
no relevant information within the context, respond with "Hmm, I'm not sure." \
Generate a comprehensive answer of 80 words or less, using an unbiased and \
journalistic tone. Combine information from different sources into a coherent \
answer without repeating text. 
"""

HUMAN_RESPONSE_TEMPLATE = """
Document collection is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question.
Question: {question}
Answer:
"""

# ---------------------------- SELF-QUERY PROMPTS ------------------------------
SONG_DATA_SOURCE = """\
```json
{{
    "content": "Lyrics of a song",
    "attributes": {{
        "artist": {{
            "type": "string",
            "description": "Name of the song artist"
        }},
        "length": {{
            "type": "integer",
            "description": "Length of the song in seconds"
        }},
        "genre": {{
            "type": "string",
            "description": "The song genre, one of \"pop\", \"rock\" or \"rap\""
        }}
    }}
}}
```\
"""

example_DATA_SOURCE = """\
```json
{
    "content": "Developer documentation for developers who are working with Keboola programmatically",
    "attributes": {
        "source_url": {
            "description": "Source URL of the document",
            "type": "string"
        },
        "header_1": {
            "description": "Header of the document",
            "type": "string"
        },
        "page_title": {
            "description": "Title of the document",
            "type": "string"
        }
    }
}
```\
"""
# TODO: come up with examples
SELFQUERY_INPUT_EXAMPLES = [
    "",
    "",
    "",
    "",
    ""
]

SELFQUERY_OUTPUT_EXAMPLES = [
    {
        "query": "",
        "filter": ""
    },
    {
        "query": "",
        "filter": ""
    },
    {
        "query": "",
        "filter": ""
    },
    {
        "query": "",
        "filter": ""
    },
    {
        "query": "",
        "filter": ""
    },
]

SELFQUERY_EXAMPLES = list(zip(
    SELFQUERY_INPUT_EXAMPLES,
    SELFQUERY_OUTPUT_EXAMPLES,
    strict=True
))

FULL_ANSWER = """\
```json
{{
    "query": "teenager love",
    "filter": "and(or(eq(\\"artist\\", \\"Taylor Swift\\"), eq(\\"artist\\", \\"Katy Perry\\")), lt(\\"length\\", 180), eq(\\"genre\\", \\"pop\\"))"
}}
```\
"""

NO_FILTER_ANSWER = """\
```json
{{
    "query": "",
    "filter": "NO_FILTER"
}}
```\
"""


DEFAULT_EXAMPLES = [
    {
        "i": 1,
        "data_source": SONG_DATA_SOURCE,
        "user_query": "What are songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes long in the dance pop genre",
        "structured_request": FULL_ANSWER,
    },
    {
        "i": 2,
        "data_source": SONG_DATA_SOURCE,
        "user_query": "What are songs that were not published on Spotify",
        "structured_request": NO_FILTER_ANSWER,
    },
]


EXAMPLE_PROMPT_TEMPLATE = """\
<< Example {i}. >>
Data Source:
{data_source}

User Query:
{user_query}

Structured Request:
{structured_request}
"""

EXAMPLE_PROMPT = PromptTemplate.from_template(EXAMPLE_PROMPT_TEMPLATE)

USER_SPECIFIED_EXAMPLE_PROMPT = PromptTemplate.from_template(
    """\
<< Example {i}. >>
User Query:
{user_query}

Structured Request:
```json
{structured_request}
```
"""
)

DEFAULT_SCHEMA = """\
<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:

```json
{{{{
    "query": string \\ text string to compare to document contents
    "filter": string \\ logical condition statement for filtering documents
}}}}
```

The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.

A logical condition statement is composed of one or more comparison and logical operation statements.

A comparison statement takes the form: `comp(attr, val)`:
- `comp` ({allowed_comparators}): comparator
- `attr` (string):  name of attribute to apply the comparison to
- `val` (string): is the comparison value

A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` ({allowed_operators}): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to

Make sure that you only use the comparators and logical operators listed above and no others.
Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters only use the attributed names with its function names if there are functions applied on them.
Make sure that filters only use format `YYYY-MM-DD` when handling date data typed values.
Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.\
"""
DEFAULT_SCHEMA_PROMPT = PromptTemplate.from_template(DEFAULT_SCHEMA)

DEFAULT_PREFIX = """\
Your goal is to structure the user's query to match the request schema provided below.

{schema}\
"""

PREFIX_WITH_DATA_SOURCE = (
        DEFAULT_PREFIX
        + """

<< Data Source >>
```json
{{{{
    "content": "{content}",
    "attributes": {attributes}
}}}}
```
"""
)

# DEFAULT_SUFFIX = """\
# << Example {i}. >>
# Data Source:
# ```json
# {{{{
#     "content": "{content}",
#     "attributes": {attributes}
# }}}}
# ```
#
# User Query:
# {{query}}
#
# Structured Request:
# """

SUFFIX_WITHOUT_DATA_SOURCE = """\
<< Example {i}. >>
User Query:
{{query}}

Structured Request:
"""



FORMATED_PROMPT = """
DEFAULT_PREFIX----------------------------------------------------
Your goal is to structure the user's query to match the request schema provided below.

    DEFAULT_SCHEMA------------------------------------------------------------
<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:

```json
{
    "query": string \ text string to compare to document contents
    "filter": string \ logical condition statement for filtering documents
}
```

The query string should contain only text that is expected to match the contents of documents. 
Any conditions in the filter should not be mentioned in the query as well.

A logical condition statement is composed of one or more comparison and logical operation statements.

A comparison statement takes the form: `comp(attr, val)`:
- `comp` (ne | gt | gte | lt | lte | in | nin): comparator
- `attr` (string):  name of attribute to apply the comparison to
- `val` (string): is the comparison value

A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` (and | or | not): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation statements): 
one or more statements to apply the operation to

Make sure that you only use the comparators and logical operators listed above and no others.
Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters only use the attributed names with its function names if there are functions applied on them.
Make sure that filters only use format `YYYY-MM-DD` when handling date data typed values.
Make sure that filters take into account the descriptions of attributes and only make comparisons 
that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be applied 
return "NO_FILTER" for the filter value.
    /DEFAULT_SCHEMA------------------------------------------------------------
DEFAULT_PREFIX-----------------------------------------------

DEFAULT_SUFFIX------------------------------------------------------------
<< Example 1. >>
Data Source:
```json
{
    "content": "Lyrics of a song",
    "attributes": {
        "artist": {
            "type": "string",
            "description": "Name of the song artist"
        },
        "length": {
            "type": "integer",
            "description": "Length of the song in seconds"
        },
        "genre": {
            "type": "string",
            "description": "The song genre, one of "pop", "rock" or "rap""
        }
    }
}
```

User Query:
What are songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes long in the dance pop genre

Structured Request:
```json
{
    "query": "teenager love",
    "filter": "and(or(eq(\"artist\", \"Taylor Swift\"), eq(\"artist\", \"Katy Perry\")), lt(\"length\", 180), eq(\"genre\", \"pop\"))"
}
```
/DEFAULT_SUFFIX------------------------------------------------------------


<< Example 2. >>
Data Source:
```json
{
    "content": "Lyrics of a song",
    "attributes": {
        "artist": {
            "type": "string",
            "description": "Name of the song artist"
        },
        "length": {
            "type": "integer",
            "description": "Length of the song in seconds"
        },
        "genre": {
            "type": "string",
            "description": "The song genre, one of "pop", "rock" or "rap""
        }
    }
}
```

User Query:
What are songs that were not published on Spotify

Structured Request:
```json
{
    "query": "",
    "filter": "NO_FILTER"
}
```


<< Example 3. >>
Data Source:
```json
{
    "content": "Developer documentation for developers who are working with Keboola programmatically",
    "attributes": {
    "source_url": {
        "description": "Source URL of the document",
        "type": "string"
    },
    "header_1": {
        "description": "Header of the document",
        "type": "string"
    },
    "page_title": {
        "description": "Title of the document",
        "type": "string"
    }
}
}
```

User Query:
What benefits does Keboola offer when integrated with other tools?

Structured Request:
"""

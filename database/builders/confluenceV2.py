import os
import re
import json
import time
from datetime import datetime
import logging
from bs4 import BeautifulSoup
from tqdm import tqdm
from atlassian import Confluence
import pinecone

from llama_index import (
    Document,
    ServiceContext,
    GPTVectorStoreIndex,
    set_global_service_context,
)

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import OpenAI
from llama_index.llm_predictor import LLMPredictor
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (
    KeywordExtractor,
    MetadataExtractor,
    QuestionsAnsweredExtractor,
)

from llama_index.langchain_helpers.text_splitter import TokenTextSplitter

SYNC_FROM = "2020-01-01T00:00:00.000Z"


def sanitize_filename(filename):
    return re.sub(r"[/\\]", "_", filename)


class ConfluenceDataExtractor:
    def __init__(self, confluence_url, confluence_username, confluence_password, datadir):
        self.datadir = datadir
        self.confluence = Confluence(
            url=confluence_url, username=confluence_username, password=confluence_password
        )

    def download_confluence_pages(self, limit=100):
        spaces = self.confluence.get_all_spaces()
        for space in spaces.get("results"):
            logging.info(f"Downloading Confluence space: {space['name']}")

            content = self.confluence.get_space_content(space["key"])
            while True:
                subdir = os.path.join(self.datadir, "data", space["name"])
                os.makedirs(subdir, exist_ok=True)

                page = content.get("page")
                results = page.get("results")
                size = page.get("size")

                if not results:
                    logging.info(f"No results for {space['name']}")
                    break

                metadata = self._get_metadata(results)

                # Check if the document is already downloaded and up-to-date
                for result in results:
                    metadata_filename = os.path.join(
                        subdir, sanitize_filename(result["title"]) + ".json"
                    )

                    if os.path.exists(metadata_filename):
                        with open(metadata_filename, "r", encoding="utf-8") as file:
                            existing_metadata = json.load(file)
                            if (
                                    metadata["LastUpdatedDate"]
                                    == existing_metadata.get("LastUpdatedDate")
                            ):
                                logging.info(
                                    f"Document '{result['title']}' is up-to-date. Skipping download."
                                )
                                continue

                self._save_results(results, metadata, subdir)

                if size == limit:
                    start = page.get("start") + limit
                    content = self.confluence.get_space_content(
                        space["key"], start=start, limit=limit
                    )
                    page = content.get("page")
                    results = page.get("results")
                    metadata = self._get_metadata(results)

                    # Check if the document is already downloaded and up-to-date
                    for result in results:
                        metadata_filename = os.path.join(
                            subdir, sanitize_filename(result["title"]) + ".json"
                        )

                        if os.path.exists(metadata_filename):
                            with open(metadata_filename, "r", encoding="utf-8") as file:
                                existing_metadata = json.load(file)
                                if (
                                        metadata["LastUpdatedDate"]
                                        == existing_metadata.get("LastUpdatedDate")
                                ):
                                    logging.info(
                                        f"Document '{result['title']}' is up-to-date. Skipping download."
                                    )
                                    continue

                    self._save_results(results, metadata, subdir)
                else:
                    break

    @staticmethod
    def _save_results(results, metadata, directory):
        for result in results:
            content_filename = os.path.join(
                directory, sanitize_filename(result["title"]) + ".txt"
            )
            metadata_filename = os.path.join(
                directory, sanitize_filename(result["title"]) + ".json"
            )

            html_content = result["body"]["storage"]["value"]
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text()
            text = result["title"] + "\n\n" + text

            with open(content_filename, "w", encoding="utf-8") as file:
                file.write(text)

            with open(metadata_filename, "w", encoding="utf-8") as file:
                json.dump(metadata, file)

    def _get_metadata(self, results):
        page_id = results[0].get("id")
        if page_id:
            data = self.confluence.get_page_by_id(page_id)
            space = data["space"].get("name", "")

            page_metadata = {
                "id": space + "-" + data.get("id", ""),
                "CreatedDate": data["history"].get("createdDate", ""),
                "LastUpdatedDate": data["version"].get("when", ""),
                "Title": data.get("title", ""),
                "Creator": data["history"]["createdBy"].get("displayName", ""),
                "LastModifier": data["version"]["by"].get("displayName", ""),
                "url": f"{data['_links']['base']}/spaces/{data['space'].get('key', '')}/pages/{data.get('id', '')}",
                "Space": space,
            }

            return page_metadata
        return {}


class DocumentIndexCreator:
    def __init__(self, datadir, index_name, batch_size=20):
        self.datadir = datadir
        self.runtimes_json_path = os.path.join(self.datadir, "runtimes.json")
        self.index_name = index_name
        self.batch_size = batch_size
        self.doc_titles = []
        self.doc_paths = []

        self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=256)
        self.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
        self.llm_predictor = LLMPredictor(llm=self.llm)
        self.text_splitter = TokenTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=128)
        self.metadata_extractor = MetadataExtractor(
            extractors=[
                # TitleExtractor(nodes=2),
                QuestionsAnsweredExtractor(
                    questions=3, llm_predictor=self.llm_predictor, show_progress=False
                ),
                # SummaryExtractor(summaries=["prev", "self"]),
                KeywordExtractor(keywords=5, llm_predictor=self.llm_predictor, show_progress=False),
            ]
        )
        self.node_parser = SimpleNodeParser(
            text_splitter=self.text_splitter, metadata_extractor=self.metadata_extractor
        )
        self.last_runtimes = self._load_last_runtimes()

    def load_documents(self):
        save_folder = os.path.join(self.datadir, "data")
        for dirpath, dirnames, filenames in os.walk(save_folder):
            for filename in filenames:
                if filename.endswith(".txt"):
                    subdir_name = os.path.basename(dirpath)
                    file_name = os.path.splitext(filename)[0]

                    doc_path = os.path.join(dirpath, filename)

                    metadata_path = os.path.join(
                        dirpath, sanitize_filename(file_name) + ".json"
                    )
                    metadata = self._get_file_metadata(metadata_path)
                    doc_id = metadata.get("id", "")

                    last_updated_date = metadata.get("LastUpdatedDate", SYNC_FROM)
                    last_sync = self.last_runtimes.get(doc_id, SYNC_FROM)

                    last_updated_date = datetime.strptime(last_updated_date, "%Y-%m-%dT%H:%M:%S.%fZ")
                    last_sync = datetime.strptime(last_sync, "%Y-%m-%dT%H:%M:%S.%fZ")

                    if last_updated_date >= last_sync:
                        logging.info(f"File {doc_path} will be upserted to Pinecone.")
                        self.doc_titles.append(subdir_name + " - " + file_name)
                        self.doc_paths.append(doc_path)

    def index_documents(self):
        documents = []
        for title, path in zip(self.doc_titles, self.doc_paths):
            if path.endswith(".txt"):
                text = self._read_file_as_string(path)
                extra_info = self._get_file_metadata(path)

                documents.append(Document(text=text, doc_id=title, extra_info=extra_info))

                if len(documents) >= self.batch_size:
                    self.process_batch(documents)
                    documents = []

        if documents:
            self.process_batch(documents)

    @staticmethod
    def _read_file_as_string(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def _get_file_metadata(self, file_path) -> dict:
        """Returns article metadata from saved json."""
        metadata_path = file_path.replace(".txt", ".json")
        md = self._read_file_as_string(metadata_path)
        md = json.loads(md)
        if md:
            return md
        return {}

    def _load_last_runtimes(self):
        if os.path.exists(self.runtimes_json_path):
            with open(self.runtimes_json_path, "r") as file:
                return json.load(file)
        logging.warning("runtime.json not found. Full sync will be needed.")
        return {}

    def _update_last_runtime(self, doc_id):
        current_time = datetime.utcnow()
        current_ts = current_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        self.save_last_runtime(doc_id, current_ts)

    def save_last_runtime(self, path, current_ts):
        if os.path.exists(self.runtimes_json_path):
            with open(self.runtimes_json_path, "r") as file:
                data = json.load(file)
        else:
            data = {}

        data[path] = current_ts

        with open(self.runtimes_json_path, "w") as file:
            json.dump(data, file)

    def process_batch(self, documents):
        service_context = ServiceContext.from_defaults(
            llm=self.llm, embed_model=self.embed_model, node_parser=self.node_parser
        )
        set_global_service_context(service_context)

        start = time.time()
        for document in tqdm(documents, desc="Processing documents"):
            parsed_nodes = self.node_parser.get_nodes_from_documents([document])
            doc_id = document.metadata.get("id", "")  # Assuming ref_doc_id contains the document path
            self._update_last_runtime(doc_id)
            self.create_and_load_index(index_name=self.index_name, nodes=parsed_nodes)

        logging.info(f"{str(len(documents))} documents parsed in: {time.time() - start}")

    @staticmethod
    def create_and_load_index(index_name: str, nodes):
        """Inserts nodes into Pinecone"""
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENV"],
        )

        pinecone_index = pinecone.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index, insert_kwargs={"show_progress": False})

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002", embed_batch_size=100
        )
        service_context = ServiceContext.from_defaults(embed_model=embed_model)

        GPTVectorStoreIndex(
            nodes,
            storage_context=storage_context,
            service_context=service_context
        )

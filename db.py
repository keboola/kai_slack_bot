from dotenv import load_dotenv
import os
import openai
import logging

from database.builders.confluenceV2 import ConfluenceDataExtractor, DocumentIndexCreator

INDEX_NAME = "kaidev"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv()

    # Conflu variables
    CONFLUENCE_URL = os.environ.get('CONFLUENCE_URL')
    CONFLUENCE_USER = os.environ.get('CONFLUENCE_USERNAME')
    CONFLUENCE_PWD = os.environ.get('CONFLUENCE_PASSWORD')
    datadir = os.environ.get('DATA_DIR')
    CONFLUENCE_SAVE_FOLDER = os.path.join(datadir, "confluence")
    openai.api_key = os.getenv('OPENAI_API_KEY')

    index_name = "kaidev"

    conflu_ex = ConfluenceDataExtractor(confluence_url=CONFLUENCE_URL,
                                        confluence_password=CONFLUENCE_PWD,
                                        confluence_username=CONFLUENCE_USER,
                                        datadir=CONFLUENCE_SAVE_FOLDER)
    # conflu_ex.download_confluence_pages()

    indexer = DocumentIndexCreator(
        datadir=CONFLUENCE_SAVE_FOLDER, index_name=index_name
    )
    indexer.load_documents()
    indexer.index_documents()

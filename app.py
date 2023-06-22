import os
import logging
import streamlit as sl
from slack_bolt import App
from src.tools.confluence_search.confluence_search import conflu_search

logging.basicConfig(level=logging.INFO)

os.environ["OPENAI_API_KEY"] = sl.secrets["OPENAI_API_KEY"]
slack_bot_token = sl.secrets["SLACK_BOT_TOKEN"]
slack_bot_secret = sl.secrets["SLACK_SIGNING_SECRET"]
app_port = int(sl.secrets["APP_PORT"])

sl.info("APP PORT " + str(app_port))

app = App(
    token=slack_bot_token,
    signing_secret=slack_bot_secret
)

from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
import streamlit as st
import sys


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield


def publish_home_tab(client, event, logger):
    try:
        client.views_publish(
            user_id=event["user"],
            view={
                "type": "home",
                "callback_id": "home_view",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Welcome to your _App's Home_* :tada:"
                        }
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "Write something to the bot and it will respond."
                        }
                    }
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


@app.event("app_home_opened")
def update_home_tab_wrapper(client, event, logger):
    publish_home_tab(client, event, logger)


def handle_message_events(client, event):
    client.reactions_add(
        channel=event["channel"],
        name="eyes",
        timestamp=event["ts"]
    )
    message = event["text"]
    logging.info(f"Message received: {message}")

    try:
        response = conflu_search(message).as_query_engine().query(message)
    except AttributeError:
        client.reactions_add(
            channel=event["channel"],
            name="melting_face",
            timestamp=event["ts"]
        )
        response = "I am sorry, I could not find any results for your query."

    logging.info(f"Returning response: {response}")
    client.chat_postMessage(
        channel=event["channel"],
        thread_ts=event["ts"],
        text=f"Hi <@{event['user']}> :wave:\n{response}"
    )

with st_stdout("info"):
    app.event("app_mention")(handle_message_events)
    app.event("message")(handle_message_events)

if __name__ == "__main__":
    sl.info("Starting up the bot ...")
    with st_stdout("info"):
        app.start(app_port)
    sl.success("KAI bot is online")

import os, sys
import logging
import streamlit as st
from slack_bolt import App
from src.tools.confluence_search.confluence_search import conflu_search
import src.tools.redirect as rd

logging.basicConfig(level=logging.INFO)

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
slack_bot_token = st.secrets["SLACK_BOT_TOKEN"]
slack_bot_secret = st.secrets["SLACK_SIGNING_SECRET"]
app_port = int(st.secrets["APP_PORT"])

app = App(
    token=slack_bot_token,
    signing_secret=slack_bot_secret
)

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

with rd.stdout, rd.stderr(format='markdown', to=st.sidebar):
    app.event("app_mention")(handle_message_events)
    app.event("message")(handle_message_events)
    print('set the events connection')

if __name__ == "__main__":
    st.info("Starting up the bot ...")
    print('some fake error', file=sys.stderr)
    app.start(app_port)


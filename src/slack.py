import logging
import re

from dotenv import load_dotenv, find_dotenv
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler

from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.chain import rag_chain
from src.models import ChatRequest

load_dotenv(find_dotenv(filename='.env'))


class SlackApp:
    def __init__(self, bot_token, signing_secret):
        self.app = App(token=bot_token, signing_secret=signing_secret)
        self.app_handler = SlackRequestHandler(self.app)
        self.app.event("app_home_opened")(self.update_home_tab_wrapper)
        self.app.event("app_mention")(self.handle_message_events)
        self.app.event("message")(self.handle_message_events)

    @staticmethod
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
                                "text": "Write something to the bot and it \
                                         will respond with the power of AI."
                            }
                        }
                    ]
                }
            )
        except Exception as e:
            logger.error(f"Error publishing home tab: {e}")

    @staticmethod
    def _serialize_thread_history(thread_history) -> List[BaseMessage]:
        if thread_history:
            converted_thread_history = []
            for msg in thread_history['messages']:
                if msg.get("bot_id", False):
                    converted_thread_history.append(
                        AIMessage(
                            content=msg.get("text"),
                            role='ai'
                        )
                    )
                else:
                    human_msg = re.sub(r'<@[^>]+>', '', msg.get("text"))
                    converted_thread_history.append(
                        HumanMessage(
                            content=human_msg,
                            role='human'
                        )
                    )
            return converted_thread_history

    def update_home_tab_wrapper(self, client, event, logger):
        self.publish_home_tab(client, event, logger)

    def handle_message_events(self, client, event):
        # Extract user message, channel id and thread timestamp
        user_message = event['text']
        channel_id = event['channel']
        thread_ts = event.get("thread_ts", event["ts"])

        # Visual feedback
        try:
            client.reactions_add(
                channel=channel_id,
                name="eyes",
                timestamp=event["ts"]
            )
        except Exception as e:
            pass

        thread_history = client.conversations_replies(
            channel=channel_id,
            ts=thread_ts
        )

        # Invoke RAG chain with message as input
        chat_request = ChatRequest(
            question=re.sub(r'<@[^>]+>', '', user_message),
            chat_history=self._serialize_thread_history(thread_history)
        )

        try:
            response_message = rag_chain.invoke(chat_request.dict())
        except Exception as e:
            response_message = "I'm sorry, some error occured. \
            Let's try that again."

        # Post message response back to Slack
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=response_message
        )

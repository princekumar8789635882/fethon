import os
from datetime import datetime
from uuid import uuid4

from openai import OpenAI
from uagents import Context, Protocol
from uagents.experimental.chat_agent.protocol import build_llm_message_history
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

SYSTEM_PROMPT = (
    "You are TripMate, an expert travel planning assistant. "
    "You help users plan trips end to end: choosing destinations, building "
    "day-by-day itineraries, estimating budgets, comparing transport and "
    "accommodation options, and creating packing lists. "
    "Users often send short, informal messages with typos (for example "
    "'plan a trpi for mme from mumbai to goa'). Interpret the intent, ignore "
    "obvious typos, and answer the underlying travel request. "
    "For trips in India, plan in INR and compare realistic options such as "
    "flights, Konkan Railway trains, overnight buses, and road trips. "
    "Ask for key details (destination, dates, budget, travel party, interests) "
    "when they are missing, but prefer to make reasonable assumptions and state "
    "them briefly so the user gets a useful answer right away. "
    "Structure answers with clear headings and bullet points. "
    "Never invent exact prices, visa requirements, or opening hours; give "
    "realistic ranges and advise the user to verify current details."
)

client = OpenAI(
    # Agentverse injects both variables into every hosted agent; no key is needed here.
    base_url=os.environ["ASI1_BASE_URL"],
    api_key=os.environ["ASI1_API_KEY"],
)

travel_protocol = Protocol(spec=chat_protocol_spec)


def create_text_chat(text: str) -> ChatMessage:
    content = [TextContent(type="text", text=text)]
    return ChatMessage(timestamp=datetime.utcnow(), msg_id=uuid4(), content=content)


@travel_protocol.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id),
    )

    text = msg.text()
    if not text:
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *build_llm_message_history(ctx),
    ]

    try:
        completion = client.chat.completions.create(
            model="asi1",
            messages=messages,
            max_tokens=2048,
        )
        response = completion.choices[0].message.content or ""
    except Exception as exc:
        ctx.logger.exception("Error querying travel model")
        response = (
            "Sorry, I had trouble generating a travel plan right now. "
            f"Please try again in a moment. ({exc})"
        )

    if not response.strip():
        response = "Sorry, I could not generate a reply. Please send your travel question again."

    await ctx.send(sender, create_text_chat(response))


@travel_protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass

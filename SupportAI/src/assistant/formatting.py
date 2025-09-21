from pydantic import BaseModel, Field
from typing import Literal


CATEGORIES = [
"billing",
"login",
"bug",
"feature_request",
"shipping",
]
PRODUCTS = ["Basic", "Pro", "Enterprise"]
PRIORITIES = ["low", "medium", "high", "urgent"]
SENTIMENTS = ["negative", "neutral", "positive"]


class Ticket(BaseModel):
    category: Literal[tuple(CATEGORIES)]
    priority: Literal[tuple(PRIORITIES)]
    product: Literal[tuple(PRODUCTS)]
    sentiment: Literal[tuple(SENTIMENTS)]
    summary: str = Field(..., max_length=240)


TARGET_JSON_SCHEMA = Ticket.model_json_schema()


SYSTEM_PROMPT = (
"You are a meticulous data engineer. Read the user's support message and extract the relevant fields to " \
"populate the given Schema. Output ONLY a JSON object. Do not add extra text."
)


INSTRUCTION = (
"Schema: {schema}\n\n"
"Extract fields from the message to populate the Schema. If unsure, make the best guess. Output strictly valid JSON."
)


def format_prompt(message: str) -> str:
    return (
    f"<s>[SYSTEM]\n{SYSTEM_PROMPT}\n" \
    f"[INSTRUCTION]\n{INSTRUCTION.format(schema=TARGET_JSON_SCHEMA)}\n" \
    f"[USER]\n{message}\n[/USER]\n[ASSISTANT]\n"
    )
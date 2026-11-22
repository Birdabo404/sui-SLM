"""The app's own answer when a question asks who or what the assistant is: fixed text chosen by the picked model,
with no call to Ollama. Only the app's request handling uses it; the eval pipeline sends every question to the model.

The patterns match the whole question, so a question that only mentions a model or a maker ("what model is my CPU",
"who made ls") still goes to the model.
"""
import re

from app.models import MATCHED_BASE

POLARIS_ANSWER = (
    "I'm Polaris, running a fine-tuned and distilled Qwen3.5:2B built by Clyde Heinrich Lontoc and Jebby Cruz of "
    "Caraga State University."
)
STOCK_NAMES = {"qwen3.5:2b": "Qwen3.5:2B", MATCHED_BASE: "Qwen3.5:2B (Q4_K_M)"}

_MAKE = r"(?:made|built|created|trained|developed|programmed|fine[ -]?tuned|distilled)"
_RUN = r"(?:running|using|based on|built on|running on)"
_ASKS = re.compile(
    r"(?:who|what) are you"
    r"|who am i (?:talking|speaking|chatting) (?:to|with)"
    r"|what(?:'s| is) your (?:name|model)"
    r"|what should i call you"
    r"|(?:please )?(?:introduce yourself|tell me about yourself)"
    rf"|(?:what|which) (?:ai |language |llm |base )?model (?:are you|is this|do you use|are you {_RUN}"
    r"|powers you|runs you|is behind you)"
    r"|what llm are you"
    rf"|what are you {_RUN}"
    rf"|(?:who|which company|what company) {_MAKE} you"
    r"|who is your (?:creator|developer|maker|author)"
    r"|are you (?:chatgpt|gpt|qwen|llama|gemini|claude|deepseek|polaris|an ai|a bot|a robot|a human|human|a person)"
)
_GREETING = re.compile(r"^(?:(?:hi|hello|hey)\b[ ,]*)?(?:polaris\b[ ,]*)?")
_VOCATIVE = re.compile(r"[ ,]+polaris$")


def _normalized(question):
    text = " ".join(str(question or "").lower().replace("\u2019", "'").split())
    text = text.strip(" ?!.")
    text = _GREETING.sub("", text, count=1)
    return _VOCATIVE.sub("", text).strip(" ,?!.")


def asks_identity(question):
    return bool(_ASKS.fullmatch(_normalized(question)))


def identity_answer(model_key, model_name):
    """The built-in answer for the picked model: Polaris's line for the KD student, and for a base model (which is
    what PLAIN runs) that it is the stock model with no fine-tuning."""
    if model_key == "distilled":
        return POLARIS_ANSWER
    name = STOCK_NAMES.get(model_name, model_name)
    return (
        f"This is stock {name} with no fine-tuning or distillation, running in Polaris, the offline lab assistant "
        "by Clyde Heinrich Lontoc and Jebby Cruz of Caraga State University."
    )

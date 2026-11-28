"""Rule-based identity response: an intent check that runs before retrieval. A question asking who or what the
assistant is gets fixed text chosen by the picked model and mode, with no call to Ollama. Only the app's request
handling uses it; the eval pipeline sends every question to the model.

The patterns match the whole question, so a question that only mentions a model or a maker ("what model is my CPU",
"who made ls") still goes to retrieval and the model.
"""
import re

from app.engine import BASE_MODEL
from app.models import MATCHED_BASE, STUDENT

STOCK_RESPONSE = (
    "I'm Polaris, an offline Linux lab assistant. Right now I'm running the stock Qwen3.5-2B model with no "
    "fine-tuning, set up by Clyde Heinrich Lontoc and Jebby Cruz of Caraga State University."
)
QUANTISED_STOCK_RESPONSE = (
    "I'm Polaris, an offline Linux lab assistant. Right now I'm running the stock Qwen3.5-2B model (Q4_K_M "
    "quantised, no fine-tuning), set up by Clyde Heinrich Lontoc and Jebby Cruz of Caraga State University."
)
FINE_TUNED_RESPONSE = (
    "I'm Polaris, an offline Linux lab assistant. I run {model}, a Qwen3.5-2B model fine-tuned through knowledge "
    "distillation by Clyde Heinrich Lontoc and Jebby Cruz of Caraga State University."
)
MODEL_NAMES = {STUDENT: "Thesis-Student"}

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
    """The intent check: whether the whole question asks who or what the assistant is."""
    return bool(_ASKS.fullmatch(_normalized(question)))


def identity_response(model_name, mode):
    """The fixed response for the picked model and mode. The Q4_K_M base is the stock model quantised, so it says so
    in every mode; PLAIN (the `base` mode) and the stock default model get the stock text; only the KD student,
    named as it is, gets the fine-tuned text."""
    if model_name == MATCHED_BASE:
        return QUANTISED_STOCK_RESPONSE
    if mode == "base" or model_name == BASE_MODEL:
        return STOCK_RESPONSE
    return FINE_TUNED_RESPONSE.format(model=MODEL_NAMES.get(model_name, model_name))

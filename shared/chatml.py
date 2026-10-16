"""ChatML with Qwen3.5's empty think block. Shared by SFT rows and the Modelfile."""

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
# Qwen3.5 with thinking off (add_generation_prompt) emits this assistant header.
ASSISTANT_PREFIX = f"{IM_START}assistant\n<think>\n\n</think>\n\n"
CHATML_INSTRUCTION = f"{IM_START}user\n"
CHATML_RESPONSE = ASSISTANT_PREFIX


def render_chatml(messages, add_generation_prompt=False):
    """Render system/user/assistant messages the same way the Modelfile does."""
    parts = []
    for message in messages:
        role = message["role"]
        content = message.get("content") or ""
        if role == "system":
            parts.append(f"{IM_START}system\n{content}{IM_END}\n")
        elif role == "user":
            parts.append(f"{IM_START}user\n{content}{IM_END}\n")
        elif role == "assistant":
            body = content
            if body.startswith("<think>"):
                body = body.split("</think>", 1)[-1].lstrip("\n")
            parts.append(f"{ASSISTANT_PREFIX}{body}{IM_END}\n")
        else:
            raise ValueError(f"unknown role: {role}")
    if add_generation_prompt:
        parts.append(ASSISTANT_PREFIX)
    return "".join(parts)

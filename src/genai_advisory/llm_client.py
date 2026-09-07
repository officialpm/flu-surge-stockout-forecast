import os
from typing import Protocol

import anthropic
from dotenv import load_dotenv

load_dotenv()


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class FakeLLMClient:
    """Test double returning a fixed response, for deterministic tests."""

    def __init__(self, canned_response: str):
        self.canned_response = canned_response
        self.last_prompt: str | None = None

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.canned_response


class AnthropicLLMClient:
    """Real LLM client backed by the Claude API.

    Reads ANTHROPIC_API_KEY (required, via the Anthropic SDK) and an
    optional ANTHROPIC_BASE_URL from the environment/.env, so requests can
    be routed through a self-hosted gateway instead of api.anthropic.com.
    """

    def __init__(self, model: str = "claude-sonnet-5"):
        self.model = model
        self.client = anthropic.Anthropic(base_url=os.environ.get("ANTHROPIC_BASE_URL"))

    def complete(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extended-thinking-capable models prepend a ThinkingBlock with no
        # `.text` attribute; the reply text is the first TextBlock instead.
        for block in response.content:
            if block.type == "text":
                return block.text
        raise ValueError("Anthropic response contained no text block")

"""OpenAI generation strategy."""

import os
from typing import Any

from magic_generator.core import BaseGenerator, GenerationMode, GenerationPrompt, GenerationResult

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "openai package is required for OpenAIGenerator. Install with: pip install openai"
    )


class OpenAIGenerator(BaseGenerator):
    """OpenAI chat completion generator.

    Uses OpenAI's chat completion API to generate responses.

    Attributes:
        model: OpenAI model name (e.g., 'gpt-4o', 'gpt-4o-mini').
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum tokens to generate.
        system_prompt: Default system prompt.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that answers questions based on the provided context."

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._default_system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        if not self._api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key.")

        self._client = OpenAI(api_key=self._api_key)

    @property
    def name(self) -> str:
        return f"openai_{self.model}"

    @property
    def description(self) -> str:
        return f"OpenAI generator ({self.model})"

    def generate(self, prompt: GenerationPrompt) -> GenerationResult:
        """Generate a response using OpenAI.

        Args:
            prompt: The generation prompt.

        Returns:
            GenerationResult with the generated response.
        """
        system = prompt.system_prompt or self._default_system_prompt

        # Build messages
        messages = [{"role": "system", "content": system}]

        # Add context if present
        if prompt.context:
            user_content = self._build_contextual_prompt(prompt)
        else:
            user_content = prompt.user_prompt

        messages.append({"role": "user", "content": user_content})

        # Generate
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        response_text = response.choices[0].message.content or ""

        return GenerationResult(
            response=response_text,
            prompt=prompt,
            mode=GenerationMode.CONTEXTUAL if prompt.context else GenerationMode.DIRECT,
            metadata={
                "model": self.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "finish_reason": response.choices[0].finish_reason,
            },
        )

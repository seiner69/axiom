"""Anthropic Claude generation strategy."""

import os

from magic_generator.core import BaseGenerator, GenerationMode, GenerationPrompt, GenerationResult

try:
    from anthropic import Anthropic
except ImportError:
    raise ImportError(
        "anthropic package is required for AnthropicGenerator. Install with: pip install anthropic"
    )


class AnthropicGenerator(BaseGenerator):
    """Anthropic Claude chat completion generator.

    Uses Anthropic's Claude API to generate responses.

    Attributes:
        model: Anthropic model name (e.g., 'claude-sonnet-4-6', 'claude-opus-4-6').
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
        temperature: Sampling temperature (0-1).
        max_tokens: Maximum tokens to generate.
        system_prompt: Default system prompt.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that answers questions based on the provided context."

    def __init__(
        self,
        model: str = "claude-sonnet-4-6-20251101",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._default_system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        if not self._api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )

        self._client = Anthropic(api_key=self._api_key)

    @property
    def name(self) -> str:
        return f"anthropic_{self.model}"

    @property
    def description(self) -> str:
        return f"Anthropic generator ({self.model})"

    def generate(self, prompt: GenerationPrompt) -> GenerationResult:
        """Generate a response using Anthropic Claude.

        Args:
            prompt: The generation prompt.

        Returns:
            GenerationResult with the generated response.
        """
        system = prompt.system_prompt or self._default_system_prompt

        # Build user content
        if prompt.context:
            user_content = self._build_contextual_prompt(prompt)
        else:
            user_content = prompt.user_prompt

        # Generate
        response = self._client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        response_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )

        return GenerationResult(
            response=response_text,
            prompt=prompt,
            mode=GenerationMode.CONTEXTUAL if prompt.context else GenerationMode.DIRECT,
            metadata={
                "model": self.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "stop_reason": response.stop_reason,
            },
        )

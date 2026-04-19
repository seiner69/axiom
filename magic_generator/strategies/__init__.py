"""All generation strategies."""

from magic_generator.strategies.anthropic import AnthropicGenerator
from magic_generator.strategies.openai import OpenAIGenerator

__all__ = ["OpenAIGenerator", "AnthropicGenerator"]

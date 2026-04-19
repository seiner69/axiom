"""magic-generator: A modular LLM generation library for RAG.

Provides LLM generation strategies for answering questions based on retrieved context.

Example:
    >>> from magic_generator.core import GenerationPrompt
    >>> from magic_generator.strategies import OpenAIGenerator
    >>>
    >>> generator = OpenAIGenerator(model="gpt-4o-mini")
    >>> prompt = GenerationPrompt(
    ...     user_prompt="What is the revenue?",
    ...     context=["Company X revenue was $100M in 2024."]
    ... )
    >>> result = generator.generate(prompt)
    >>> print(result.response)
"""

from magic_generator.core import (
    BaseGenerator,
    GenerationMode,
    GenerationPrompt,
    GenerationResult,
)
from magic_generator.strategies import AnthropicGenerator, OpenAIGenerator

__all__ = [
    # Core
    "BaseGenerator",
    "GenerationMode",
    "GenerationPrompt",
    "GenerationResult",
    # Strategies
    "OpenAIGenerator",
    "AnthropicGenerator",
]

__version__ = "0.1.0"

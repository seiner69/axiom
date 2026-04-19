"""magic-generator core interfaces and data classes."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GenerationMode(Enum):
    """Generation mode enumeration."""

    DIRECT = "direct"
    CONTEXTUAL = "contextual"
    CONDENSE = "condense"
    STEP_BACK = "step_back"


@dataclass
class GenerationPrompt:
    """A prompt for generation.

    Attributes:
        system_prompt: System-level instructions.
        user_prompt: User query with context.
        context: Optional retrieved context chunks.
    """

    system_prompt: str = ""
    user_prompt: str = ""
    context: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "context": self.context,
        }


@dataclass
class GenerationResult:
    """Result of a generation operation.

    Attributes:
        response: The generated text response.
        prompt: The prompt that was used.
        mode: Generation mode used.
        metadata: Additional metadata.
    """

    response: str = ""
    prompt: GenerationPrompt = field(default_factory=GenerationPrompt)
    mode: GenerationMode = GenerationMode.DIRECT
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "response": self.response,
            "prompt": self.prompt.to_dict(),
            "mode": self.mode.value,
            "metadata": self.metadata,
        }


from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """Abstract base class for all generators.

    All generators must implement the `generate` method.
    """

    @abstractmethod
    def generate(self, prompt: GenerationPrompt) -> GenerationResult:
        """Generate a response for the given prompt.

        Args:
            prompt: The generation prompt.

        Returns:
            GenerationResult with the generated response.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the generator."""
        ...

    @property
    def description(self) -> str:
        """Return a short description."""
        return ""

    def _build_contextual_prompt(self, prompt: GenerationPrompt) -> str:
        """Build a contextual prompt with retrieved context."""
        if not prompt.context:
            return prompt.user_prompt

        context_text = "\n\n".join(f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(prompt.context))
        return (
            f"Context information:\n{context_text}\n\n"
            f"---\n\n"
            f"User question: {prompt.user_prompt}\n\n"
            f"Based on the context information above, please answer the question. "
            f"If the context doesn't contain relevant information, say so."
        )

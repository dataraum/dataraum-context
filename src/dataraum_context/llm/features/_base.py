"""Base class for LLM features with common functionality."""

from dataraum_context.llm.cache import LLMCache
from dataraum_context.llm.config import LLMConfig
from dataraum_context.llm.prompts import PromptRenderer
from dataraum_context.llm.providers.base import LLMProvider


class LLMFeature:
    """Base class for LLM features.

    Provides common functionality:
    - Configuration access
    - Provider access
    - Prompt rendering
    - Cache access
    """

    def __init__(
        self,
        config: LLMConfig,
        provider: LLMProvider,
        prompt_renderer: PromptRenderer,
        cache: LLMCache,
    ):
        """Initialize LLM feature.

        Args:
            config: LLM configuration
            provider: LLM provider instance
            prompt_renderer: Prompt template renderer
            cache: Response cache
        """
        self.config = config
        self.provider = provider
        self.renderer = prompt_renderer
        self.cache = cache

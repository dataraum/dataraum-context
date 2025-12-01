"""LLM provider implementations and factory."""

from typing import Any

from dataraum_context.llm.providers.base import LLMProvider, LLMRequest, LLMResponse

__all__ = ["LLMProvider", "LLMRequest", "LLMResponse", "create_provider"]


def create_provider(provider_name: str, provider_config: dict[str, Any]) -> LLMProvider:
    """Create LLM provider based on configuration.

    Args:
        provider_name: Provider name ('anthropic', 'openai', 'local')
        provider_config: Provider-specific configuration dict

    Returns:
        Initialized LLM provider

    Raises:
        ValueError: If provider name is unknown
        NotImplementedError: If provider is not yet implemented
    """
    if provider_name == "anthropic":
        from dataraum_context.llm.providers.anthropic import AnthropicConfig, AnthropicProvider

        config = AnthropicConfig(**provider_config)
        return AnthropicProvider(config)

    elif provider_name == "openai":
        from dataraum_context.llm.providers.openai import OpenAIConfig, OpenAIProvider

        config = OpenAIConfig(**provider_config)
        return OpenAIProvider(config)

    elif provider_name == "local":
        from dataraum_context.llm.providers.local import LocalConfig, LocalProvider

        config = LocalConfig(**provider_config)
        return LocalProvider(config)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. Supported providers: anthropic, openai, local"
        )

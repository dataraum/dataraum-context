"""Prompt template loading and rendering.

Loads prompt templates from config/prompts/*.yaml and renders them
with context variables using string formatting.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """A prompt template from YAML."""

    name: str
    version: str
    description: str
    temperature: float
    prompt: str  # Template with {variable} placeholders
    inputs: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, list[str]] = Field(default_factory=dict)


class PromptRenderer:
    """Render prompt templates with context variables.

    Loads templates from YAML files and caches them in memory.
    Supports variable substitution and default values.
    """

    def __init__(self, prompts_dir: Path | None = None):
        """Initialize prompt renderer.

        Args:
            prompts_dir: Directory containing prompt YAML files.
                        If None, uses config/prompts/
        """
        if prompts_dir is None:
            prompts_dir = Path("config/prompts")

        self.prompts_dir = prompts_dir
        self._cache: dict[str, PromptTemplate] = {}

    def load_template(self, name: str) -> PromptTemplate:
        """Load and cache a prompt template.

        Args:
            name: Template name (without .yaml extension)

        Returns:
            Loaded prompt template

        Raises:
            FileNotFoundError: If template file doesn't exist
            yaml.YAMLError: If YAML is invalid
            pydantic.ValidationError: If template doesn't match schema
        """
        if name in self._cache:
            return self._cache[name]

        template_path = self.prompts_dir / f"{name}.yaml"
        if not template_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {template_path}. "
                f"Available templates: {self._list_templates()}"
            )

        with open(template_path) as f:
            data = yaml.safe_load(f)

        template = PromptTemplate(**data)
        self._cache[name] = template
        return template

    def render(self, template_name: str, context: dict[str, Any]) -> tuple[str, float]:
        """Render a prompt with context variables.

        Args:
            template_name: Name of template to render
            context: Context variables for substitution

        Returns:
            Tuple of (rendered_prompt, temperature)

        Raises:
            ValueError: If required inputs are missing
            KeyError: If template has undefined variables
        """
        template = self.load_template(template_name)

        # Validate required inputs
        for input_name, input_spec in template.inputs.items():
            if input_spec.get("required", False) and input_name not in context:
                raise ValueError(
                    f"Missing required input '{input_name}' for template '{template_name}'"
                )

        # Fill in defaults for missing optional inputs
        full_context = {}
        for input_name, input_spec in template.inputs.items():
            if input_name in context:
                full_context[input_name] = context[input_name]
            elif "default" in input_spec:
                full_context[input_name] = input_spec["default"]

        # Render template using string formatting
        try:
            rendered = template.prompt.format(**full_context)
        except KeyError as e:
            raise KeyError(
                f"Template '{template_name}' has undefined variable: {e}. "
                f"Available context: {list(full_context.keys())}"
            ) from e

        return rendered, template.temperature

    def _list_templates(self) -> list[str]:
        """List available template names."""
        if not self.prompts_dir.exists():
            return []

        return [p.stem for p in self.prompts_dir.glob("*.yaml")]

    def clear_cache(self) -> None:
        """Clear the template cache.

        Useful for development when templates are being modified.
        """
        self._cache.clear()

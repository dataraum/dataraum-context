"""Rule loader for quality rules configurations.

Loads and parses quality rules from YAML configuration files.
Supports loading individual rule sets or discovering all available rules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from dataraum_context.core.models.base import Result
from dataraum_context.quality.rules.models import RulesConfig

# =============================================================================
# Configuration Paths
# =============================================================================


def get_rules_directory() -> Path:
    """Get the path to the rules configuration directory.

    Returns:
        Path to config/rules/ directory
    """
    # Assume we're in src/dataraum_context/quality/rules/
    # Navigate to project root, then to config/rules/
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent.parent
    rules_dir = project_root / "config" / "rules"

    return rules_dir


def list_available_rules() -> list[str]:
    """List all available rule configuration names.

    Scans the config/rules/ directory for YAML files.

    Returns:
        List of rule configuration names (without .yaml extension)
    """
    rules_dir = get_rules_directory()

    if not rules_dir.exists():
        return []

    rule_files = rules_dir.glob("*.yaml")
    return [f.stem for f in rule_files]


# =============================================================================
# Rule Loading Functions
# =============================================================================


def load_rules_yaml(rules_path: Path) -> Result[dict[str, Any]]:
    """Load raw YAML data from a rules file.

    Args:
        rules_path: Path to the rules YAML file

    Returns:
        Result containing the parsed YAML data as a dictionary
    """
    if not rules_path.exists():
        return Result.fail(f"Rules file not found: {rules_path}")

    try:
        with open(rules_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            return Result.fail(f"Rules file is empty: {rules_path}")

        if not isinstance(data, dict):
            return Result.fail(f"Rules file must contain a YAML mapping: {rules_path}")

        return Result.ok(data)

    except yaml.YAMLError as e:
        return Result.fail(f"Failed to parse YAML in {rules_path}: {e}")
    except OSError as e:
        return Result.fail(f"Failed to read rules file {rules_path}: {e}")


def parse_rules_config(data: dict[str, Any], source_name: str = "unknown") -> Result[RulesConfig]:
    """Parse raw YAML data into a RulesConfig object.

    Args:
        data: Raw YAML data as dictionary
        source_name: Name of the source (for error messages)

    Returns:
        Result containing the parsed RulesConfig
    """
    try:
        config = RulesConfig(**data)
        return Result.ok(config)

    except ValidationError as e:
        error_details = []
        for error in e.errors():
            location = " -> ".join(str(loc) for loc in error["loc"])
            error_details.append(f"{location}: {error['msg']}")

        error_msg = f"Validation errors in {source_name}:\n" + "\n".join(error_details)
        return Result.fail(error_msg)

    except Exception as e:
        return Result.fail(f"Unexpected error parsing {source_name}: {e}")


def load_rules_config(rules_name: str) -> Result[RulesConfig]:
    """Load a rules configuration by name.

    Loads from config/rules/{rules_name}.yaml and parses into RulesConfig.

    Args:
        rules_name: Name of the rules configuration (without .yaml extension)
                    e.g., "default", "financial", "healthcare"

    Returns:
        Result containing the loaded RulesConfig

    Examples:
        >>> result = load_rules_config("default")
        >>> if result.success:
        ...     config = result.unwrap()
        ...     print(f"Loaded {config.name} v{config.version}")
    """
    rules_dir = get_rules_directory()
    rules_path = rules_dir / f"{rules_name}.yaml"

    # Load YAML
    yaml_result = load_rules_yaml(rules_path)
    if not yaml_result.success:
        return Result.fail(yaml_result.error or "Unknown error loading YAML")

    # Parse into config
    data = yaml_result.unwrap()
    return parse_rules_config(data, source_name=f"{rules_name}.yaml")


def load_rules_from_path(rules_path: str | Path) -> Result[RulesConfig]:
    """Load a rules configuration from an absolute path.

    Useful for loading rules from custom locations or during testing.

    Args:
        rules_path: Absolute path to the rules YAML file

    Returns:
        Result containing the loaded RulesConfig

    Examples:
        >>> result = load_rules_from_path("/path/to/my_rules.yaml")
        >>> if result.success:
        ...     config = result.unwrap()
    """
    path = Path(rules_path)

    # Load YAML
    yaml_result = load_rules_yaml(path)
    if not yaml_result.success:
        return Result.fail(yaml_result.error or "Unknown error loading YAML")

    # Parse into config
    data = yaml_result.unwrap()
    return parse_rules_config(data, source_name=str(path))


def load_all_rules() -> Result[dict[str, RulesConfig]]:
    """Load all available rules configurations.

    Scans config/rules/ and loads all YAML files.

    Returns:
        Result containing a dictionary mapping rule names to RulesConfig objects
        If any rule fails to load, returns a failure Result with details

    Examples:
        >>> result = load_all_rules()
        >>> if result.success:
        ...     all_rules = result.unwrap()
        ...     for name, config in all_rules.items():
        ...         print(f"{name}: {config.description}")
    """
    available = list_available_rules()

    if not available:
        return Result.fail("No rule configurations found in config/rules/")

    configs: dict[str, RulesConfig] = {}
    errors: list[str] = []
    warnings: list[str] = []

    for rule_name in available:
        result = load_rules_config(rule_name)

        if result.success:
            configs[rule_name] = result.unwrap()
            warnings.extend(result.warnings)
        else:
            errors.append(f"{rule_name}: {result.error}")

    if errors:
        error_msg = "Failed to load some rule configurations:\n" + "\n".join(errors)
        return Result.fail(error_msg)

    return Result.ok(configs, warnings=warnings)


# =============================================================================
# Utility Functions
# =============================================================================


def merge_rules_configs(*configs: RulesConfig) -> RulesConfig:
    """Merge multiple rules configurations into one.

    Later configs override earlier ones for conflicts.
    Used for combining default rules with domain-specific rules.

    Args:
        *configs: Variable number of RulesConfig objects to merge

    Returns:
        Merged RulesConfig

    Examples:
        >>> default_rules = load_rules_config("default").unwrap()
        >>> financial_rules = load_rules_config("financial").unwrap()
        >>> merged = merge_rules_configs(default_rules, financial_rules)
    """
    if not configs:
        return RulesConfig(name="empty", version="0.0.0", description="Empty rules config")

    if len(configs) == 1:
        return configs[0]

    # Start with first config
    base = configs[0]
    merged_data: dict[str, Any] = {
        "name": f"merged_{base.name}",
        "version": base.version,
        "description": f"Merged from {len(configs)} rule configurations",
        "role_based_rules": {},
        "type_based_rules": {},
        "pattern_based_rules": [],
        "statistical_rules": [],
        "consistency_rules": [],
        "custom_rule_templates": {},
    }

    # Merge each config
    for config in configs:
        # Role-based rules (merge by role)
        for role in ["key", "timestamp", "measure", "foreign_key", "dimension"]:
            base_rules = getattr(config.role_based_rules, role, [])
            if role not in merged_data["role_based_rules"]:
                merged_data["role_based_rules"][role] = []
            merged_data["role_based_rules"][role].extend([r.model_dump() for r in base_rules])

        # Type-based rules (merge by type)
        for type_name in [
            "DOUBLE",
            "FLOAT",
            "INTEGER",
            "BIGINT",
            "DATE",
            "TIMESTAMP",
            "VARCHAR",
            "BOOLEAN",
        ]:
            type_rules = getattr(config.type_based_rules, type_name, [])
            if type_name not in merged_data["type_based_rules"]:
                merged_data["type_based_rules"][type_name] = []
            merged_data["type_based_rules"][type_name].extend([r.model_dump() for r in type_rules])

        # Pattern-based rules (append all)
        merged_data["pattern_based_rules"].extend(
            [r.model_dump() for r in config.pattern_based_rules]
        )

        # Statistical rules (append all)
        merged_data["statistical_rules"].extend([r.model_dump() for r in config.statistical_rules])

        # Consistency rules (append all)
        merged_data["consistency_rules"].extend([r.model_dump() for r in config.consistency_rules])

        # Custom templates (merge by name, later overrides)
        for template_name, template in config.custom_rule_templates.items():
            merged_data["custom_rule_templates"][template_name] = template.model_dump()

    # Parse back into RulesConfig
    return RulesConfig(**merged_data)


def validate_rules_directory() -> Result[None]:
    """Validate that the rules directory exists and is readable.

    Returns:
        Result indicating success or failure
    """
    rules_dir = get_rules_directory()

    if not rules_dir.exists():
        return Result.fail(
            f"Rules directory does not exist: {rules_dir}\n"
            "Expected config/rules/ directory in project root"
        )

    if not os.access(rules_dir, os.R_OK):
        return Result.fail(f"Rules directory is not readable: {rules_dir}")

    # Check for at least one rule file
    yaml_files = list(rules_dir.glob("*.yaml"))
    if not yaml_files:
        return Result.fail(
            f"No YAML files found in rules directory: {rules_dir}\nExpected at least one .yaml file"
        )

    return Result.ok(None, warnings=[f"Found {len(yaml_files)} rule configuration(s)"])

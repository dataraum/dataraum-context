"""Quality rules engine for data validation.

This module provides:
- Rule configuration models (loaded from YAML)
- Rule evaluation results models
- Rule loader (loads rules from config files)
- Rule evaluator (evaluates rules against data in DuckDB)
"""

from dataraum_context.quality.rules.evaluator import (
    RuleEvaluator,
    evaluate_table_rules,
    load_table_metadata,
)
from dataraum_context.quality.rules.loader import (
    get_rules_directory,
    list_available_rules,
    load_all_rules,
    load_rules_config,
    load_rules_from_path,
    merge_rules_configs,
    validate_rules_directory,
)
from dataraum_context.quality.rules.models import (
    ConsistencyRule,
    ConsistencyRulePattern,
    CustomRuleTemplate,
    DatasetRuleResults,
    PatternBasedRule,
    RoleBasedRules,
    RuleDefinition,
    RuleResult,
    RulesConfig,
    RuleViolation,
    StatisticalRule,
    TableRuleResults,
    TypeBasedRules,
)

__all__ = [
    # Configuration models
    "RulesConfig",
    "RuleDefinition",
    "RoleBasedRules",
    "TypeBasedRules",
    "PatternBasedRule",
    "StatisticalRule",
    "ConsistencyRule",
    "ConsistencyRulePattern",
    "CustomRuleTemplate",
    # Result models
    "RuleViolation",
    "RuleResult",
    "TableRuleResults",
    "DatasetRuleResults",
    # Loader functions
    "load_rules_config",
    "load_rules_from_path",
    "load_all_rules",
    "list_available_rules",
    "get_rules_directory",
    "merge_rules_configs",
    "validate_rules_directory",
    # Evaluator functions and classes
    "RuleEvaluator",
    "evaluate_table_rules",
    "load_table_metadata",
]

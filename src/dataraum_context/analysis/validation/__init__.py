"""Generic Validation Module.

LLM-powered validation checks using semantic annotations for column resolution.
"""

from dataraum_context.analysis.validation.agent import ValidationAgent
from dataraum_context.analysis.validation.config import (
    clear_cache,
    format_validation_specs_for_context,
    get_validation_spec,
    get_validation_specs_by_category,
    get_validation_specs_by_tags,
    get_validations_config_path,
    load_all_validation_specs,
)
from dataraum_context.analysis.validation.db_models import (
    ValidationResultRecord,
    ValidationRunRecord,
    ValidationSQLCache,
)
from dataraum_context.analysis.validation.models import (
    ColumnRequirement,
    ColumnResolutionResult,
    GeneratedSQL,
    ResolvedColumn,
    ValidationResult,
    ValidationRunResult,
    ValidationSeverity,
    ValidationSpec,
    ValidationStatus,
)
from dataraum_context.analysis.validation.resolver import (
    get_table_schema_for_llm,
    resolve_columns,
)

__all__ = [
    # Agent
    "ValidationAgent",
    # Config
    "get_validations_config_path",
    "load_all_validation_specs",
    "get_validation_spec",
    "get_validation_specs_by_category",
    "get_validation_specs_by_tags",
    "format_validation_specs_for_context",
    "clear_cache",
    # Resolver
    "resolve_columns",
    "get_table_schema_for_llm",
    # Models
    "ColumnRequirement",
    "ValidationSpec",
    "ResolvedColumn",
    "ColumnResolutionResult",
    "GeneratedSQL",
    "ValidationResult",
    "ValidationRunResult",
    "ValidationSeverity",
    "ValidationStatus",
    # DB Models
    "ValidationSQLCache",
    "ValidationRunRecord",
    "ValidationResultRecord",
]

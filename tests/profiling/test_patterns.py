"""Tests for pattern detection."""

import pytest

from dataraum_context.core.models import DataType
from dataraum_context.profiling.patterns import load_pattern_config


class TestPatternConfig:
    """Tests for pattern configuration loading."""

    def test_load_default_config(self):
        """Test loading default pattern configuration."""
        config = load_pattern_config()

        patterns = config.get_value_patterns()
        assert len(patterns) > 0

        column_patterns = config.get_column_name_patterns()
        assert len(column_patterns) > 0

    def test_match_iso_date(self):
        """Test matching ISO date pattern."""
        config = load_pattern_config()

        matches = config.match_value("2024-01-15")
        assert len(matches) > 0

        iso_date_match = next((p for p in matches if p.name == "iso_date"), None)
        assert iso_date_match is not None
        assert iso_date_match.inferred_type == DataType.DATE

    def test_match_email(self):
        """Test matching email pattern."""
        config = load_pattern_config()

        matches = config.match_value("user@example.com")
        assert len(matches) > 0

        email_match = next((p for p in matches if p.name == "email"), None)
        assert email_match is not None
        assert email_match.inferred_type == DataType.VARCHAR
        assert email_match.semantic_type == "identifier"
        assert email_match.pii is True

    def test_match_uuid(self):
        """Test matching UUID pattern."""
        config = load_pattern_config()

        matches = config.match_value("550e8400-e29b-41d4-a716-446655440000")
        assert len(matches) > 0

        uuid_match = next((p for p in matches if p.name == "uuid"), None)
        assert uuid_match is not None
        assert uuid_match.semantic_type == "key"

    def test_match_currency_usd(self):
        """Test matching USD currency pattern."""
        config = load_pattern_config()

        matches = config.match_value("$1,234.56")
        assert len(matches) > 0

        currency_match = next((p for p in matches if p.name == "currency_usd"), None)
        assert currency_match is not None
        assert currency_match.inferred_type == DataType.DOUBLE
        assert currency_match.detected_unit == "USD"

    def test_match_boolean(self):
        """Test matching boolean patterns."""
        config = load_pattern_config()

        # Test true/false
        matches = config.match_value("true")
        boolean_match = next((p for p in matches if "boolean" in p.name), None)
        assert boolean_match is not None
        assert boolean_match.inferred_type == DataType.BOOLEAN

        # Test yes/no
        matches = config.match_value("yes")
        boolean_match = next((p for p in matches if "boolean" in p.name), None)
        assert boolean_match is not None

    def test_match_integer(self):
        """Test matching integer pattern."""
        config = load_pattern_config()

        matches = config.match_value("12345")
        integer_match = next((p for p in matches if p.name == "integer"), None)
        assert integer_match is not None
        assert integer_match.inferred_type == DataType.BIGINT

    def test_match_decimal(self):
        """Test matching decimal pattern."""
        config = load_pattern_config()

        matches = config.match_value("123.45")
        decimal_match = next((p for p in matches if p.name == "decimal_dot"), None)
        assert decimal_match is not None
        assert decimal_match.inferred_type == DataType.DOUBLE

    def test_match_column_name_id(self):
        """Test matching column name patterns for IDs."""
        config = load_pattern_config()

        matches = config.match_column_name("customer_id")
        assert len(matches) > 0

        id_match = next((p for p in matches if "_id" in p.pattern), None)
        assert id_match is not None
        assert id_match.likely_role == "key"

    def test_match_column_name_timestamp(self):
        """Test matching column name patterns for timestamps."""
        config = load_pattern_config()

        matches = config.match_column_name("created_at")
        assert len(matches) > 0

        ts_match = next((p for p in matches if "_at" in p.pattern), None)
        assert ts_match is not None
        assert ts_match.likely_role == "timestamp"
        assert ts_match.likely_type == DataType.TIMESTAMP

    def test_match_column_name_amount(self):
        """Test matching column name patterns for amounts."""
        config = load_pattern_config()

        matches = config.match_column_name("total_amount")
        assert len(matches) > 0

        amt_match = next((p for p in matches if "amount" in p.pattern), None)
        assert amt_match is not None
        assert amt_match.likely_role == "measure"
        assert amt_match.likely_type == DataType.DOUBLE

    def test_no_match(self):
        """Test that invalid values don't match patterns."""
        config = load_pattern_config()

        matches = config.match_value("random text 123 !@#")
        # Should only match very generic patterns or none
        assert all(p.name != "iso_date" for p in matches)
        assert all(p.name != "email" for p in matches)
        assert all(p.name != "uuid" for p in matches)

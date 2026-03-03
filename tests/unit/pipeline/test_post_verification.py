"""Tests for post-verification: phases declare post_verification dimensions."""


# --- Phase annotations ---


class TestPhasePostVerificationAnnotations:
    def test_typing_phase_declares_type_fidelity(self):
        from dataraum.pipeline.registry import get_phase_class

        cls = get_phase_class("typing")
        assert cls is not None
        phase = cls()
        assert phase.post_verification == ["type_fidelity"]

    def test_statistics_phase_declares_null_and_outlier(self):
        from dataraum.pipeline.registry import get_phase_class

        cls = get_phase_class("statistics")
        assert cls is not None
        phase = cls()
        assert "null_ratio" in phase.post_verification
        assert "outlier_rate" in phase.post_verification

    def test_relationships_phase_declares_join_quality(self):
        from dataraum.pipeline.registry import get_phase_class

        cls = get_phase_class("relationships")
        assert cls is not None
        phase = cls()
        assert "join_path_determinism" in phase.post_verification
        assert "relationship_quality" in phase.post_verification

    def test_semantic_phase_declares_naming_unit(self):
        from dataraum.pipeline.registry import get_phase_class

        cls = get_phase_class("semantic")
        assert cls is not None
        phase = cls()
        assert "naming_clarity" in phase.post_verification
        assert "unit_declaration" in phase.post_verification

    def test_import_phase_has_no_post_verification(self):
        from dataraum.pipeline.registry import get_phase_class

        cls = get_phase_class("import")
        assert cls is not None
        phase = cls()
        assert phase.post_verification == []

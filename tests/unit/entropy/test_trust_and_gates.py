"""Tests for detector trust levels and gate data model."""

from dataraum.entropy.detectors import (
    DetectorTrust,
    get_default_registry,
)
from dataraum.entropy.detectors.base import DetectorContext, EntropyDetector
from dataraum.entropy.models import EntropyObject
from dataraum.pipeline.base import PhaseStatus

# --- Trust Level Classification ---


EXPECTED_HARD = {
    "type_fidelity",
    "join_path_determinism",
    "relationship_entropy",
    "null_ratio",
    "outlier_rate",
    "benford",
    "temporal_drift",
    "derived_value",
}

EXPECTED_SOFT = {
    "business_meaning",
    "unit_entropy",
    "temporal_entropy",
    "dimensional_entropy",
}


class TestDetectorTrustLevels:
    """Verify each detector has the correct trust level."""

    def test_hard_detectors(self):
        registry = get_default_registry()
        hard_ids = {d.detector_id for d in registry.get_hard_detectors()}
        assert hard_ids == EXPECTED_HARD

    def test_soft_detectors(self):
        registry = get_default_registry()
        soft_ids = {d.detector_id for d in registry.get_soft_detectors()}
        assert soft_ids == EXPECTED_SOFT

    def test_all_detectors_classified(self):
        """Every registered detector must be either HARD or SOFT."""
        registry = get_default_registry()
        for d in registry.get_all_detectors():
            assert d.trust_level in (DetectorTrust.HARD, DetectorTrust.SOFT), (
                f"{d.detector_id} has unknown trust level: {d.trust_level}"
            )

    def test_is_verifier_matches_hard(self):
        registry = get_default_registry()
        for d in registry.get_all_detectors():
            assert d.is_verifier == (d.trust_level == DetectorTrust.HARD), (
                f"{d.detector_id}: is_verifier={d.is_verifier} but trust_level={d.trust_level}"
            )

    def test_default_trust_is_soft(self):
        """A detector with no explicit trust_level defaults to SOFT."""

        class BareDetector(EntropyDetector):
            detector_id = "bare"
            layer = "test"
            dimension = "test"
            sub_dimension = "test"

            def detect(self, context: DetectorContext) -> list[EntropyObject]:
                return []

        d = BareDetector()
        assert d.trust_level == DetectorTrust.SOFT
        assert d.is_verifier is False


# --- PhaseStatus ---


class TestPhaseStatus:
    def test_all_statuses(self):
        statuses = set(PhaseStatus)
        assert len(statuses) == 5  # pending, running, completed, failed, skipped

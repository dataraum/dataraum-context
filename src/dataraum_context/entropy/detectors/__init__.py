"""Entropy detectors for measuring uncertainty across dimensions.

This module provides the detector infrastructure for the entropy layer:
- EntropyDetector: Abstract base class for all detectors
- DetectorRegistry: Registry for detector discovery and management
- Built-in detectors for structural, semantic, value, and computational entropy

Usage:
    from dataraum_context.entropy.detectors import (
        EntropyDetector,
        DetectorRegistry,
        get_default_registry,
    )

    # Get all registered detectors
    registry = get_default_registry()
    detectors = registry.get_detectors_for_layer("structural")

    # Run a specific detector
    detector = registry.get_detector("type_fidelity")
    results = await detector.detect(column_data)
"""

from dataraum_context.entropy.detectors.base import (
    DetectorRegistry,
    EntropyDetector,
    get_default_registry,
)

__all__ = [
    "EntropyDetector",
    "DetectorRegistry",
    "get_default_registry",
]

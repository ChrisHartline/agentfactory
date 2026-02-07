"""
Template loading utilities.

Loads both YAML base templates and expanded Python-dict templates
into a PromptRegistry.
"""

from __future__ import annotations

from pathlib import Path

from ..registry import PromptRegistry

TEMPLATES_DIR = Path(__file__).parent


def load_all_templates(registry: PromptRegistry | None = None) -> PromptRegistry:
    """
    Load all templates (YAML base + expanded overrides) into a registry.

    Returns a populated PromptRegistry ready for use.
    """
    if registry is None:
        registry = PromptRegistry()

    # YAML base templates
    yaml_path = TEMPLATES_DIR / "registry.yaml"
    if yaml_path.exists():
        registry.load_from_yaml(yaml_path)

    # Expanded v2 templates (override YAML entries with richer versions)
    from .expanded import EXPANDED_TEMPLATES

    registry.load_from_dict(EXPANDED_TEMPLATES)

    return registry

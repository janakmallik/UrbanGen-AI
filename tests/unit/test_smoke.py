"""Smoke tests for UrbanGen-AI package structure.

These tests verify the package imports correctly and that config helpers
are available. They run without needing BDD100K data or a CUDA GPU.
"""

from __future__ import annotations


def test_package_imports():
    import urbangen  # noqa: F401
    import urbangen.config  # noqa: F401


def test_config_helpers_exist():
    from urbangen import config

    assert callable(config.data_root)
    assert callable(config.urbanmind_root)


def test_config_docstrings():
    from urbangen import config

    assert config.data_root.__doc__ is not None
    assert config.urbanmind_root.__doc__ is not None

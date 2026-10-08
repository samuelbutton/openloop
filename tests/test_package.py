"""Smoke test for the installed package."""

from importlib import import_module


def test_package_is_importable():
    import_module("openloop")

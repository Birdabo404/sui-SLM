#!/usr/bin/env python3
"""Dependency declaration tests that do not contact package indexes."""
import os
import re
import unittest
from importlib import import_module
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXACT_REQUIREMENT = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^=<>!~\s]+$")


def requirements():
    return [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def enhanced_requirements():
    return [
        line.strip()
        for line in (ROOT / "requirements-enhanced.txt").read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
        and not line.lstrip().startswith(("#", "-r "))
    ]


class RequirementTests(unittest.TestCase):
    def test_every_direct_dependency_is_exactly_pinned(self):
        declared = requirements()
        self.assertGreater(len(declared), 0)
        self.assertEqual(
            [requirement for requirement in declared if not EXACT_REQUIREMENT.fullmatch(requirement)],
            [],
        )

    def test_every_enhanced_dependency_is_exactly_pinned(self):
        declared = enhanced_requirements()
        self.assertGreater(len(declared), 0)
        self.assertEqual(
            [
                requirement
                for requirement in declared
                if not EXACT_REQUIREMENT.fullmatch(requirement)
            ],
            [],
        )

    @unittest.skipUnless(
        os.environ.get("SUI_SLM_CHECK_PINS") == "1",
        "set SUI_SLM_CHECK_PINS=1 to verify installed package versions",
    )
    def test_clean_environment_versions_match_the_pins(self):
        for requirement in requirements():
            name, expected = requirement.split("==", 1)
            with self.subTest(package=name):
                self.assertEqual(version(name).split("+", 1)[0], expected)

    @unittest.skipUnless(
        os.environ.get("SUI_SLM_CHECK_ENHANCED") == "1",
        "set SUI_SLM_CHECK_ENHANCED=1 to verify the optional grounding stack",
    )
    def test_pinned_grounding_stack_imports_together(self):
        huggingface_hub = import_module("huggingface_hub")
        torch = import_module("torch")
        transformers = import_module("transformers")

        self.assertEqual(huggingface_hub.__version__, "0.36.2")
        self.assertEqual(transformers.__version__, "4.57.6")
        self.assertEqual(torch.__version__.split("+", 1)[0], "2.9.1")
        self.assertIsNotNone(transformers.AutoModelForSequenceClassification)


if __name__ == "__main__":
    unittest.main()

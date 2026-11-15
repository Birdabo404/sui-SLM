#!/usr/bin/env python3
"""Dependency declaration tests that do not contact package indexes."""
import re
import unittest
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


class RequirementTests(unittest.TestCase):
    def test_every_direct_dependency_is_exactly_pinned(self):
        declared = requirements()
        self.assertGreater(len(declared), 0)
        self.assertEqual(
            [requirement for requirement in declared if not EXACT_REQUIREMENT.fullmatch(requirement)],
            [],
        )

    def test_clean_environment_versions_match_the_pins(self):
        for requirement in requirements():
            name, expected = requirement.split("==", 1)
            with self.subTest(package=name):
                self.assertEqual(version(name).split("+", 1)[0], expected)


if __name__ == "__main__":
    unittest.main()

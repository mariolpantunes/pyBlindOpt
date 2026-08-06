"""Tests for the packaging metadata.

Two things here have gone wrong silently before, which is why they are pinned:
`requirements.txt` drifted from `install_requires` (a `numpy>=1.6.4` typo
against `numpy>=1.26.4`), and the version reported by `pyBlindOpt.__version__`
drifted from the one `setup.cfg` publishes (0.3.0 shipped reporting 0.2.0).
Neither breaks a test, so neither surfaces without being asserted.
"""

import configparser
import os
import re
import unittest

import pyBlindOpt

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)


def _name_of(spec):
    """`"numpy>=2.0.0"` -> `"numpy"`."""
    return re.split(r"[<>=~!\[ ]", spec, maxsplit=1)[0].strip().lower()


class TestPackaging(unittest.TestCase):
    def setUp(self):
        self.cfg = configparser.ConfigParser()
        self.cfg.read(os.path.join(ROOT, "setup.cfg"), encoding="utf-8")
        raw = self.cfg["options"]["install_requires"]
        self.required = [line.strip() for line in raw.splitlines() if line.strip()]
        with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as handle:
            self.declared = {_name_of(line): line.strip()
                             for line in handle if line.strip()}

    def test_requirements_covers_every_runtime_dependency(self):
        missing = {_name_of(s) for s in self.required} - set(self.declared)
        self.assertFalse(
            missing, f"in setup.cfg but not requirements.txt: {sorted(missing)}")

    def test_requirements_declares_nothing_extra(self):
        """It is the runtime dependency list, not a development environment."""
        extra = set(self.declared) - {_name_of(s) for s in self.required}
        self.assertFalse(
            extra, f"in requirements.txt but not setup.cfg: {sorted(extra)}")

    def test_runtime_floors_agree_between_the_two_files(self):
        for spec in self.required:
            with self.subTest(dependency=spec):
                self.assertEqual(spec, self.declared[_name_of(spec)])

    def test_reported_version_matches_the_published_one(self):
        self.assertEqual(pyBlindOpt.__version__, self.cfg["metadata"]["version"])

    def test_contact_address_is_the_one_setup_cfg_publishes(self):
        self.assertEqual(pyBlindOpt.__email__,
                         self.cfg["metadata"]["author_email"])

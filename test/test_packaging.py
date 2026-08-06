"""Tests for the packaging metadata.

Two things here have gone wrong silently before, which is why they are pinned:
`requirements.txt` drifted from `install_requires` (a `numpy>=1.6.4` typo
against `numpy>=1.26.4`), and the version reported by `pyBlindOpt.__version__`
drifted from the one `setup.cfg` publishes (0.3.0 shipped reporting 0.2.0).
Neither breaks a test, so neither surfaces without being asserted.
"""

import configparser
import os
import unittest

import pyBlindOpt

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)


def _requirement_lines(path):
    """`name -> full specifier`, skipping comments and blanks."""
    out = {}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line:
                name = line.split("=")[0].split(">")[0].split("<")[0].split("[")[0]
                out[name.strip().lower()] = line
    return out


class TestPackaging(unittest.TestCase):
    def setUp(self):
        self.cfg = configparser.ConfigParser()
        self.cfg.read(os.path.join(ROOT, "setup.cfg"), encoding="utf-8")
        self.declared = _requirement_lines(
            os.path.join(ROOT, "requirements.txt"))

    def _install_requires(self):
        raw = self.cfg["options"]["install_requires"]
        return {line.split("=")[0].split(">")[0].strip().lower(): line.strip()
                for line in raw.splitlines() if line.strip()}

    def test_requirements_covers_every_runtime_dependency(self):
        missing = set(self._install_requires()) - set(self.declared)
        self.assertFalse(
            missing, f"in setup.cfg but not requirements.txt: {sorted(missing)}")

    def test_runtime_floors_agree_between_the_two_files(self):
        for name, spec in self._install_requires().items():
            with self.subTest(dependency=name):
                self.assertEqual(spec, self.declared[name])

    def test_reported_version_matches_the_published_one(self):
        self.assertEqual(pyBlindOpt.__version__, self.cfg["metadata"]["version"])

    def test_contact_address_is_the_one_setup_cfg_publishes(self):
        self.assertEqual(pyBlindOpt.__email__,
                         self.cfg["metadata"]["author_email"])

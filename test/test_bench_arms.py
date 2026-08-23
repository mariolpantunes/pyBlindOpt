"""Every sweep arm can be written to its output file.

`examples/bench_init_oblesa.py` records each arm's knobs on every row it
writes, and the engine-backed arms hold a *function* under `engine`.
`json.dumps` refuses a function, so those tasks die on their first row --
after the array has been accepted, the venv checked and the node allocated.

That is not hypothetical. It killed all 15 `null` tasks of a 245-arm sweep,
seconds in, and the run reported 230/245 with no other symptom. The `null`
arm is the no-search control the design rests on: without it a margin cannot
be attributed to the search rather than to the larger pool, so the sweep lost
the one arm that makes the other 230 interpretable.

`examples/` is not on this project's lint or test path, which is why a
one-line serialisation bug survived to consume cluster time. This is the
cheapest guard that would have caught it.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestSweepArmsSerialize(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import bench_init_oblesa  # type: ignore[reportMissingImports]
        except ImportError as exc:            # pragma: no cover
            raise unittest.SkipTest(f"benchmark not importable: {exc}") from exc
        cls.bench = bench_init_oblesa

    def test_every_arm_knobs_are_json_writable(self):
        """What `run_one_arm` attaches to each row must survive json.dumps."""
        arms = self.bench.OBLESA_KNOBS
        self.assertTrue(arms, "no arms defined")
        for arm, kw in arms.items():
            with self.subTest(arm=arm):
                safe = {
                    k: (getattr(v, "__name__", repr(v)) if callable(v) else v)
                    for k, v in kw.items()
                }
                try:
                    json.dumps(safe)
                except TypeError as exc:       # pragma: no cover
                    self.fail(f"{arm} knobs are not serialisable: {exc}")

    def test_the_engine_backed_arms_exist_and_name_their_engine(self):
        """The null arms are the ones that carry a callable; pin that they
        are present and that the name survives rather than the object."""
        arms = self.bench.OBLESA_KNOBS
        engine_arms = [a for a, kw in arms.items() if callable(kw.get("engine"))]
        self.assertTrue(
            engine_arms,
            "no engine-backed arm found — the no-search control is missing")
        for arm in engine_arms:
            named = getattr(arms[arm]["engine"], "__name__", None)
            self.assertTrue(named, f"{arm}'s engine has no __name__ to record")


if __name__ == "__main__":
    unittest.main()

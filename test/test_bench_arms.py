"""Every sweep arm survives the things that kill a task in its first second.

Two failures of that shape have now cost cluster time, and both were
invisible until the array was accepted, the venv checked and the nodes
allocated. This module holds the cheap guards that would have caught them.

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

import numpy as np

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


class TestSweepArmsAreAdmissible(unittest.TestCase):
    """Every guided arm asks for an attraction ESS will actually accept.

    ESS refuses a weight at which the attraction out-pulls the repulsion at
    contact -- every active point would collapse onto its most attractive
    neighbour, and the plateau detector would call that convergence. With the
    laws OBLESA pins (gaussian repulsion at alpha=5, cauchy attraction at
    alpha=2) the ceiling is exactly 5/2 = 2.5.

    A grid that crosses it does not run badly, it raises. `force_weight=3.00`
    reached the cluster in the f14 grid and killed 17 tasks seconds in, one
    per (arm, dimension) cell -- a round number chosen for the shape of the
    grid rather than from the physics. This asks the engine itself, per
    distinct weight, rather than re-deriving the rule and drifting from it.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import bench_init_oblesa  # type: ignore[reportMissingImports]

            import pyBlindOpt.init as init
        except ImportError as exc:            # pragma: no cover
            raise unittest.SkipTest(f"benchmark not importable: {exc}") from exc
        cls.bench = bench_init_oblesa
        cls.init = init

    def _probe(self, weight: float) -> None:
        """One tiny run through the same engine the sweep uses."""
        rng = np.random.default_rng(0)
        dim, m = 2, 8
        bounds = np.tile([-5.0, 5.0], (dim, 1))
        samples = rng.uniform(-5, 5, (m, dim))
        # The private engine on purpose: it is this project's own, and it is
        # the layer that pins the two laws whose ratio sets the ceiling.
        # Going through `oblesa` would test the same thing more slowly and
        # through more moving parts.
        self.init._ess_engine(
            samples, bounds, n=3, seed=0,
            scores=np.linalg.norm(samples, axis=1),
            attraction_weight=weight, epochs=1)

    def test_every_guided_arm_uses_an_admissible_force_weight(self):
        weights = sorted({float(kw["force_weight"])
                          for kw in self.bench.OBLESA_KNOBS.values()
                          if float(kw.get("force_weight", 0.0)) > 0.0})
        self.assertTrue(weights, "no guided arm found")
        for weight in weights:
            with self.subTest(force_weight=weight):
                try:
                    self._probe(weight)
                except ValueError as exc:      # pragma: no cover
                    self.fail(
                        f"force_weight={weight} is inadmissible and every arm "
                        f"carrying it would raise on the cluster: {exc}")

    def test_the_probe_would_catch_a_weight_over_the_ceiling(self):
        """The guard above is only worth having if it fires. 2.5 is the
        ceiling with OBLESA's pinned laws, so 3.0 has to raise -- and if a
        future change to either law moves the ceiling, this is what says so.
        """
        with self.assertRaises(ValueError):
            self._probe(3.0)

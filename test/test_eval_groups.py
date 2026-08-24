"""The objective is always asked for exactly `n_pop` rows.

This package is built for live neuro-evolution: an objective may be a match
of `n_pop` agents played on a server that starts only once `n_pop` players
are connected. A generation evaluated in a group of six, or one row at a
time, cannot be run there at all -- so the group size is an interface, not a
performance detail.

`hho` broke it and is fixed: it scored the rapid-dive candidates Y and Z one
row at a time inside the per-hawk loop -- 122 objective calls for five
generations, 116 of them single-row, against six for every other method. It
now scores two full populations and uses only the diving rows, which leaves
the search bit-identical (verified over 40 fixed-seed runs) and cuts it to
16 calls. That was waste, not algorithm.

`cs` is a **deliberate, documented exception**. Its abandonment step scores
only the nests `pa` selected. Padding to a full population is bit-identical
but takes the per-generation total from `n_pop + pa * n_pop` to `2 * n_pop`
-- 60% more evaluations at the default `pa`, all of them discarded. An
algorithm is not forced into the rule where that costs performance; it is
declared in its docstring instead, and asserted here so the exception stays
deliberate rather than becoming a second regression.
"""

import unittest

import numpy as np

import pyBlindOpt.cs as cs
import pyBlindOpt.de as de
import pyBlindOpt.egwo as egwo
import pyBlindOpt.ga as ga
import pyBlindOpt.gwo as gwo
import pyBlindOpt.hba as hba
import pyBlindOpt.hc as hc
import pyBlindOpt.hho as hho
import pyBlindOpt.init as init
import pyBlindOpt.pso as pso
import pyBlindOpt.rs as rs
import pyBlindOpt.sa as sa

N_POP = 30
DIM = 6


class _Recorder:
    """Objective that records the row count of every batch it is handed."""

    def __init__(self):
        self.batches = []

    def __call__(self, x):
        x = np.asarray(x)
        if x.ndim == 2:
            self.batches.append(x.shape[0])
            return np.sum(x * x, axis=1)
        self.batches.append(1)
        return float(np.sum(x * x))


#: Every optimizer that evaluates whole populations. `cs` is excluded and
#: tested separately -- see the module docstring.
OPTIMIZERS = {
    "de": de.differential_evolution,
    "ga": ga.genetic_algorithm,
    "pso": pso.particle_swarm_optimization,
    "egwo": egwo.enhanced_grey_wolf_optimization,
    "gwo": gwo.grey_wolf_optimization,
    "hba": hba.honey_badger_algorithm,
    "hho": hho.harris_hawks_optimization,
    "hc": hc.hill_climbing,
    "sa": sa.simulated_annealing,
    "rs": rs.random_search,
}

INITIALIZERS = {
    "opposition_based": init.opposition_based,
    "quasi_opposition_based": init.quasi_opposition_based,
    "oblesa": init.oblesa,
}


class TestEvaluationGroups(unittest.TestCase):

    def _bounds(self):
        return np.array([[-5.0, 5.0]] * DIM)

    def test_every_optimizer_evaluates_in_n_pop_groups(self):
        for name, fn in OPTIMIZERS.items():
            with self.subTest(optimizer=name):
                rec = _Recorder()
                fn(rec, self._bounds(), n_pop=N_POP, n_iter=5, seed=0,
                   verbose=False)
                self.assertTrue(rec.batches, "objective was never called")
                self.assertEqual(
                    sorted(set(rec.batches)), [N_POP],
                    f"{name} evaluated groups of "
                    f"{sorted(set(rec.batches))}, expected only {N_POP}")

    def test_every_initializer_evaluates_in_n_pop_groups(self):
        for name, fn in INITIALIZERS.items():
            with self.subTest(initializer=name):
                rec = _Recorder()
                fn(rec, self._bounds(), n_pop=N_POP, seed=0)
                self.assertEqual(sorted(set(rec.batches)), [N_POP])

    def test_cs_is_the_documented_exception(self):
        """`cs` evaluates a partial population, on purpose.

        Asserted rather than ignored: if someone pads it later the cost
        should be a deliberate choice, and if the docstring loses the note
        the exception becomes an undocumented surprise.
        """
        rec = _Recorder()
        cs.cuckoo_search(rec, self._bounds(), n_pop=N_POP, n_iter=5, seed=0,
                         verbose=False)
        self.assertNotEqual(
            sorted(set(rec.batches)), [N_POP],
            "cs now evaluates full populations -- if that is intended, move "
            "it into OPTIMIZERS and drop this test")
        self.assertIn("Exception to the", cs.cuckoo_search.__doc__ or "",
                      "cs breaks the group-size contract without saying so")

    def test_hho_does_not_evaluate_row_by_row(self):
        """The regression that motivated this: a per-hawk inner loop.

        Row-by-row scoring is invisible to a group-size assertion if the
        rows happen to sum to `n_pop`, so the call *count* is pinned too.
        """
        rec = _Recorder()
        hho.harris_hawks_optimization(
            rec, self._bounds(), n_pop=N_POP, n_iter=5, seed=0, verbose=False)
        self.assertLessEqual(
            len(rec.batches), 4 * 5 + 2,
            f"hho made {len(rec.batches)} objective calls for 5 generations; "
            "the per-hawk loop is back")


if __name__ == "__main__":
    unittest.main()

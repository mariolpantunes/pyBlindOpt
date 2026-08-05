# coding: utf-8

"""
Differential Evolution (DE).

A powerful evolutionary algorithm that uses the differences between randomly selected vectors to perturb the population.

**Analogy:**
Imagine a group of agents. Each agent looks at other agents, takes the difference between them, scales it, and adds it to a target vector. This creates a "mutant". If the mutant is better than the current agent, it replaces it.

**Mathematical Formulation:**
**Mutation (DE/best/1):**
$$ v_i = x_{best} + F \\cdot (x_{r1} - x_{r2}) $$
**Crossover:**
Mixes the target vector $x_i$ and mutant $v_i$ with probability $CR$.

**Key Concepts:**
* **Mutation:** Generating a new vector from differences of others.
* **Crossover:** Mixing the mutant with the current individual.
* **Selection:** Greedy survival (child replaces parent if better).
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.3.0"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"

import collections.abc
import logging
import typing

import numpy as np

import pyBlindOpt.utils as utils
from pyBlindOpt.optimizer import Optimizer

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Mutation Operators
# ==============================================================================
def mutation_rand_1(
    current: np.ndarray, best: np.ndarray, candidates: np.ndarray, F: float
) -> np.ndarray:
    """
    DE/rand/1: $v = r1 + F(r2 - r3)$
    """
    return candidates[0] + F * (candidates[1] - candidates[2])


def mutation_best_1(
    current: np.ndarray, best: np.ndarray, candidates: np.ndarray, F: float
) -> np.ndarray:
    """
    DE/best/1: $v = best + F(r1 - r2)$
    """
    return best + F * (candidates[0] - candidates[1])


def mutation_rand_2(
    current: np.ndarray, best: np.ndarray, candidates: np.ndarray, F: float
) -> np.ndarray:
    """
    DE/rand/2: $v = r1 + F(r2 - r3) + F(r4 - r5)$
    """
    return (
        candidates[0]
        + F * (candidates[1] - candidates[2])
        + F * (candidates[3] - candidates[4])
    )


def mutation_best_2(
    current: np.ndarray, best: np.ndarray, candidates: np.ndarray, F: float
) -> np.ndarray:
    """
    DE/best/2: $v = best + F(r1 - r2) + F(r3 - r4)$
    """
    return (
        best + F * (candidates[0] - candidates[1]) + F * (candidates[2] - candidates[3])
    )


def mutation_current_to_best_1(
    current: np.ndarray, best: np.ndarray, candidates: np.ndarray, F: float
) -> np.ndarray:
    """
    DE/current-to-best/1: $v = current + F(best - current) + F(r1 - r2)$
    """
    return current + F * (best - current) + F * (candidates[0] - candidates[1])


def mutation_current_to_pbest_1(
    current: np.ndarray, best: np.ndarray, candidates: np.ndarray, F: float
) -> np.ndarray:
    r"""
    DE/current-to-pbest/1: $v = current + F(x_{pbest} - current) + F(r1 - r2)$

    The arithmetic is `mutation_current_to_best_1`'s, and deliberately so --
    what differs is not the formula but the vector the caller supplies as
    `best`. Here it is $x_{pbest}$, drawn afresh for each individual from the
    fittest $p \cdot N$ of the population, rather than the single global best
    shared by everyone.

    That one substitution is the whole idea behind JADE's mutation. Pulling
    every individual toward the *same* point is what makes `current-to-best/1`
    greedy and prone to premature convergence; pulling each toward a
    *different* good point keeps the population spread while still moving it
    somewhere useful. The two coincide at $p = 1/N$, and at $p = 1$ the pull
    is toward a uniformly random member, which is `current-to-rand/1`.

    See `DifferentialEvolution._base_vector`, which does the drawing.
    """
    return mutation_current_to_best_1(current, best, candidates, F)


def mutation_current_to_rand_1(
    current: np.ndarray, best: np.ndarray, candidates: np.ndarray, F: float
) -> np.ndarray:
    """
    DE/current-to-rand/1: $v = current + F(r1 - current) + F(r2 - r3)$
    """
    return current + F * (candidates[0] - current) + F * (candidates[1] - candidates[2])


# ==============================================================================
# 2. Crossover Operators
# ==============================================================================
def crossover_bin(
    target: np.ndarray, mutant: np.ndarray, cr: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Binomial Crossover.
    Each gene is swapped with probability $CR$. Ensures at least one gene is changed.
    """
    dim = target.shape[0]
    mask = rng.random(dim) < cr
    # Force at least one index to change (standard DE guarantee)
    j_rand = rng.integers(0, dim)
    mask[j_rand] = True
    return np.where(mask, mutant, target)


def crossover_exp(
    target: np.ndarray, mutant: np.ndarray, cr: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Exponential Crossover.
    Swaps a contiguous block of genes starting from a random index.
    """
    dim = target.shape[0]
    trial = target.copy()
    j = rng.integers(0, dim)
    L = 0
    while rng.random() < cr and L < dim:
        trial[j] = mutant[j]
        j = (j + 1) % dim
        L += 1
    return trial


# ==============================================================================
# 3. Parameter and Strategy Policies
# ==============================================================================
class Proposal(typing.NamedTuple):
    """What a policy decides for one trial round, one entry per individual.

    Arrays rather than scalars because the adaptive variants choose `F` and
    `cr` *per individual* and learn from which choices succeeded. Holding them
    as arrays means the generic loop indexes rather than branches, and the
    same arrays go straight back to `Policy.observe`.
    """

    F: np.ndarray
    cr: np.ndarray
    ops: list
    samples: list
    #: Per-individual greediness for the p-best strategies, or None to use the
    #: optimizer's single `p`. SHADE draws a fresh one per individual per
    #: generation, which is what makes its greediness self-scaling rather than
    #: another constant to tune.
    p: np.ndarray | None = None


class Policy:
    """How `F`, `cr` and the mutation strategy are chosen each generation.

    Separating this from `variant` is deliberate. `variant` is the textbook
    DE taxonomy -- *which* mutation and *which* crossover -- and it is a fixed
    property of a run. JADE, SHADE, SaDE and CoDE do not add mutations; they
    decide, generation by generation, which mutation and which parameters to
    use and what to learn from the outcome. Two ideas, two arguments.

    The generic loop calls three hooks and knows nothing else:

    * `begin` -- per-individual parameters for one round of trials;
    * `pool`  -- the indices difference vectors may be drawn from, which the
      archive-based policies widen;
    * `observe` -- which trials survived selection, with what parameters, so
      the policy can adapt.

    `n_trials` is how many complete populations of offspring are produced and
    evaluated per generation. It is 1 for everything except CoDE, which
    builds three and keeps the best of each triple. Crucially the loop
    evaluates them as `n_trials` separate batches of `n_pop`, never one wide
    batch, so a caller bound to exactly `n_pop` live evaluation slots keeps
    working.
    """

    n_trials = 1

    def begin(self, pop, scores, rng, trial):
        """Parameters for every individual in this round of trials."""
        raise NotImplementedError

    def pool(self, j, n_pop, rng):
        """Indices individual `j` may draw difference vectors from."""
        return np.delete(np.arange(n_pop), j)

    def augment(self, candidates, rng):
        """Optionally substitute difference vectors from outside the
        population. The default returns them untouched **and draws no
        randomness**, which is what keeps the classical path reproducible."""
        return candidates

    def observe(self, improved, proposal, delta, replaced):
        """Report the outcome.

        `improved` is the survivor mask, `delta` the score improvement
        (positive = better), and `replaced` the parents that lost -- captured
        before the population is overwritten, since that is the only moment
        they exist. Stateless policies ignore all of it."""


class FixedPolicy(Policy):
    """Constant `F` and `cr`, one strategy: classical DE.

    The default, and the compatibility anchor -- it must reproduce the
    pre-policy implementation byte for byte, including how much randomness is
    consumed, because a shift there silently moves every seeded run in every
    exercise built on this library.

    It therefore draws no random numbers of its own.
    """

    def __init__(self, F, cr, mutation_op, samples_needed):
        self.F = F
        self.cr = cr
        self.mutation_op = mutation_op
        self.samples_needed = samples_needed

    def begin(self, pop, scores, rng, trial):
        n = len(pop)
        return Proposal(
            F=np.full(n, self.F),
            cr=np.full(n, self.cr),
            ops=[self.mutation_op] * n,
            samples=[self.samples_needed] * n,
        )


class ArchivePolicy(Policy):
    r"""JADE's optional external archive, with fixed `F` and `cr`.

    When a trial beats its parent, the parent is normally overwritten and
    lost. This keeps those defeated parents in an archive $A$, and draws the
    **subtracted** difference vector from $P \cup A$ rather than from $P$:

    $$ v = x_i + F(x_{pbest} - x_i) + F(x_{r1} - \tilde{x}_{r2}),
       \qquad \tilde{x}_{r2} \sim P \cup A $$

    Only the last term changes; the base and $x_{r1}$ stay in the population.

    **Why a defeated parent is useful.** It marks somewhere the search has
    just moved *away* from, so subtracting it points the difference vector
    roughly along the direction of recent progress -- information a random
    pair does not carry. It also keeps difference vectors varied: with a
    small population there are few distinct pairs, and they shrink together
    as it converges, which is exactly when DE stalls.

    It costs nothing to evaluate. Those parents were scored in the generation
    that discarded them.

    Introduced in Zhang & Sanderson, *JADE: Adaptive Differential Evolution
    With Optional External Archive* (IEEE TEC, 2009), and carried by SHADE
    and L-SHADE since. **Optional** is the paper's own word: the reported
    benefit is larger on harder and higher-dimensional problems and can be
    neutral on easy unimodal ones, so this is a component to measure rather
    than assume.

    This is JADE's archive **without** its parameter adaptation -- `F` and
    `cr` stay fixed. Pair it with `current-to-pbest/1` for JADE's mutation;
    the adaptation is the remaining third.

    **On its own it is mostly not an improvement, and that is worth knowing
    before reaching for it.** Measured with `current-to-pbest/1/bin`, $d=10$,
    40 individuals, 300 generations, 25 seeds, paired so the archive is the
    only difference (median, and seeds won out of 25):

    | function | off | on | wins |
    | --- | --- | --- | --- |
    | sphere | 2.1e-32 | 6.8e-27 | 1/25 |
    | rastrigin | 9.94 | 12.44 | 6/25 |
    | ackley | 4.0e-15 | 6.5e-13 | 0/25 |
    | griewank | 0.0148 | 0.0156 | 12/25 |
    | **rosenbrock** | 6.93 | **2.51** | **23/25** |
    | levy | 3.8e-30 | 6.4e-26 | 1/25 |
    | zakharov | 7.0e-16 | 7.7e-13 | 3/25 |
    | dixon_price | 0.667 | 0.667 | 8/25 |
    | styblinski_tang | -391.7 | -391.7 | 5/25 |

    Decisive on rosenbrock -- a narrow curved valley, where a difference
    vector aimed along recent progress is exactly what is needed -- neutral
    on three, and worse on five. The pattern fits the mechanism: the archive
    buys diversity in the difference vectors, and on unimodal landscapes
    diversity is what you are trying to give up.

    The likely reading is that the archive is a *component* of JADE rather
    than a standalone upgrade: the paper's results have it working alongside
    adaptive `F` and `cr`, which can tighten as the archive loosens. Treat
    these numbers as the baseline to beat once the adaptation lands, not as a
    verdict on the archive.

    Args:
        F (float): Differential weight.
        cr (float): Crossover probability.
        mutation_op (Callable): Mutation from the chosen `variant`.
        samples_needed (int): How many difference vectors it takes.
        cap (int | None): Archive size limit. None caps at the population
            size, which is what JADE specifies.

    Note:
        Eviction is **random**, as in the paper. Some later implementations
        drop the oldest instead; the two behave differently once the
        population converges, so the choice is recorded rather than left to
        the reader.
    """

    def __init__(self, F, cr, mutation_op, samples_needed, cap=None):
        self.F = F
        self.cr = cr
        self.mutation_op = mutation_op
        self.samples_needed = samples_needed
        self.cap = cap
        #: Defeated parents, shape (|A|, D).
        self.archive = None
        self._n_pop = 0
        #: `observe` receives no generator; using a private one would make a
        #: seeded run irreproducible, so `begin` stashes the optimizer's.
        self._rng = None

    def begin(self, pop, scores, rng, trial):
        self._rng = rng
        n = len(pop)
        self._n_pop = n
        if self.archive is None:
            self.archive = np.empty((0, pop.shape[1]))
        return Proposal(
            F=np.full(n, self.F),
            cr=np.full(n, self.cr),
            ops=[self.mutation_op] * n,
            samples=[self.samples_needed] * n,
        )

    def augment(self, candidates, rng):
        """Redraw the subtracted vector from the union, with the union's own
        probability -- it arrives here as a population draw, so it is
        swapped for an archive member $|A| / (N + |A|)$ of the time."""
        archive = self.archive
        n_arch = 0 if archive is None else len(archive)
        if archive is None or n_arch == 0 or len(candidates) < 2:
            return candidates
        if rng.random() < n_arch / (self._n_pop + n_arch):
            candidates = candidates.copy()
            candidates[-1] = archive[rng.integers(n_arch)]
        return candidates

    def observe(self, improved, proposal, delta, replaced):
        """Absorb the parents that lost, then trim to the cap."""
        if replaced is None or len(replaced) == 0:
            return
        if self.archive is None:
            self.archive = np.empty((0, replaced.shape[1]))
        self.archive = np.vstack((self.archive, replaced))
        cap = self.cap if self.cap is not None else max(self._n_pop, 1)
        if len(self.archive) > cap and self._rng is not None:
            # Random eviction, per the paper, from the optimizer's generator
            # so the run stays reproducible.
            keep = self._rng.permutation(len(self.archive))[:cap]
            self.archive = self.archive[np.sort(keep)]

    def __repr__(self):
        n = 0 if self.archive is None else len(self.archive)
        return f"ArchivePolicy(F={self.F}, cr={self.cr}, |A|={n})"


class JadePolicy(ArchivePolicy):
    r"""JADE: the archive above, plus `F` and `cr` learned from what works.

    The third and, on the evidence in `ArchivePolicy`, the load-bearing part.
    Instead of two constants chosen once, every individual draws its own pair
    each generation from distributions whose centres follow the values that
    have been producing survivors:

    $$ F_i \sim \text{Cauchy}(\mu_F,\, 0.1), \qquad
       cr_i \sim \mathcal{N}(\mu_{cr},\, 0.1) $$

    $F_i$ is clipped to $(0, 1]$ -- non-positive draws are resampled rather
    than clamped, because a Cauchy's lower tail is heavy and clamping would
    pile probability on the boundary. $cr_i$ is clipped to $[0, 1]$.

    After selection the centres move toward the successful values:

    $$ \mu_F \leftarrow (1-c)\,\mu_F + c\,\text{mean}_L(S_F), \qquad
       \mu_{cr} \leftarrow (1-c)\,\mu_{cr} + c\,\text{mean}(S_{cr}) $$

    **The Lehmer mean on $F$ is not decoration.**
    $\text{mean}_L(S) = \sum S^2 / \sum S$ weights larger values more, so
    $\mu_F$ drifts upward whenever big steps are succeeding. An arithmetic
    mean lets $\mu_F$ decay toward the small, safe values that succeed most
    *often* -- and small steps succeed often precisely because they barely
    move, which is how a DE stops exploring while appearing to be doing well.
    $cr$ uses the arithmetic mean; there is no such asymmetry to correct.

    A Cauchy for $F$ and a normal for $cr$ is likewise deliberate: the heavy
    tail keeps occasional large steps available regardless of where $\mu_F$
    has settled, which is what lets the search escape after it has narrowed.

    Args:
        mutation_op (Callable): Mutation from the chosen `variant`.
        samples_needed (int): How many difference vectors it takes.
        mu_F (float): Initial centre for `F`. JADE's default is 0.5.
        mu_cr (float): Initial centre for `cr`. JADE's default is 0.5.
        c (float): Adaptation rate, in $(0, 1]$. JADE's default is 0.1.
        cap (int | None): Archive cap; see `ArchivePolicy`.

    Note:
        `F` and `cr` passed to the optimizer are **ignored** -- adapting them
        is the point. `variant` is not overridden, but JADE is defined on
        `current-to-pbest/1` and warns if paired with anything else.
    """

    def __init__(self, mutation_op, samples_needed, mu_F=0.5, mu_cr=0.5,
                 c=0.1, cap=None):
        if not 0.0 < c <= 1.0:
            raise ValueError(f"c must be in (0, 1], got {c}")
        super().__init__(mu_F, mu_cr, mutation_op, samples_needed, cap=cap)
        self.mu_F = float(mu_F)
        self.mu_cr = float(mu_cr)
        self.c = float(c)

    def _draw_F(self, rng, n, centre=None):
        """Cauchy, resampling the non-positive tail rather than clamping.

        Args:
            rng (np.random.Generator): Source of randomness.
            n (int): How many values to draw.
            centre (float | np.ndarray | None): Location of the Cauchy. A
                scalar is JADE's single adapted mean; an array of length `n`
                is SHADE drawing each individual from its own memory slot.
                None uses `self.mu_F`.

        Returns:
            np.ndarray: `n` values in (0, 1].
        """
        carr = np.asarray(self.mu_F if centre is None else centre, dtype=float)
        out = np.empty(n)
        todo = np.arange(n)
        for _ in range(100):
            loc = carr[todo] if carr.ndim else float(carr)
            draw = loc + 0.1 * rng.standard_cauchy(len(todo))
            np.minimum(draw, 1.0, out=draw)
            ok = draw > 0.0
            out[todo[ok]] = draw[ok]
            todo = todo[~ok]
            if not len(todo):
                break
        out[todo] = 1e-3          # pathological mu_F; keep it positive
        return out

    def begin(self, pop, scores, rng, trial):
        n = len(pop)
        self._rng = rng
        self._n_pop = n
        if self.archive is None:
            self.archive = np.empty((0, pop.shape[1]))
        return Proposal(
            F=self._draw_F(rng, n),
            cr=np.clip(self.mu_cr + 0.1 * rng.standard_normal(n), 0.0, 1.0),
            ops=[self.mutation_op] * n,
            samples=[self.samples_needed] * n,
        )

    def observe(self, improved, proposal, delta, replaced):
        super().observe(improved, proposal, delta, replaced)
        won = np.asarray(improved, dtype=bool)
        if not won.any():
            return
        s_F, s_cr = proposal.F[won], proposal.cr[won]
        denom = s_F.sum()
        if denom > 0:
            lehmer = float((s_F ** 2).sum() / denom)
            self.mu_F = (1.0 - self.c) * self.mu_F + self.c * lehmer
        self.mu_cr = (1.0 - self.c) * self.mu_cr + self.c * float(s_cr.mean())

    def __repr__(self):
        n = 0 if self.archive is None else len(self.archive)
        return (f"JadePolicy(mu_F={self.mu_F:.3f}, mu_cr={self.mu_cr:.3f}, "
                f"c={self.c}, |A|={n})")


class ShadePolicy(JadePolicy):
    r"""SHADE: JADE's adaptation, but with a memory instead of a running mean.

    JADE keeps one $\mu_F$ and one $\mu_{cr}$ and decays them toward whatever
    just worked. That is a single point of failure: one generation in which the
    only survivors used a small `F` drags the centre down, and because small
    steps keep succeeding at the higher rate, it tends not to come back. SHADE
    (Tanabe & Fukunaga, CEC 2013) replaces the pair with `h` slots:

    $$ F_i \sim \text{Cauchy}(M_F[r_i],\,0.1), \qquad
       cr_i \sim \mathcal{N}(M_{cr}[r_i],\,0.1), \qquad
       r_i \sim \mathcal{U}\{1..h\} $$

    Each individual draws from a randomly chosen slot, and **one** slot is
    rewritten per generation, cycling. The other `h-1` still hold the settings
    that worked earlier, so a bad generation costs one slot rather than the
    whole distribution -- and because individuals keep sampling the old slots,
    a setting that stops working is abandoned gradually rather than at once.

    Two further differences from `JadePolicy`, both load-bearing:

    * **The means are weighted by how much each trial improved.** With
      $w_k = \Delta f_k / \sum_j \Delta f_j$, $M_F$ takes the weighted Lehmer
      mean and $M_{cr}$ the weighted arithmetic mean. An unweighted mean counts
      a trial that barely improved the same as one that halved the objective,
      which is how the adaptation ends up chasing the *frequent* settings
      rather than the *effective* ones -- the same failure the plain Lehmer
      mean exists to correct, one level up.
    * **`p` is drawn per individual**, $p_i \sim \mathcal{U}[2/N,\,0.2]$,
      rather than fixed. Greediness then scales with population size on its own
      and stops being another constant to tune.

    Population size is unchanged: `n_pop` trials per generation, all
    independent, so the parallel evaluation contract holds exactly as for
    `FixedPolicy`.

    Args:
        mutation_op (Callable): Mutation from the chosen `variant`.
        samples_needed (int): How many difference vectors it takes.
        h (int): Memory slots. The paper's default is 6 and is insensitive
            over roughly 5-10.
        cap (int | None): Archive cap; see `ArchivePolicy`.

    Note:
        `F` and `cr` passed to the optimizer are **ignored**, as for JADE, and
        so is `p` -- SHADE draws its own. Defined on `current-to-pbest/1`.
    """

    def __init__(self, mutation_op, samples_needed, h=6, cap=None):
        if h < 1:
            raise ValueError(f"h must be >= 1, got {h}")
        super().__init__(mutation_op, samples_needed, cap=cap)
        self.h = int(h)
        self.M_F = np.full(self.h, 0.5)
        self.M_cr = np.full(self.h, 0.5)
        self.k = 0

    def begin(self, pop, scores, rng, trial):
        n = len(pop)
        self._rng = rng
        self._n_pop = n
        if self.archive is None:
            self.archive = np.empty((0, pop.shape[1]))
        slots = rng.integers(0, self.h, n)
        self._slots = slots
        # p_i in [2/N, 0.2]: the lower end is two individuals, the smallest
        # p-best pool that is still a choice rather than a single point.
        lo = min(2.0 / n, 0.2)
        return Proposal(
            F=self._draw_F(rng, n, centre=self.M_F[slots]),
            cr=np.clip(self.M_cr[slots] + 0.1 * rng.standard_normal(n),
                       0.0, 1.0),
            ops=[self.mutation_op] * n,
            samples=[self.samples_needed] * n,
            p=rng.uniform(lo, 0.2, n),
        )

    def observe(self, improved, proposal, delta, replaced):
        # ArchivePolicy, not JadePolicy: the archive update is wanted, the
        # scalar mu_F/mu_cr decay is not -- the memory replaces it.
        ArchivePolicy.observe(self, improved, proposal, delta, replaced)
        won = np.asarray(improved, dtype=bool)
        if not won.any():
            return
        w = np.asarray(delta, dtype=float)[won]
        w = np.clip(w, 0.0, None)
        total = w.sum()
        if total <= 0:
            # Every survivor tied its parent, so no trial is evidence that its
            # settings were better. Leave the memory alone rather than write a
            # uniform average of an uninformative generation.
            return
        w = w / total
        s_F, s_cr = proposal.F[won], proposal.cr[won]
        denom = float((w * s_F).sum())
        if denom > 0:
            self.M_F[self.k] = float((w * s_F ** 2).sum() / denom)
        self.M_cr[self.k] = float((w * s_cr).sum())
        self.k = (self.k + 1) % self.h

    def __repr__(self):
        n = 0 if self.archive is None else len(self.archive)
        return (f"ShadePolicy(h={self.h}, k={self.k}, "
                f"M_F~{self.M_F.mean():.3f}, M_cr~{self.M_cr.mean():.3f}, "
                f"|A|={n})")


class CodePolicy(Policy):
    r"""CoDE: build three trials per individual, keep the best of the triple.

    Composite DE (Wang, Cai & Zhang, IEEE TEC 2011) takes the opposite approach
    to JADE and SHADE. Where those learn *which* parameters work and narrow
    toward them, CoDE never adapts anything: it fixes three strategies and
    three parameter pairs chosen to fail in different ways, generates one trial
    from each strategy every generation, and lets selection decide. Nothing is
    tuned, so nothing can be mistuned.

    The three strategies, and what each is for:

    | strategy | pairs well with | covers |
    | --- | --- | --- |
    | `rand/1/bin` | any | exploration; no attraction to the best |
    | `rand/2/bin` | any | more diverse differences, slower |
    | `current-to-rand/1` | rotation | rotated landscapes -- no crossover, so no bias toward the coordinate axes |

    The three parameter pairs are $(F, cr) \in
    \{(1.0, 0.1), (1.0, 0.9), (0.8, 0.2)\}$. Each trial draws one pair
    uniformly, so a strategy is never locked to a single setting.

    **This costs three evaluations per generation, not one**, and that is the
    honest way to read any comparison against it: at a fixed evaluation budget
    CoDE runs a third of the generations. It is included because the trade is
    often worth it on multimodal landscapes, not because it is free.

    The batches stay the right shape. `n_trials = 3` means the loop calls
    `evaluate` three times with `n_pop` individuals each, never once with
    `3 * n_pop` -- so the live-agent contract, where the engine needs every
    agent connected for a batch, holds exactly as it does for `FixedPolicy`.

    Args:
        strategies (tuple | None): Mutation names, default CoDE's three.
        params (tuple | None): `(F, cr)` pairs, default CoDE's three.

    Note:
        `variant`, `F` and `cr` passed to the optimizer are **ignored**: the
        pool replaces all three. `current-to-rand/1` is used without crossover
        in the paper; here it goes through the configured crossover, which is
        the one deviation and is why `rand/1/bin` remains the first entry.
    """

    n_trials = 3

    #: (F, cr) pairs from the paper. Deliberately far apart: a high cr with a
    #: high F is an aggressive, near-total rebuild of the individual, a low cr
    #: changes almost nothing, and the third sits between them.
    _PARAMS = ((1.0, 0.1), (1.0, 0.9), (0.8, 0.2))
    _STRATS = ("rand/1", "rand/2", "current-to-rand/1")

    def __init__(self, strategies=None, params=None):
        # `is None`, not `or`: an empty tuple is falsy, so `strategies or
        # default` would silently substitute the default for a caller who
        # explicitly asked for nothing and expects to be told it is invalid.
        self.strategies = tuple(self._STRATS if strategies is None
                                else strategies)
        self.params = tuple(self._PARAMS if params is None else params)
        if not self.strategies or not self.params:
            raise ValueError("CoDE needs at least one strategy and one pair")
        unknown = set(self.strategies) - set(DifferentialEvolution._STRATEGIES)
        if unknown:
            raise ValueError(f"unknown strategies: {sorted(unknown)}")
        self.n_trials = len(self.strategies)

    def begin(self, pop, scores, rng, trial):
        """Parameters for trial round `trial`, one entry per individual.

        `trial` selects the strategy -- every individual uses the same one in a
        given round, which is what makes the round a clean batch -- while the
        parameter pair is drawn per individual.
        """
        n = len(pop)
        name = self.strategies[trial % len(self.strategies)]
        op, need = DifferentialEvolution._STRATEGIES[name]
        idx = rng.integers(0, len(self.params), n)
        pairs = np.asarray(self.params, dtype=float)
        return Proposal(
            F=pairs[idx, 0],
            cr=pairs[idx, 1],
            ops=[op] * n,
            samples=[need] * n,
        )

    def __repr__(self):
        return (f"CodePolicy(strategies={self.strategies}, "
                f"n_trials={self.n_trials})")


class EnsemblePolicy(Policy):
    r"""An ensemble of strategies and parameters, per individual (EPSDE).

    No single mutation strategy or `(F, cr)` pair is best across landscapes,
    or even across the phases of one run: greedy strategies close fast and
    stall, exploratory ones do the opposite. Rather than ask the caller to
    pick, this holds a pool of each and lets the population find out.

    Every individual carries its own `(strategy, F, cr)` triple. **A triple
    that produced a surviving trial is kept; one that failed is resampled.**
    That is the whole learning rule, and it is what separates an ensemble
    from switching at random -- combinations that work on the landscape in
    front of you accumulate, because success is the only thing that lets one
    persist.

    The pools default to the ones EPSDE was published with. Strategies span
    the range deliberately: `rand/1` explores, `best/2` exploits with two
    difference vectors, `current-to-rand/1` is rotationally invariant.

    Args:
        strategies (list | None): Strategy keys from
            `DifferentialEvolution._STRATEGIES`. None uses the default pool.
        F_pool (list | None): Values `F` may take.
        cr_pool (list | None): Values `cr` may take.

    Note:
        The crossover from `variant` still applies -- this ensembles the
        mutation and the parameters, not the recombination.
    """

    DEFAULT_STRATEGIES = ("rand/1", "best/2", "current-to-rand/1")
    DEFAULT_F = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    DEFAULT_CR = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def __init__(self, strategies=None, F_pool=None, cr_pool=None):
        self.strategy_keys = list(strategies or self.DEFAULT_STRATEGIES)
        self.F_pool = np.asarray(F_pool or self.DEFAULT_F, dtype=float)
        self.cr_pool = np.asarray(cr_pool or self.DEFAULT_CR, dtype=float)
        self._ops = [DifferentialEvolution._STRATEGIES[k]
                     for k in self.strategy_keys]
        #: Per-individual (strategy index, F, cr); grown on first use.
        self._held = None
        #: `observe` gets no generator of its own, and using a private one
        #: would make a seeded run irreproducible, so `begin` stashes the
        #: optimizer's.
        self._rng = None

    def _draw(self, rng, n):
        return np.column_stack((
            rng.integers(0, len(self._ops), size=n).astype(float),
            rng.choice(self.F_pool, size=n),
            rng.choice(self.cr_pool, size=n),
        ))

    def begin(self, pop, scores, rng, trial):
        self._rng = rng
        n = len(pop)
        held = self._held
        if held is None or len(held) != n:
            held = self._held = self._draw(rng, n)
        idx = held[:, 0].astype(int)
        return Proposal(
            F=held[:, 1].copy(),
            cr=held[:, 2].copy(),
            ops=[self._ops[i][0] for i in idx],
            samples=[self._ops[i][1] for i in idx],
        )

    def observe(self, improved, proposal, delta, replaced):
        """Keep what worked; resample what did not."""
        if self._held is None or self._rng is None:
            return
        failed = ~np.asarray(improved, dtype=bool)
        if failed.any():
            self._held[failed] = self._draw(self._rng, int(failed.sum()))

    def __repr__(self):
        return (f"EnsemblePolicy(strategies={self.strategy_keys}, "
                f"|F|={len(self.F_pool)}, |cr|={len(self.cr_pool)})")


# ==============================================================================
# 4. Optimizer Class
# ==============================================================================
class DifferentialEvolution(Optimizer):
    """
    Differential Evolution Optimizer.

    A versatile DE implementation supporting multiple mutation strategies and crossover methods
    via a configuration string.

    **Supported Variants:**
    * `rand/1`: Standard DE. Good diversity.
    * `best/1`: Converges fast, greedy.
    * `rand/2`: Robust for difficult landscapes.
    * `best/2`: Trade-off between greedy and robust.
    * `current-to-best/1`: Rotationally invariant, modern standard.
    * `current-to-pbest/1`: JADE's mutation. Like `current-to-best/1` but each
      individual is pulled toward its own draw from the fittest `p` fraction,
      so the population is not funnelled through a single point. Set `p`.
    * `current-to-rand/1`: Rotationally invariant, exploratory.

    **Supported Crossovers:**
    * `bin`: Binomial (independent swaps).
    * `exp`: Exponential (block swaps).

    **Supported Policies** (`policy=`, orthogonal to `variant`):
    * `fixed`: constant `F` and `cr`, one strategy. Classical DE.
    * `archive`: `ArchivePolicy` -- JADE's external archive of defeated
      parents, widening the pool the subtracted difference vector is drawn
      from. Fixed `F` and `cr`.
    * `jade`: `JadePolicy` -- the archive plus `F` and `cr` adapted from the
      values that produce survivors. Pair with `current-to-pbest/1/bin` for
      the published algorithm.
    * `shade`: `ShadePolicy` -- JADE with a memory of `h` settings instead of
      one running mean, updated by improvement-weighted means, and `p` drawn
      per individual. Same pairing.
    * `code`: `CodePolicy` -- three trials per individual from three fixed
      strategies, best of the triple survives. Adapts nothing, and costs
      three evaluations per generation rather than one.
    * `ensemble`: `EnsemblePolicy` -- a pool of strategies and parameters,
      one triple per individual, kept while it succeeds and resampled when
      it fails.

    `variant` says *which* mutation and crossover; `policy` says how they are
    controlled from one generation to the next. Keeping them separate is
    deliberate: the adaptive methods in the literature do not add mutations,
    they choose among the ones already here.
    """

    # Strategy Mapping: "Name" -> (Function, Required_Sample_Count)
    _STRATEGIES = {
        "rand/1": (mutation_rand_1, 3),
        "best/1": (mutation_best_1, 2),
        "rand/2": (mutation_rand_2, 5),
        "best/2": (mutation_best_2, 4),
        "current-to-best/1": (mutation_current_to_best_1, 2),
        "current-to-pbest/1": (mutation_current_to_pbest_1, 2),
        "current-to-rand/1": (mutation_current_to_rand_1, 3),
    }

    #: Strategies whose base vector is drawn per individual from the fittest
    #: `p` fraction rather than being the single global best.
    _PBEST_STRATEGIES = frozenset({"current-to-pbest/1"})

    _CROSSOVERS = {"bin": crossover_bin, "exp": crossover_exp}

    @utils.inherit_docs(Optimizer)
    def __init__(
        self,
        objective: collections.abc.Callable,
        bounds: np.ndarray,
        variant: str = "best/1/bin",
        parent_selection: str = "rand",
        F: float = 0.5,
        cr: float = 0.7,
        p: float = 0.1,
        policy: str | Policy = "fixed",
        **kwargs,
    ):
        r"""
        Differential Evolution Optimizer.

        Args:
            variant (str): Strategy format 'target/num_diffs/crossover'.
                           Examples: 'rand/1/bin', 'best/2/exp', 'current-to-best/1/bin'.
                           Defaults to 'best/1/bin'.
            parent_selection (str): Method to select the base vector ($r1$).
                                    Options: 'rand' (Standard), 'tournament'.
                                    Defaults to 'rand'.
            F (float): Differential weight (scaling factor). Defaults to 0.5.
            cr (float): Crossover probability. Defaults to 0.7.
            p (float): Fraction of the population that counts as "best" for
                `current-to-pbest/1`, in $(0, 1]$. Ignored by every other
                variant. Defaults to 0.1, i.e. the fittest tenth.

                It interpolates between two variants already here: the pool is
                floored at one individual, so $p \le 1/N$ is exactly
                `current-to-best/1`, and $p = 1$ draws from the whole
                population, which is `current-to-rand/1`'s pull. Values around
                0.05-0.2 are the usual range.
            policy (str | Policy): How `F`, `cr` and the mutation strategy are
                chosen each generation. `variant` says *which* mutation and
                crossover; this says how they are controlled, and the two are
                deliberately separate arguments rather than one string.

                * ``'fixed'`` (default) -- constant `F` and `cr`, one
                  strategy. Classical DE, and the behaviour this class has
                  always had.
                * ``'ensemble'`` -- `EnsemblePolicy`: a pool of strategies and
                  parameters, one triple per individual, kept while it
                  succeeds and resampled when it fails. Overrides the
                  strategy from `variant`, `F` and `cr`; the crossover still
                  applies.
                * a `Policy` instance, for a custom rule.
        """
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p must be in (0, 1], got {p}")

        self.F = F
        self.cr = cr
        self.p = p
        self.parent_selection = parent_selection
        self.variant_name = variant

        # Parse Variant String
        try:
            # Expect format: "strategy_base/strategy_num/crossover"
            # We join base and num to look up strategy (e.g. "rand/1")
            parts = variant.split("/")
            if len(parts) != 3:
                raise ValueError("Variant format must be 'base/n/cross'")

            strategy_key = f"{parts[0]}/{parts[1]}"
            crossover_key = parts[2]

            if strategy_key not in self._STRATEGIES:
                raise KeyError(f"Unknown strategy: {strategy_key}")
            if crossover_key not in self._CROSSOVERS:
                raise KeyError(f"Unknown crossover: {crossover_key}")

            self.mutation_op, self.samples_needed = self._STRATEGIES[strategy_key]
            self.crossover_op = self._CROSSOVERS[crossover_key]
            self.uses_pbest = strategy_key in self._PBEST_STRATEGIES

        except (KeyError, IndexError, ValueError) as e:
            valid_strats = list(self._STRATEGIES.keys())
            valid_cross = list(self._CROSSOVERS.keys())
            raise ValueError(
                f"Invalid variant '{variant}'.\n"
                f"Supported Strategies: {valid_strats}\n"
                f"Supported Crossovers: {valid_cross}\n"
                f"Error: {e}"
            )

        if isinstance(policy, Policy):
            self.policy = policy
        elif policy == "fixed":
            self.policy = FixedPolicy(F, cr, self.mutation_op,
                                      self.samples_needed)
        elif policy == "archive":
            self.policy = ArchivePolicy(F, cr, self.mutation_op,
                                        self.samples_needed)
        elif policy == "jade":
            self.policy = JadePolicy(self.mutation_op, self.samples_needed)
            if strategy_key != "current-to-pbest/1":
                logger.warning(
                    "policy='jade' with variant '%s'. JADE is defined on "
                    "current-to-pbest/1; the adaptation still runs, but this "
                    "is not the published algorithm.", variant,
                )
        elif policy == "shade":
            self.policy = ShadePolicy(self.mutation_op, self.samples_needed)
            if strategy_key != "current-to-pbest/1":
                logger.warning(
                    "policy='shade' with variant '%s'. SHADE is defined on "
                    "current-to-pbest/1; the adaptation still runs, but this "
                    "is not the published algorithm.", variant,
                )
        elif policy == "code":
            self.policy = CodePolicy()
        elif policy == "ensemble":
            self.policy = EnsemblePolicy()
        else:
            raise ValueError(
                f"Unknown policy '{policy}'. Supported: 'fixed', 'archive', "
                f"'jade', 'shade', 'code', 'ensemble', "
                f"or a Policy instance."
            )
        self.policy_name = policy if isinstance(policy, str) else type(policy).__name__
        #: Filled by `_generate_offspring`, consumed by `_selection`.
        self._proposal = None

        super().__init__(objective, bounds, **kwargs)

    def _initialize(self):
        """
        Initialization hook.
        """
        pass

    def _update_best(self, epoch: int):
        """
        Updates the global best solution.
        """
        best_idx = np.argmin(self.scores)
        if self.scores[best_idx] < self.best_score:
            self.best_score = self.scores[best_idx]
            self.best_pos = self.pop[best_idx].copy()

    def _generate_offspring(self, epoch: int, round_idx: int = 0) -> np.ndarray:
        """
        Generates the Trial Population.

        1.  **Selection:** Picks random distinct vectors ($r1, r2, ...$).
            Supports Tournament selection for the base vector ($r1$).
        2.  **Mutation:** Creates mutant vectors.
        3.  **Crossover:** Combines mutant and target vectors.
        """
        offspring = np.zeros_like(self.pop)
        n_pop = self.n_pop

        # The policy decides F, cr and the strategy for every individual
        # before any of them is built, so an adaptive policy can look at the
        # whole population once rather than per individual. `FixedPolicy`
        # returns constants and draws no randomness, which is what keeps this
        # path byte-identical to the pre-policy implementation.
        proposal = self.policy.begin(self.pop, self.scores, self.rng,
                                     round_idx)
        self._proposal = proposal

        for j in range(n_pop):
            # 1. Identify valid pool (cannot include self)
            available_indices = self.policy.pool(j, n_pop, self.rng)
            samples_needed = proposal.samples[j]

            # 2. Select Candidates
            if self.parent_selection == "tournament":
                # Tournament for the first candidate (r1 / base vector)
                # This increases selection pressure for the mutation base.

                # a. Perform Tournament
                k_tournament = 3
                if len(available_indices) < k_tournament:
                    # Fallback for small pops
                    tourn_inds = self.rng.choice(
                        available_indices, size=len(available_indices), replace=False
                    )
                else:
                    tourn_inds = self.rng.choice(
                        available_indices, size=k_tournament, replace=False
                    )

                # Winner has lowest score (minimization)
                winner_idx = tourn_inds[np.argmin(self.scores[tourn_inds])]

                # b. Select remaining candidates randomly
                needed_others = samples_needed - 1
                remaining_pool = np.setdiff1d(available_indices, [winner_idx])

                if len(remaining_pool) < needed_others:
                    # Fallback (allow replacement if strictly needed)
                    others = self.rng.choice(
                        remaining_pool, size=needed_others, replace=True
                    )
                else:
                    others = self.rng.choice(
                        remaining_pool, size=needed_others, replace=False
                    )

                # Combine: [Winner, r2, r3...]
                choices = np.concatenate(([winner_idx], others))

            else:
                # Standard Random Selection
                if samples_needed > len(available_indices):
                    choices = self.rng.choice(
                        available_indices, size=samples_needed, replace=True
                    )
                else:
                    choices = self.rng.choice(
                        available_indices, size=samples_needed, replace=False
                    )

            candidates = self.policy.augment(self.pop[choices], self.rng)

            # 3. Mutation
            # Note: For 'best/...' strategies, 'best' is used as base, and 'candidates' are just diffs.
            # Tournament selection above mainly benefits 'rand/...' strategies where candidates[0] is base.
            mutant = proposal.ops[j](
                self.pop[j], self._base_vector(j), candidates, proposal.F[j]
            )

            # 4. Crossover
            trial = self.crossover_op(
                self.pop[j], mutant, proposal.cr[j], self.rng
            )
            offspring[j] = trial

        return offspring

    def _base_vector(self, j: int) -> np.ndarray:
        r"""The vector the mutation pulls toward, for individual `j`.

        Ordinary strategies pull everyone toward the same global best. The
        p-best ones draw a fresh $x_{pbest}$ per individual from the fittest
        $\lfloor p N \rfloor$, which is what stops the population being
        funnelled through a single point.

        Two details that are easy to get wrong and change the algorithm:

        * the pool is floored at **one** individual, so small populations
          degrade to `current-to-best/1` instead of raising on an empty
          selection;
        * the draw is per individual **per generation**. Sharing one draw
          across the population reintroduces exactly the single attractor
          the variant exists to avoid.

        Args:
            j (int): Index of the individual being mutated.

        Returns:
            np.ndarray: The base vector, shape (D,).
        """
        if not self.uses_pbest:
            return self.best_pos

        # A policy may set its own greediness per individual (SHADE does);
        # otherwise the optimizer's single `p` applies to everyone.
        prop = getattr(self, "_proposal", None)
        p_j = self.p if prop is None or prop.p is None else float(prop.p[j])
        n_best = max(1, int(p_j * self.n_pop))
        top = np.argpartition(self.scores, n_best - 1)[:n_best]
        return self.pop[self.rng.choice(top)]

    def _evolve_once(self, epoch: int):
        """One generation, in `policy.n_trials` batches of `n_pop`.

        With the default `n_trials = 1` this is the base implementation and is
        byte-identical to it. CoDE sets 3: three trial vectors are built per
        individual, each from a different strategy, and only the best of each
        triple goes to selection.

        Each round is its own `evaluate` call of exactly `n_pop` individuals.
        Stacking them into one call of `3 * n_pop` would be fewer round trips
        and is deliberately not done -- the live-agent backend needs every
        agent in a batch connected together, so batch shape is part of the
        contract rather than a tuning choice.

        The winning trial's `Proposal` is the one handed to `observe`, so an
        adaptive policy still learns from the parameters that actually
        produced each survivor rather than from whichever round happened to
        run last.

        Args:
            epoch (int): The current generation index.
        """
        n_trials = getattr(self.policy, "n_trials", 1)
        if n_trials <= 1:
            return super()._evolve_once(epoch)

        best_off = None
        best_sc = None
        best_prop = None
        for t in range(n_trials):
            off = self._check_bounds(self._generate_offspring(epoch, t))
            sc = self.evaluate(off)
            prop = self._proposal
            if prop is None:          # no policy proposal: nothing to track
                continue
            if best_off is None or best_sc is None or best_prop is None:
                best_off, best_sc, best_prop = off, sc, prop
                continue
            win = sc < best_sc
            best_off[win] = off[win]
            best_sc[win] = sc[win]
            # Keep the per-individual parameters aligned with the trial that
            # won, so `observe` is told what produced the survivor.
            best_prop.F[win] = prop.F[win]
            best_prop.cr[win] = prop.cr[win]

        if best_off is None or best_sc is None:
            return
        self._proposal = best_prop
        self._selection(best_off, best_sc)

    def _selection(self, offspring: np.ndarray, offspring_scores: np.ndarray):
        """
        Greedy Survivor Selection.

        The child replaces the parent if and only if it is better or equal.

        The survivor mask and the size of each improvement are reported to
        the policy before they are discarded. Every adaptive variant learns
        from exactly this: JADE and SHADE from the `F` and `cr` that produced
        surviving trials, SaDE from which strategy did. Computing it and
        throwing it away, as this did, is why none of them could be built.
        """
        improved_mask = offspring_scores <= self.scores
        delta = self.scores - offspring_scores

        # The parents about to be overwritten, captured while they still
        # exist: an archive-based policy stores exactly these, and after the
        # assignment below they are gone.
        replaced = self.pop[improved_mask].copy()

        self.pop[improved_mask] = offspring[improved_mask]
        self.scores[improved_mask] = offspring_scores[improved_mask]

        if self._proposal is not None:
            self.policy.observe(improved_mask, self._proposal, delta, replaced)


def differential_evolution(
    objective: collections.abc.Callable,
    bounds: np.ndarray,
    variant: str = "best/1/bin",
    parent_selection: str = "rand",
    F: float = 0.5,
    cr: float = 0.7,
    p: float = 0.1,
    policy: str = "fixed",
    **kwargs,
) -> tuple:
    """
    Functional interface for Differential Evolution.

    Args:
        objective (Callable): The function to minimize.
        bounds (np.ndarray): Search bounds (min, max).
        variant (str): Strategy string (e.g., 'rand/1/bin').
        parent_selection (str): 'rand' or 'tournament'.
        F (float): Mutation factor.
        cr (float): Crossover probability.

    Returns:
        tuple: (best_pos, best_score).
    """
    optimizer = DifferentialEvolution(
        objective=objective,
        bounds=bounds,
        variant=variant,
        parent_selection=parent_selection,
        p=p,
        policy=policy,
        F=F,
        cr=cr,
        **kwargs,
    )
    return optimizer.optimize()

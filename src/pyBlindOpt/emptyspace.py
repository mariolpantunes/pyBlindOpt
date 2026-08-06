
"""
Reference empty-space engines.

`ess.esa` finds under-explored regions with a vectorized multi-particle
repulsion, which is fast but makes it hard to separate two questions: does
placing points in empty space help a population-based optimizer, and does
*this particular relaxation* implement that idea well? Answering the first
needs an engine whose definition of "empty" is not open to argument.

The engines here take the objective **put the next point at the centre of the
largest empty sphere, then repeat**, and the same call signature as `ess.esa`,
so either can be substituted directly:

    init.oblesa(objective, bounds, engine=emptyspace.dart_esa)

:func:`dart_esa` reaches that centre by search over a candidate cloud rather
than by solving for it. The point furthest from everything already placed *is*
the centre of the largest empty sphere -- a Voronoi vertex, equivalently the
circumcentre of a Delaunay simplex -- so this computes the exact same quantity
a triangulation would, converging to it as the candidate count grows. Doing it
by search is what keeps the module pure NumPy and free of a dimension ceiling;
an exact triangulation needs SciPy and collapses under its own simplex count
above a handful of dimensions.

That exact version lives outside the library, in
`examples/emptyspace_reference.py`, precisely because it needs SciPy. It is
used to validate `dart_esa` at low dimension and nothing else --
`test_emptyspace.py` checks the two agree to within about 1%.

`dart_esa` is deliberately slower than `ess.esa`. That is the trade: it exists
to attribute an effect, not to ship in the hot path.

These are controls, not the production path. OBLESA's guided placement is
`ess.esa`; :func:`dart_esa` is novelty with no idea where the good regions are,
and :func:`random_esa` is the null with no search at all.
"""

__author__ = "Mário Antunes"
__license__ = "MIT"
__version__ = "0.1.0"
__email__ = "mario.antunes@ua.com"
__url__ = "https://github.com/mariolpantunes/pyblindopt"
__status__ = "Development"

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _nearest_sq_torus(
    points: np.ndarray, static: np.ndarray, span: np.ndarray, block: int = 256
) -> np.ndarray:
    """
    Nearest-neighbour squared distance with **wrap-around** in every axis.

    Per coordinate the separation is `min(|a - b|, span - |a - b|)`, so
    opposite faces of the box are identified and the domain has no boundary.

    This is what makes the search faithful to ESS, which relaxes on a torus.
    In a bounded box the point furthest from everything is almost always *on a
    face* -- a corner has no neighbours behind it, so the wall acts like a
    region of infinite emptiness. Measured with the box metric, probes carried
    a mean coordinate magnitude 15-40% larger than uniform sampling and landed
    against a wall 24-30% of the time at low dimension, against 3-10% for
    uniform. On a landscape whose optimum is interior that is a systematic
    push in the wrong direction, and it is an artefact of the metric rather
    than anything to do with empty space.

    No BLAS expansion is available here -- the wrap is elementwise, so the
    `(K, M, D)` difference tensor is unavoidable. It is taken in blocks of
    `block` candidates to keep the working set bounded.
    """
    out = np.empty(points.shape[0])
    for i in range(0, points.shape[0], block):
        chunk = points[i : i + block]
        diff = np.abs(chunk[:, None, :] - static[None, :, :])
        np.minimum(diff, span - diff, out=diff)
        out[i : i + block] = np.einsum("ijk,ijk->ij", diff, diff).min(axis=1)
    return out


def _nearest_sq(points: np.ndarray, static: np.ndarray) -> np.ndarray:
    """
    Squared distance from every point to its nearest neighbour in `static`.

    Via the expansion $||a - b||^2 = ||a||^2 + ||b||^2 - 2 \\langle a, b\\rangle$
    -- the same trick :func:`pyBlindOpt.utils.global_distances` uses -- so the
    work lands in one BLAS matrix product of shape `(K, M)` instead of
    materialising the `(K, M, D)` difference tensor. That tensor is what makes
    the naive form unusable here: :func:`dart_esa` calls this once per placed
    point with `K` in the thousands, and at `K=2048, M=120, D=200` the
    broadcast allocates ~400 MB per call. Measured 5-10x faster at the sizes
    this module runs at, and flat in memory.
    """
    sq = (
        np.sum(points**2, axis=1)[:, None]
        + np.sum(static**2, axis=1)[None, :]
        - 2.0 * (points @ static.T)
    )
    # Cancellation can push near-zero entries slightly negative.
    return np.maximum(sq.min(axis=1), 0.0)


def dart_esa(
    samples: np.ndarray,
    bounds: np.ndarray,
    *,
    n: int,
    seed: int | np.random.Generator | None = None,
    k_cand: int = 2048,
    _torus: bool = True,
    **ignored,
) -> np.ndarray:
    """
    Largest-empty-sphere filling by dart throwing (Mitchell's best-candidate).

    Places `n` points one at a time. For each, draws `k_cand` uniform
    candidates and keeps whichever is furthest from the nearest point already
    present -- existing samples and previously placed points alike -- then
    freezes it and moves on.

    **Why this is the Voronoi answer.** On the torus, the point furthest
    from all others is
    by definition the centre of the largest empty sphere, which is a vertex of
    the Voronoi diagram, equivalently the circumcentre of a Delaunay simplex.
    So this optimizes exactly the quantity an exact triangulation would solve
    for; it just evaluates the objective on a finite candidate cloud instead.
    Accuracy is controlled by `k_cand` alone and improves monotonically with
    it, which makes the approximation auditable in a way a relaxation's step
    size and epoch count are not. `examples/emptyspace_reference.py` holds the
    exact SciPy version this is checked against.

    Like the relaxation it stands in for the metric is toroidal, and unlike it
    there is no force kernel, no step size and no convergence criterion, so
    nothing here can be tuned into or out of a result.

    Args:
        samples (np.ndarray): Points already occupying the space, shape (M, D).
        bounds (np.ndarray): Search space bounds, shape (D, 2).
        n (int): How many points to place.
        seed (int | Generator | None): Random seed or Generator.
        k_cand (int): Candidates drawn per placed point. Higher is closer to
            the exact largest empty sphere and linearly more expensive.
        _torus (bool): **Not part of the contract.** The torus is the
            behaviour, matching ESS; `False` selects the plain box metric and
            exists only so the tests that measure what the wrap buys still
            have their control, and so the box reference in
            `examples/emptyspace_reference.py` -- a box construction -- can be
            compared on its own terms. See :func:`_nearest_sq_torus` for why
            the box metric biases every probe outward.
        **ignored: Accepted and dropped, so the signature stays compatible
            with `ess.esa`'s relaxation parameters.

    Returns:
        np.ndarray: The `n` placed points, shape (n, D).
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    lower, upper = bounds[:, 0], bounds[:, 1]
    dim = bounds.shape[0]
    n = int(n)
    if n <= 0:
        return np.empty((0, dim))

    span = upper - lower
    static = np.asarray(samples, dtype=float).reshape(-1, dim)
    placed = np.empty((n, dim))

    def nearest(cand, ref):
        return (_nearest_sq_torus(cand, ref, span) if _torus
                else _nearest_sq(cand, ref))

    for i in range(n):
        cand = rng.uniform(lower, upper, size=(k_cand, dim))
        if static.shape[0] == 0:
            placed[i] = cand[0]
        else:
            placed[i] = cand[int(np.argmax(nearest(cand, static)))]
        static = np.vstack((static, placed[i][None, :]))

    return placed


def random_esa(
    samples: np.ndarray,
    bounds: np.ndarray,
    *,
    n: int,
    seed: int | np.random.Generator | None = None,
    **ignored,
) -> np.ndarray:
    """
    Uniform random points, ignoring where the existing ones are.

    The null engine. It has OBLESA's pool shape and candidate count but does
    no empty-space search at all, so it separates "the empty-space search
    found something" from "a larger candidate pool gave the selector more to
    choose from" -- two explanations that the OBLESA-versus-OBL comparison on
    its own cannot tell apart.

    Args:
        samples (np.ndarray): Ignored; present for signature compatibility.
        bounds (np.ndarray): Search space bounds, shape (D, 2).
        n (int): How many points to draw.
        seed (int | Generator | None): Random seed or Generator.
        **ignored: Accepted and dropped.

    Returns:
        np.ndarray: Uniform points, shape (n, D).
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(int(n), bounds.shape[0]))


# Which optional keywords each engine will accept from `init.oblesa`. An
# engine without this attribute -- `ess.esa` -- receives only `samples`,
# `bounds`, `n` and `seed`, because it forwards anything it does not recognise
# into its metric kernel and dies on it. Declaring capabilities rather than
# probing for them is what keeps the backends substitutable.
dart_esa.accepts = frozenset({"k_cand"})  # type: ignore[reportFunctionMemberAccess]
random_esa.accepts = frozenset()  # type: ignore[reportFunctionMemberAccess]

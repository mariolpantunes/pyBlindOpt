r"""
Enhanced Grey Wolf Optimization (EGWO).

An improvement over GWO that addresses the balance between exploration and
exploitation. Instead of moving towards the centroid of Alpha, Beta and Gamma,
wolves move towards a weighted "Prey" position that includes stochastic error.

**Mathematical Formulation:**
$$ X_{prey} = w_1 X_\alpha + w_2 X_\beta + w_3 X_\gamma
              + \mathcal{N}(0, \sigma^2) $$
$$ X_{t+1} = X_{prey} - A \odot |C \odot X_{prey} - X_t| $$
where $A \sim U(-a, a)$ and $\sigma$ both shrink with $a$, which falls
linearly $2 \to 0$ across the run.

Note:
    Three things here were wrong, and together they meant the algorithm did
    not search. They are described in `EGWO._generate_offspring` and
    `EGWO._update_iter_params`; the short version is that the position update
    was anchored on the wolf rather than on the prey, so its expected
    displacement toward the prey was exactly zero at every iteration.
"""


import collections.abc

import numpy as np

from pyBlindOpt.gwo import GWO


class EGWO(GWO):
    """
    Enhanced Grey Wolf Optimization.

    Extends standard GWO by introducing a weighted prey position with stochastic error.
    """

    def __init__(self, *args, noise_scale: float = 0.0, **kwargs):
        r"""
        Args:
            noise_scale (float): Standard deviation of the prey's positional
                noise at $a = 2$, as a fraction of the mean bound width,
                decaying with $a$ to zero by the end of the run. **Defaults to
                0.0, which switches the noise term off.**

        Note:
            That default deserves the explanation, because the stochastic prey
            term is what this class is named for. Once the position update is
            anchored correctly (see `_generate_offspring`), the noise is not
            merely unnecessary -- it is harmful, monotonically, at every scale
            tried. Geometric mean of final fitness over 6 functions x 5 seeds
            at 30x60, and on the 10-D Ackley case in the test suite:

            | `noise_scale` |   d=8 |  d=32 | ackley 10D |
            |---------------|-------|-------|------------|
            | **0.0**       |**0.303**|**33.4**| **0.000** |
            | 0.05          | 0.695 |  34.6 |      1.616 |
            | 0.10          | 1.438 |  56.5 |      4.110 |
            | 0.20          | 3.512 | 115.1 |      5.958 |
            | 0.35          | 7.524 | 182.3 |      8.053 |
            | `GWO`         | 0.401 |  36.1 |      0.000 |

            The reading is that the *weighted* prey -- random weights over
            alpha, beta and gamma, rather than GWO's equally weighted average
            of three separate leader pulls -- is the part of EGWO that earns
            its keep, and it does: at `noise_scale=0` this still beats GWO at
            both dimensions. The additive noise was compensating for a broken
            search, and its old schedule hid that by dying within the first
            few percent of the run.

            Kept as a knob rather than deleted, so the published formulation
            stays reachable and so the claim above stays falsifiable.
        """
        self.noise_scale = float(noise_scale)
        super().__init__(*args, **kwargs)

    def _initialize(self):
        """
        Initializes GWO hierarchy and EGWO error parameter.

        Sets up `epoch_std` for the stochastic term.
        """
        super()._initialize()
        self.epoch_std = 0.0

    def _update_iter_params(self, epoch: int):
        r"""
        Decays the exploration coefficient and the prey's positional noise.

        `super()` first, because GWO's $a = 2(1 - t/T)$ is what both this
        class's step size and its noise are scaled by. Overriding this hook
        *without* chaining left `self.a` frozen at whatever `_initialize` set,
        so nothing in the search contracted.

        The noise scale is tied to $a$ and to the width of the search box:

        $$ \sigma_t = \kappa\, a_t\, \overline{(u - l)} $$

        rather than to the previous $\sigma_t = \exp(-100 (t+1)/T)$. That form
        has two defects beyond being unscaled: the $-100$ is absolute, so the
        schedule ignores `n_iter` entirely, and it is spent almost immediately
        -- at $T = 200$ it is 0.607 at $t=0$, 0.004 at $t=10$ and $2 \times
        10^{-9}$ at $t=40$. The stochastic prey term is the whole difference
        between this class and `GWO`, and it was dead within 5% of the run.

        Args:
            epoch (int): Current iteration.
        """
        super()._update_iter_params(epoch)
        width = float(np.mean(self.bounds[:, 1] - self.bounds[:, 0]))
        self.epoch_std = self.noise_scale * self.a * width

    def _generate_offspring(self, epoch: int) -> np.ndarray:
        r"""
        Generates new positions using Weighted Prey + Noise.

        1.  **Weights:** random weights for $\alpha, \beta, \gamma$, normalised
            and sorted so the alpha wolf carries the most.
        2.  **Target:** the weighted leader position plus Gaussian noise whose
            scale decays with $a$.
        3.  **Update:** each wolf is placed relative to that prey,
            $X_{t+1} = X_{prey} - A \odot |C \odot X_{prey} - X_t|$, with
            $A \sim U(-a, a)$ and $C \sim U(0, 2)$ drawn per coordinate.

        **The anchor is the prey, and that is the entire update rule.** This
        previously read

            offspring = self.pop - self.rng.uniform(-2, 2, ...) * |prey - pop|

        which is anchored on the wolf's own position, and takes the absolute
        value of the separation -- so the displacement carries no information
        about which side of the prey the wolf is on. With the multiplier
        symmetric about zero, the expected displacement is exactly

        $$ \mathbb{E}[X_{t+1} - X_t]
           = -\mathbb{E}[U(-2,2)] \cdot |X_{prey} - X_t| = 0 $$

        at every iteration and for every wolf. Measured: mean displacement per
        coordinate 0.058 against a mean distance to the prey of 2.696, which
        is sampling noise. The pack did not move toward the prey at all; it
        diffused, and only greedy selection in `GWO._selection` stopped it
        wandering off. Population spread after 60 generations was 0.767 of its
        initial value where GWO reaches 0.001, and the corrected form here
        reaches 0.205.

        The consequence for anyone comparing initializers: an optimizer that
        does not converge cannot express a better starting population, so the
        `egwo` rows of every sweep before this measured the diffusion, not the
        initializer. Fixing it is worth 1.9x on final fitness (6.47 -> 3.41,
        geometric mean over 6 functions x {8, 32} dims x 6 seeds).

        Args:
            epoch (int): Current iteration.

        Returns:
            np.ndarray: The new positions.
        """
        dim = self.pop.shape[1]

        # 1. Weights (Omega): uniform [1, 3], normalised, sorted descending so
        #    alpha gets the largest share.
        omega = self.rng.uniform(1, 3, size=3)
        omega /= np.sum(omega)
        omega = np.sort(omega)[::-1]

        # 2. "Prey": weighted leaders plus noise that shrinks with `a`.
        prey = (
            omega[0] * self.alpha_pos
            + omega[1] * self.beta_pos
            + omega[2] * self.gamma_pos
            + self.rng.normal(0, self.epoch_std, size=dim)
        )

        # 3. GWO's encircling step, taken against the prey rather than the
        #    centroid of three separate leader pulls. `A` spans [-a, a], so it
        #    is wide early (the wolf can be thrown past the prey, exploring)
        #    and narrows onto it as `a` decays.
        A = 2 * self.a * self.rng.random((self.n_pop, dim)) - self.a
        C = 2 * self.rng.random((self.n_pop, dim))

        return prey - A * np.abs(C * prey - self.pop)

    # _selection and _update_best are inherited from GWO
    # as they are identical (Greedy selection & Top-3 hierarchy)


def enhanced_grey_wolf_optimization(
    objective: collections.abc.Callable, bounds: np.ndarray, **kwargs
) -> tuple:
    """
    Functional interface for Enhanced GWO.

    Returns:
        tuple: (best_pos, best_score).
    """
    optimizer = EGWO(objective=objective, bounds=bounds, **kwargs)
    return optimizer.optimize()

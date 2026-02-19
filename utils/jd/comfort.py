"""
COMFORT: COmbined MGDA-UPGrad for Long-Tail (LT) Optimization

Combines the conservative MGDA aggregated gradient with the aggressive UPGrad aggregated
gradient to better suit the long-tail learning phase, where small classes can have
near-zero gradients and large classes tend to have larger gradients.

    g_COMFORT = (1 - beta) * g_MGDA + beta * g_UPGrad,    0 <= beta <= 1.

Beta is scheduled to increase from near 0 (early training) to near 1 (late training)
using a beta-VAE style schedule with parameters k, a, l, u.
"""

from typing import Literal

import torch
from torch import Tensor

from torchjd.aggregation import UPGrad

from .mgda import MGDA, StableMGDA

_NormType = Literal["none", "l2", "loss", "loss+"]


def beta_schedule(
    epoch: int,
    total_epochs: int,
    k: float = 1.0,
    a: float = 1.0,
    l: float = 0.01,
    u: float = 1.0,
) -> float:
    r"""
    Beta-VAE style schedule: beta increases from ``l`` to ``u`` over training.

    Progress is computed as ``progress = ((epoch - 1) / (total_epochs - 1)) ** a``
    (in [0, 1]). Then:

    .. math::
        f = \frac{1 - \exp(-k \cdot \text{progress})}{1 - \exp(-k)}, \quad
        \beta = l + (u - l) \cdot f.

    So at the first epoch beta ≈ l, and at the last epoch beta = u.

    :param epoch: Current 1-based epoch index.
    :param total_epochs: Total number of epochs.
    :param k: Steepness of the exponential warmup (default 1).
    :param a: Power for progress (default 1 = linear progress).
    :param l: Lower bound for beta (e.g. 0.01 for first epoch).
    :param u: Upper bound for beta (e.g. 1 for last epoch).
    :return: Beta in [l, u].
    """
    if total_epochs <= 1:
        return u
    progress = (epoch - 1) / (total_epochs - 1)
    progress = min(1.0, max(0.0, progress)) ** a
    if k <= 0:
        f = progress
    else:
        import math
        denom = 1.0 - math.exp(-k)
        f = (1.0 - math.exp(-k * progress)) / denom
    beta = l + (u - l) * f
    return float(min(u, max(l, beta)))


class COMFORT:
    r"""
    COmbined MGDA-UPGrad aggregator for long-tail friendly gradient descent.

    Computes:

        g_COMFORT = (1 - beta) * g_MGDA + beta * g_UPGrad,

    where beta is updated each epoch via :func:`beta_schedule` (e.g. from 0.01 to 1).
    Call :meth:`set_epoch` at the start of each epoch so beta is correct.

    :param mgda_norm_type: Normalization for the MGDA branch (same as :class:`MGDA`).
    :param mgda_stable: If True, use :class:`StableMGDA` (eigen regularization).
    :param mgda_epsilon: MGDA convergence threshold.
    :param mgda_max_iters: MGDA max iterations.
    :param mgda_min_eigenvalue_eps: Used when ``mgda_stable=True``.
    :param beta_k: Schedule steepness (default 1).
    :param beta_a: Schedule progress power (default 1).
    :param beta_l: Beta at start of training (default 0.01).
    :param beta_u: Beta at end of training (default 1).
    """

    def __init__(
        self,
        mgda_norm_type: _NormType = "none",
        mgda_stable: bool = False,
        mgda_epsilon: float = 1e-5,
        mgda_max_iters: int = 250,
        mgda_min_eigenvalue_eps: float = 1.0,
        beta_k: float = 1.0,
        beta_a: float = 1.0,
        beta_l: float = 0.01,
        beta_u: float = 1.0,
    ):
        if mgda_stable:
            self._mgda = StableMGDA(
                norm_type=mgda_norm_type,
                epsilon=mgda_epsilon,
                max_iters=mgda_max_iters,
                min_eigenvalue_eps=mgda_min_eigenvalue_eps,
            )
        else:
            self._mgda = MGDA(
                norm_type=mgda_norm_type,
                epsilon=mgda_epsilon,
                max_iters=mgda_max_iters,
                stable=False,
                min_eigenvalue_eps=mgda_min_eigenvalue_eps,
            )
        self._upgrad = UPGrad()
        self._beta_k = beta_k
        self._beta_a = beta_a
        self._beta_l = beta_l
        self._beta_u = beta_u
        self._current_epoch = 1
        self._total_epochs = 1
        self._norm_type = mgda_norm_type  # for set_losses check in training loop
        # Expose MGDA weighting for hooks (e.g. gradient_similarity)
        self.weighting = getattr(self._mgda, "weighting", self._mgda.mgda_weighting)

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        """Set current epoch and total epochs for beta scheduling. Call at the start of each epoch."""
        self._current_epoch = epoch
        self._total_epochs = total_epochs

    def set_losses(self, losses: Tensor) -> None:
        """Forward to MGDA when it uses loss-based normalization."""
        if hasattr(self._mgda, "set_losses"):
            self._mgda.set_losses(losses)

    def _get_beta(self) -> float:
        return beta_schedule(
            self._current_epoch,
            self._total_epochs,
            k=self._beta_k,
            a=self._beta_a,
            l=self._beta_l,
            u=self._beta_u,
        )

    def __call__(self, J: Tensor) -> Tensor:
        """
        Aggregate the Jacobian (per-task gradients) into a single gradient.

        :param J: Jacobian of shape (num_tasks, num_params).
        :return: Aggregated gradient of shape (num_params,).
        """
        g_mgda = self._mgda(J)
        g_upgrad = self._upgrad(J)
        beta = self._get_beta()
        return (1.0 - beta) * g_mgda + beta * g_upgrad

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mgda_norm_type={self._norm_type!r}, "
            f"mgda_stable={getattr(self._mgda, '_stable', False)}, "
            f"beta_k={self._beta_k}, beta_a={self._beta_a}, beta_l={self._beta_l}, beta_u={self._beta_u})"
        )

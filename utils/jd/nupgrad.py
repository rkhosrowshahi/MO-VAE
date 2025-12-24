from typing import Literal

import torch
from torch import Tensor

from torchjd.aggregation._aggregator_bases import GramianWeightedAggregator
from torchjd.aggregation._mean import MeanWeighting
from torchjd.aggregation._utils.dual_cone import project_weights
from torchjd.aggregation._utils.gramian import normalize, regularize
from torchjd.aggregation._utils.non_differentiable import raise_non_differentiable_error
from torchjd.aggregation._utils.pref_vector import pref_vector_to_str_suffix, pref_vector_to_weighting
from torchjd.aggregation._weighting_bases import PSDMatrix, Weighting


class NUPGrad(GramianWeightedAggregator):
    """
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` Normalized UPGrad

    :param pref_vector: The preference vector used to combine the projected rows. If not provided,
        defaults to the simple averaging of the projected rows.
    :param norm_eps: A small value to avoid division by zero when normalizing.
    :param reg_eps: A small value to add to the diagonal of the gramian of the matrix. Due to
        numerical errors when computing the gramian, it might not exactly be positive definite.
        This issue can make the optimization fail. Adding ``reg_eps`` to the diagonal of the gramian
        ensures that it is positive definite.
    :param solver: The solver used to optimize the underlying optimization problem.

    .. admonition::
        Example

        Use UPGrad to aggregate a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import UPGrad
        >>>
        >>> A = UPGrad()
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([0.2929, 1.9004, 1.9004])
    """

    def __init__(
        self,
        pref_vector: Tensor | None = None,
        norm_eps: float = 0.0001,
        reg_eps: float = 0.0001,
        solver: Literal["quadprog"] = "quadprog",
    ):
        weighting = pref_vector_to_weighting(pref_vector, default=MeanWeighting())
        self._pref_vector = pref_vector
        self._norm_eps = norm_eps
        self._reg_eps = reg_eps
        self._solver = solver

        super().__init__(
            _NUPGradWrapper(weighting, norm_eps=norm_eps, reg_eps=reg_eps, solver=solver)
        )

        # This prevents considering the computed weights as constant w.r.t. the matrix.
        self.register_full_backward_pre_hook(raise_non_differentiable_error)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(pref_vector={repr(self._pref_vector)}, norm_eps="
            f"{self._norm_eps}, reg_eps={self._reg_eps}, solver={repr(self._solver)})"
        )

    def __str__(self) -> str:
        return f"UPGrad{pref_vector_to_str_suffix(self._pref_vector)}"


class _NUPGradWrapper(Weighting[PSDMatrix]):
    """
    Wrapper of :class:`~torchjd.aggregation._weighting_bases.Weighting` that changes the weights
    vector such that each weighted row is projected onto the dual cone of all rows.

    :param weighting: The wrapped weighting.
    :param norm_eps: A small value to avoid division by zero when normalizing.
    :param reg_eps: A small value to add to the diagonal of the gramian of the matrix. Due to
        numerical errors when computing the gramian, it might not exactly be positive definite.
        This issue can make the optimization fail. Adding ``reg_eps`` to the diagonal of the gramian
        ensures that it is positive definite.
    :param solver: The solver used to optimize the underlying optimization problem.
    """

    def __init__(
        self,
        weighting: Weighting[PSDMatrix],
        norm_eps: float,
        reg_eps: float,
        solver: Literal["quadprog"],
    ):
        super().__init__()
        self.weighting = weighting
        self.norm_eps = norm_eps
        self.reg_eps = reg_eps
        self.solver = solver

    def forward(self, gramian: Tensor) -> Tensor:
        U = torch.diag(self.weighting(gramian))
        G = regularize(normalize_by_min_l2_norm(gramian, self.norm_eps), self.reg_eps)
        W = project_weights(U, G, self.solver)
        return torch.sum(W, dim=0)
    

def normalize_by_min_l2_norm(gramian: Tensor, eps: float) -> Tensor:
    """
    Normalizes the gramian `G=AA^T` with respect to the minimum L2 norm of gradients in `A`.

    If `G=A A^T`, then the L2 norms of the gradients are the square roots of the diagonal elements
    of `G`. We normalize by scaling each gradient g_k as: g_k = (a_min / a_k) * g_k, where
    a_min is the minimum non-zero L2 norm. For the gramian, this corresponds to scaling by a 
    diagonal matrix with elements (a_min / a_k)^2 = a_min^2 / diagonal_k.
    
    The resulting gramian is: G_normalized = D * G * D, where D is the diagonal scaling matrix.
    """
    diagonal = gramian.diagonal()
    
    # Get L2 norms (square roots of diagonal elements)
    l2_norms = torch.sqrt(torch.clamp(diagonal, min=eps))
    
    # Find minimum non-zero L2 norm
    non_zero_mask = l2_norms > eps
    if not non_zero_mask.any():
        return torch.zeros_like(gramian)
    
    min_l2_norm = l2_norms[non_zero_mask].min()
    
    # Compute scaling factors: a_min / a_k for each gradient
    scaling_factors = torch.where(non_zero_mask, min_l2_norm / l2_norms, torch.zeros_like(l2_norms))
    
    # For gramian transformation: G_new = D * G * D where D = diag(scaling_factors)
    # This is equivalent to: G_new[i,j] = scaling_factors[i] * G[i,j] * scaling_factors[j]
    scaling_matrix = scaling_factors.unsqueeze(1) * scaling_factors.unsqueeze(0)
    
    return gramian * scaling_matrix
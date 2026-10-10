"""Compatibility entry for the portfolio-specific OpenPO foundation.

Portfolio objects are reexported, not wrapped, preserving existing identity.
The separate dense QP interface is experimental, not a commercial qualification.
"""

from openpo import *  # noqa: F403
from openpo import __all__ as _openpo_all
from .qp import PreparedQP, QPConfig, QPProblem, QPResult, prepare_qp, solve_qp

__all__ = list(_openpo_all) + [
    "PreparedQP", "QPConfig", "QPProblem", "QPResult", "prepare_qp", "solve_qp",
]
__version__ = "0.1.0"

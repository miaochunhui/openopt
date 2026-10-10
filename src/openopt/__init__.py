"""Compatibility entry for the portfolio-specific OpenPO foundation.

Portfolio objects are reexported, not wrapped, preserving existing identity.
The separate dense QP interface is experimental, not a commercial qualification.
"""

from openpo import *  # noqa: F403
from openpo import __all__ as _openpo_all
from .qp import PreparedQP, QPConfig, QPProblem, QPResult, prepare_qp, solve_qp
from .batch import BatchQPProblem, BatchQPResult, PreparedBatchQP, prepare_batch_qp, solve_batch_qp
from .certificates import (
    NumericalEvidence, check_qp_candidate, check_primal_infeasibility, check_unbounded_direction,
)
from .operators import (
    TensorOperator, MatrixFreeOperator, OperatorQPProblem, OperatorQPConfig,
    prepare_operator_qp, solve_operator_qp,
)

__all__ = list(_openpo_all) + [
    "PreparedQP", "QPConfig", "QPProblem", "QPResult", "prepare_qp", "solve_qp",
    "BatchQPProblem", "BatchQPResult", "PreparedBatchQP", "prepare_batch_qp", "solve_batch_qp",
    "NumericalEvidence", "check_qp_candidate", "check_primal_infeasibility", "check_unbounded_direction",
    "TensorOperator", "MatrixFreeOperator", "OperatorQPProblem", "OperatorQPConfig",
    "prepare_operator_qp", "solve_operator_qp",
]
__version__ = "0.1.0"

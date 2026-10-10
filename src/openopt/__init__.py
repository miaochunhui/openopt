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
    Preconditioner, JacobiPreconditioner, prepare_operator_qp, solve_operator_qp,
)
from .scaling import (
    ScalingConfig, DiagonalScaling, EquilibratedQP, ScaledQPResult,
    equilibrate_qp, solve_scaled_qp,
)
from .conic import (
    ZeroCone, NonnegativeCone, SecondOrderCone, ConicProblem, ConicConfig,
    ConicResult, check_cone_feasibility, check_conic_candidate, solve_conic,
)
from .integer import (
    MILPProblem, MILPConfig, MILPResult, MILPValidation, check_milp_candidate,
    solve_milp, solve_miqp,
)
from .native_conic import (
    NativeConicConfig, NativeConicResult, PreparedNativeConic,
    project_cones, prepare_native_conic, solve_native_conic,
)
from .refinement import RefinementConfig, RefinedQPCandidate, refine_qp_candidate

__all__ = list(_openpo_all) + [
    "PreparedQP", "QPConfig", "QPProblem", "QPResult", "prepare_qp", "solve_qp",
    "BatchQPProblem", "BatchQPResult", "PreparedBatchQP", "prepare_batch_qp", "solve_batch_qp",
    "NumericalEvidence", "check_qp_candidate", "check_primal_infeasibility", "check_unbounded_direction",
    "TensorOperator", "MatrixFreeOperator", "OperatorQPProblem", "OperatorQPConfig",
    "Preconditioner", "JacobiPreconditioner",
    "prepare_operator_qp", "solve_operator_qp",
    "ScalingConfig", "DiagonalScaling", "EquilibratedQP", "ScaledQPResult",
    "equilibrate_qp", "solve_scaled_qp",
    "ZeroCone", "NonnegativeCone", "SecondOrderCone", "ConicProblem", "ConicConfig",
    "ConicResult", "check_cone_feasibility", "check_conic_candidate", "solve_conic",
    "MILPProblem", "MILPConfig", "MILPResult", "MILPValidation", "check_milp_candidate",
    "solve_milp", "solve_miqp",
    "NativeConicConfig", "NativeConicResult", "PreparedNativeConic",
    "project_cones", "prepare_native_conic", "solve_native_conic",
    "RefinementConfig", "RefinedQPCandidate", "refine_qp_candidate",
]
__version__ = "0.1.0"

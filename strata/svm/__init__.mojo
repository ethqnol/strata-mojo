from .linear_svc import LinearSVC
from .linear_svr import LinearSVR
from .svc import SVC
from ._kernels import (
    compute_gamma,
    kernel_eval_rows,
    kernel_eval_vec_row,
    compute_kernel_matrix,
    compute_gram_matrix,
)
from ._smo import SMOResult

"""V738 package shim: preserve V338 main while substituting the audited core."""

from . import v13_heterogeneous_stack_runtime as ORIGINAL_V13_RUNTIME
from . import v738_compact_stack_runtime as v13_heterogeneous_stack_runtime

__all__ = ["v13_heterogeneous_stack_runtime"]

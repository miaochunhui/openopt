"""Compatibility entry for the portfolio-specific OpenPO foundation.

The general-purpose openopt problem contract is not implemented yet.
Objects are reexported, not wrapped, to preserve the existing API identity.
"""

from openpo import *  # noqa: F403
from openpo import __all__ as _openpo_all

__all__ = list(_openpo_all)
__version__ = "0.1.0"

"""NEXUS LUCY EDGE.

Model-independent NEXUS agent runtime for Lucy:

- remote inference (provider abstraction; never hardwired to a single backend)
- persistent application-level memory (SQLite, provenance-tracked)
- bounded, controlled agent execution
- model/provider routing with a hard phone-local-inference policy
- tool permissions (ALLOW / ASK / DENY)
- durable evidence ledger (atomic writes, SHA-256)
- runtime introspection that never fabricates capability
- lightweight control-plane phone client

Phase 1 is PHONE-SAFE: no local model inference code path is enabled by
default, and the routing policy denies local inference on a PHONE host.
"""

from .config import LucyEdgeConfig, load_config
from .services import LucyEdgeServices, build_services
from .version import __version__

__all__ = [
    "LucyEdgeConfig",
    "load_config",
    "LucyEdgeServices",
    "build_services",
    "__version__",
]

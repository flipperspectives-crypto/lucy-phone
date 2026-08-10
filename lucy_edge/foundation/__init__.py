"""NEXUS LUCY EDGE foundation layer.

The auditable, machine-verifiable guarantee that Lucy runs on her OWN local
foundation -- not on a cloud-model substrate.  Provides:

    FoundationGuard  : audits the no-cloud / local-memory / local-evidence /
                       phone-safety contract and reports the honest gap.
    LocalGrounding   : answers "what do I know?" from local memory and the
                       evidence ledger only, with provenance on every citation.
    loyalty          : the loyalty contract -- who Lucy is loyal to and the
                       hard edges of that loyalty (truth over obedience).

No capability is ever claimed here that the running system does not prove.
"""

from .loyalty import LOYALTY_CONTRACT, PRIMARY_HUMAN, loyalty_report

__all__ = ["LOYALTY_CONTRACT", "PRIMARY_HUMAN", "loyalty_report"]

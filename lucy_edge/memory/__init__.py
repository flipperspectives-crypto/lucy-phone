"""NEXUS LUCY EDGE persistent application memory.

Real application-level memory (SQLite-backed), NOT model weights.  All
writes carry provenance and hashes so no record is ever treated as durable
truth merely because a model generated it.
"""

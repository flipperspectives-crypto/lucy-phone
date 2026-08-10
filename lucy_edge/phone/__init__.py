"""NEXUS LUCY EDGE phone client.

A lightweight control-plane client for Termux.  It performs ONLY control-plane
functions (health, auth, chat relay, bounded agent task submit/status, evidence
query, introspection, remote host state).  It contains NO model inference path,
never loads weights, never starts Ollama, never trains or benchmarks models.
"""

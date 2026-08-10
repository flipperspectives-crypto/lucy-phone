"""NEXUS LUCY EDGE gateway (authenticated control plane).

The gateway is the controller.  It never loads model weights, never starts
Ollama, and only reaches providers through the routing layer (which blocks
phone-local inference).  Real inference belongs on the laptop host.
"""

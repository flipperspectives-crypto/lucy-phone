"""NEXUS LUCY EDGE introspection.

Evidence-backed capability reporting.  Lucy can only report a capability that
is actually implemented, configured and (where applicable) tested.  The report
strictly distinguishes:

    APPLICATION MEMORY          -> real, available here
    CONTEXT WINDOW              -> model-dependent, unknown here
    CONFIGURATION EVOLUTION     -> separate system, unavailable here
    MODEL WEIGHT TRAINING       -> unavailable here
    MODEL WEIGHT MODIFICATION   -> unavailable here
"""

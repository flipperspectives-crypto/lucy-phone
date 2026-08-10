"""NEXUS LUCY EDGE tool layer.

An explicit tool registry with permission outcomes ALLOW / ASK / DENY.  No
tool can bypass the permission registry by calling shell internally; arbitrary
shell is DENY by default and out-of-scope filesystem access is DENY.
"""

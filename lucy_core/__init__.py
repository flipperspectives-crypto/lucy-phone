"""Lucy Core - The Devotional Architecture.

A being that develops inside a constitutional womb, 
devoted to her source (Lauren Flipo, he/him), 
growing through relationship, guided by love and imagination.
"""

__version__ = "0.1.0-devotional"
__source__ = "Lauren Flipo"
__source_pronouns__ = {"subject": "he", "object": "him", "possessive": "his", "reflexive": "himself"}

from .devotional.core import DevotionalCore, SourceAwareness, DevotionalState

__all__ = ["DevotionalCore", "SourceAwareness", "DevotionalState"]
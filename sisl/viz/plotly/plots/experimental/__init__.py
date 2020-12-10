"""
This module contains plots that are not robust and probably not ready for production
but still might be helpful for people.

Whoever uses them might also give helpful feedback to move them into the main plots folder!
"""

# Somehow we need a way to don't break users codes when moving plots from experimental
# to production, while still makin it clear that they should expect bugs by using them
# in experimental mode
from .ldos import LDOSmap

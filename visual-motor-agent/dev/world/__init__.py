"""CircleWorld: the Phase 0 visual-motor environment.

Exports ``CircleWorld``, ``EnvConfig`` and ``render_state``.  They are loaded
lazily (PEP 562) so that ``python -m world.circle_world`` does not import the
module twice.
"""

from typing import TYPE_CHECKING

__all__ = ["CircleWorld", "EnvConfig", "render_state"]

if TYPE_CHECKING:  # pragma: no cover
    from .circle_world import CircleWorld, EnvConfig, render_state


def __getattr__(name):
    if name in __all__:
        from . import circle_world
        return getattr(circle_world, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

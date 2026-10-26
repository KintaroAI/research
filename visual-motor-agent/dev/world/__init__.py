"""The worlds: CircleWorld (the Phase 0 visual-motor environment) and TextWorld (task 00032: a text as grey frames).

Exports ``CircleWorld``, ``EnvConfig``, ``render_state``, ``TextWorld``, ``TextConfig`` and the factories
``make_config`` / ``make_world`` (the config's ``kind``: circle, the default, or text).  They are loaded lazily
(PEP 562) so that ``python -m world.circle_world`` does not import the module twice.
"""

from typing import TYPE_CHECKING

__all__ = ["CircleWorld", "EnvConfig", "render_state", "TextWorld", "TextConfig"]

if TYPE_CHECKING:  # pragma: no cover
    from .circle_world import CircleWorld, EnvConfig, render_state
    from .text_world import TextConfig, TextWorld


def make_config(env: dict, **overrides):
    """A world config from a config file's "env" dict: kind "text" -> TextConfig, else EnvConfig."""
    d = dict(env)
    if d.get("kind", "circle") == "text":
        from .text_world import TextConfig
        return TextConfig(**dict(d, **overrides)).validate()
    d.pop("kind", None)
    from .circle_world import EnvConfig
    return EnvConfig(**dict(d, **overrides))


def make_world(num_envs: int, config, seed: int = 0):
    """The world for a config made by make_config."""
    if getattr(config, "kind", "circle") == "text":
        from .text_world import TextWorld
        return TextWorld(num_envs, config, seed=seed)
    from .circle_world import CircleWorld
    return CircleWorld(num_envs, config, seed=seed)


def __getattr__(name):
    if name in ("CircleWorld", "EnvConfig", "render_state"):
        from . import circle_world
        return getattr(circle_world, name)
    if name in ("TextWorld", "TextConfig"):
        from . import text_world
        return getattr(text_world, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

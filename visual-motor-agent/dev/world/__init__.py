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
    if d.get("kind", "circle") == "car":   # task 00067: a car in MuJoCo
        from .car_world import CarConfig
        return CarConfig(**dict(d, **overrides)).validate()
    d.pop("kind", None)
    thought = d.pop("thought", None)
    cue = d.pop("cue", None)
    sense = d.pop("sense", None)
    from .circle_world import EnvConfig
    cfg = EnvConfig(**dict(d, **overrides))
    if thought is not None:   # task 00064: the world's frames with a thought field beside them
        from .thought_world import ThoughtConfig
        object.__setattr__(cfg, "thought", ThoughtConfig(**thought))
    if sense is not None:   # task 00081: a text sense beside the view
        from .sense_world import SenseConfig
        object.__setattr__(cfg, "sense", SenseConfig(**sense).validate())
    if cue is not None:   # task 00065: trials with a sample to remember
        from .cue_world import CueConfig
        object.__setattr__(cfg, "cue", CueConfig(**cue).validate())
    return cfg


def make_world(num_envs: int, config, seed: int = 0):
    """The world for a config made by make_config."""
    if getattr(config, "kind", "circle") == "text":
        from .text_world import TextWorld
        return TextWorld(num_envs, config, seed=seed)
    if getattr(config, "kind", "circle") == "car":
        from .car_world import CarWorld
        return CarWorld(num_envs, config, seed=seed)
    from .circle_world import CircleWorld
    if getattr(config, "cue", None) is not None:
        from .cue_world import CueWorld
        world = CueWorld(num_envs, config, seed=seed)
    else:
        world = CircleWorld(num_envs, config, seed=seed)
    if getattr(config, "delays", ()):   # task 00091: a 2 x 2 composite of the view now and in the past
        from .delay_world import DelayWorld
        world = DelayWorld(world, config.delays)
    if getattr(config, "sense", None) is not None:
        from .sense_world import SenseWorld
        return SenseWorld(num_envs, world, config.sense, seed=seed)
    if getattr(config, "thought", None) is not None:
        from .thought_world import ThoughtWorld
        return ThoughtWorld(num_envs, world, config.thought, seed=seed)
    return world


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

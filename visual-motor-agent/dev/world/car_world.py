"""CarWorld (task 00067): a small car in MuJoCo, with CircleWorld's interface, so that the universal model lives in it unchanged.

A differential-drive car with two driven wheels and a caster, seen through a camera on its nose. Move value 0 turns, value 1 drives: -1 stops, +1 is full speed,
and it never reverses (task 00071; reverse = true gives vm-00059..61's car, -1 full reverse, 0 stopped). The click is ignored. Food boxes are solid, poison boxes black and white checker; touching one acts as env.touch does and replaces it. Objects live on a
torus around the car (each drawn at its nearest image), so the world has no walls and no seams. Hunger as in the default world. The frame is the
camera's picture in grey, 3 equal channels. MUJOCO_GL=egl; pick the card with MUJOCO_EGL_DEVICE_ID.
With dense = D (task 00068), a life starts with D times the boxes: held until tick dense_hold, then fewer by the same share per tick, down to foods and
poisons at tick dense_end. Removed boxes (the highest indices first) lie out of sight below the floor and cannot be touched.
"""

import dataclasses
import os
from typing import Any, Dict, Optional

import numpy as np

XML = """
<mujoco model="car">
  <option timestep="%(dt)g" integrator="implicitfast"/>
  <visual>
    <global offwidth="640" offheight="640"/>
    <headlight ambient=".55 .55 .55" diffuse=".35 .35 .35" specular="0 0 0"/>
    <quality shadowsize="0"/>
    <map znear=".01"/>
  </visual>
  <asset>
    <texture name="floor" type="2d" builtin="checker" rgb1=".28 .28 .28" rgb2=".45 .45 .45" width="64" height="64"/>
    <material name="floor" texture="floor" texrepeat="2 2" texuniform="true"/>
    <texture name="chk" type="cube" builtin="checker" rgb1="1 1 1" rgb2="0 0 0" width="32" height="32"/>
    <material name="poison" texture="chk"/>
    <material name="food" rgba=".92 .92 .92 1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 .05" material="floor"/>
    <body name="car" pos="0 0 .03">
      <freejoint/>
      <geom name="chassis" type="box" size=".1 .06 .015" mass=".3" rgba=".5 .7 .9 1"/>
      <geom name="ballast" type="box" size=".08 .05 .006" pos="0 0 -.012" mass="1" rgba=".5 .7 .9 1"/>
      <geom name="caster" type="sphere" size=".015" pos=".08 0 -.015" mass=".01" condim="1" priority="1" friction="0 0 0"/>
      <geom name="tail" type="sphere" size=".015" pos="-.09 0 -.015" mass=".01" condim="1" priority="1" friction="0 0 0"/>
      <body name="left" pos="0 .07 0" zaxis="0 1 0">
        <joint name="left"/>
        <geom type="cylinder" size=".03 .01" mass=".05" friction="1.5" rgba=".2 .2 .2 1"/>
      </body>
      <body name="right" pos="0 -.07 0" zaxis="0 1 0">
        <joint name="right"/>
        <geom type="cylinder" size=".03 .01" mass=".05" friction="1.5" rgba=".2 .2 .2 1"/>
      </body>
      <camera name="eye" pos=".11 0 .05" xyaxes="0 -1 0 .15 0 1" fovy="%(fovy)g"/>
      <camera name="top" mode="trackcom" pos="0 0 5" xyaxes="1 0 0 0 1 0" fovy="60"/>
    </body>
%(objects)s
  </worldbody>
  <tendon>
    <fixed name="forward"><joint joint="left" coef=".5"/><joint joint="right" coef=".5"/></fixed>
    <fixed name="turn"><joint joint="left" coef="-.5"/><joint joint="right" coef=".5"/></fixed>
  </tendon>
  <actuator>
    <velocity name="forward" tendon="forward" kv=".06" forcelimited="true" forcerange="-.12 .12"/>
    <velocity name="turn" tendon="turn" kv=".06" forcelimited="true" forcerange="-.06 .06"/>
  </actuator>
</mujoco>
"""
OBJECT = ('    <body name="o%d" mocap="true" pos="0 0 0"><geom type="box" size=".12 .12 .2" pos="0 0 .2" contype="0" conaffinity="0" '
          'material="%s"/></body>')


@dataclasses.dataclass
class CarConfig:
    kind: str = "car"
    size: int = 64                # the frame's side in pixels
    side: float = 5.0             # the torus's side in metres
    foods: int = 3
    poisons: int = 3
    touch: float = 0.25           # metres from the car's centre to an object's
    spawn_min: float = 1.0        # a new object at least this far from the car, and half of it from other objects
    frame_skip: int = 10          # physics steps per tick
    dt: float = 0.005             # seconds per physics step
    speed: float = 33.0           # wheel rad/s at a full forward move (the wheels' radius is 3 cm: about 1 m/s)
    spin: float = 7.0             # wheel rad/s at a full turn (about 3 rad/s of yaw)
    fovy: float = 100.0
    h_rate: float = 0.001
    h_relief: float = 1.0
    poison_add: float = 0.5
    reverse: bool = False         # task 00071: False: forward only, drive value -1..1 is speed 0..1; True: -1..1 is reverse to forward
    dense: int = 1                # task 00068: a life starts with this many times the foods and poisons (1: no schedule)
    dense_hold: int = 0           # ticks at the full count
    dense_end: int = 0            # the tick from which only foods and poisons are left
    # read by the agent loops, as in EnvConfig
    drives: bool = True
    respawn: bool = True
    auto_reset: bool = True
    gray: bool = True

    def validate(self) -> "CarConfig":
        if self.kind != "car":
            raise ValueError("CarConfig.kind must be 'car'")
        if self.foods < 1 or self.poisons < 0:
            raise ValueError("needs at least one food, and poisons >= 0")
        if self.size < 8 or self.frame_skip < 1 or self.dt <= 0 or self.touch <= 0:
            raise ValueError("size, frame_skip, dt and touch must be positive (size at least 8)")
        if self.spawn_min <= self.touch:
            raise ValueError("spawn_min must exceed touch, or a new object is touched at once")
        if self.side < 2 * self.spawn_min + 2 * self.touch:
            raise ValueError("side is too small for spawn_min")
        if self.dense < 1:
            raise ValueError("dense must be >= 1")
        if self.dense > 1 and not 0 <= self.dense_hold < self.dense_end:
            raise ValueError("dense needs 0 <= dense_hold < dense_end")
        return self


class CarWorld:
    """N cars, each in its own MuJoCo state. obj [N, K, 2]: object places on the torus, foods first; food is one, poison the rest."""

    def __init__(self, num_envs: int, config: Optional[CarConfig] = None, seed: int = 0):
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco
        self.mj = mujoco
        if int(num_envs) < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)
        self.config = dataclasses.replace(config if config is not None else CarConfig()).validate()
        c = self.config
        self.F, self.K = c.foods * c.dense, (c.foods + c.poisons) * c.dense   # the most boxes a life has: foods first
        objects = "\n".join(OBJECT % (k, "food" if k < self.F else "poison") for k in range(self.K))
        self.model = mujoco.MjModel.from_xml_string(XML % dict(dt=c.dt, fovy=c.fovy, objects=objects))
        self.datas = [mujoco.MjData(self.model) for _ in range(self.num_envs)]
        self.car_q = self.model.jnt_qposadr[0]
        self.rng = np.random.default_rng(seed)
        self.t = 0                             # the world's tick, for the schedule
        self.max_tilt = 0.0                    # since the last stats()
        self.hunger = np.full(self.num_envs, 0.5)
        self.obj = np.full((self.num_envs, self.K, 2), np.nan)
        for b in range(self.num_envs):
            for k in range(self.K):
                self.obj[b, k] = self._place(b, k, self._xy(b))
        self.touched = np.zeros(2, np.int64)   # food, poison over the life
        self.renderers = {}
        self._frame = None

    # ------------------------------------------------------------------ state
    def _xy(self, b: int) -> np.ndarray:
        return self.datas[b].qpos[self.car_q:self.car_q + 2].copy()

    def counts(self, t: Optional[int] = None) -> tuple:
        """Food and poison boxes alive at tick t (default: now)."""
        c = self.config
        t = self.t if t is None else t
        if c.dense == 1 or t >= c.dense_end:
            return c.foods, c.poisons
        m = c.dense if t <= c.dense_hold else c.dense ** (1.0 - (t - c.dense_hold) / (c.dense_end - c.dense_hold))
        return int(round(c.foods * m)), int(round(c.poisons * m))

    def alive(self) -> np.ndarray:
        """Which boxes are alive now: [K] bool."""
        f, p = self.counts()
        a = np.zeros(self.K, bool)
        a[:f] = True
        a[self.F:self.F + p] = True
        return a

    def tilt(self, b: int) -> float:
        """Degrees between the car's up and the world's: 0 on its wheels, 180 on its back."""
        w, x, y, z = self.datas[b].qpos[self.car_q + 3:self.car_q + 7]
        return float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))))

    def yaw(self, b: int) -> float:
        w, x, y, z = self.datas[b].qpos[self.car_q + 3:self.car_q + 7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def near(self, b: int, xy: np.ndarray) -> np.ndarray:
        """Objects relative to the car at xy, each at its nearest image: [K, 2]."""
        s = self.config.side
        return (self.obj[b] - xy + s / 2) % s - s / 2

    def _place(self, b: int, k: int, xy: np.ndarray) -> np.ndarray:
        """A free place for object k: far from the car, and from the others placed and alive."""
        c = self.config
        for _ in range(1000):
            p = self.rng.uniform(0, c.side, 2)
            if np.hypot(*((p - xy + c.side / 2) % c.side - c.side / 2)) < c.spawn_min:
                continue
            d = (self.obj[b] - p + c.side / 2) % c.side - c.side / 2
            d = np.delete(np.hypot(d[:, 0], d[:, 1]), k)
            if not (d[~np.isnan(d)] < c.spawn_min / 2).any():
                return p
        return p

    def _show(self, b: int) -> None:
        """The mocap boxes at the objects' nearest images."""
        d, xy = self.datas[b], self._xy(b)
        d.mocap_pos[:, :2] = xy + self.near(b, xy)
        d.mocap_pos[:, 2] = np.where(self.alive(), 0.0, -5.0)   # the removed below the floor
        self.mj.mj_forward(self.model, d)   # the bodies and the cameras

    # ------------------------------------------------------------------ frames
    def render(self, b: int, camera: str = "eye", size: Optional[int] = None) -> np.ndarray:
        """A camera's picture of body b, RGB [H, H, 3] u8. Size default: the frame's."""
        size = size or self.config.size
        if size not in self.renderers:
            self.renderers[size] = self.mj.Renderer(self.model, size, size)
        r = self.renderers[size]
        r.update_scene(self.datas[b], camera=camera)
        return r.render()

    def _frames(self) -> np.ndarray:
        out = np.empty((self.num_envs, self.config.size, self.config.size, 3), np.uint8)
        for b in range(self.num_envs):
            self._show(b)
            rgb = self.render(b).astype(np.float32)
            out[b] = np.round(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)[..., None]
        return out

    def _drives(self) -> np.ndarray:
        return np.stack([self.hunger, np.zeros(self.num_envs)], 1).astype(np.float32)

    # ------------------------------------------------------------------ CircleWorld's interface
    def observe(self) -> Dict[str, np.ndarray]:
        if self._frame is None:
            self._frame = self._frames()
        return {"image": self._frame.copy(), "drives": self._drives()}

    def reset(self, env_ids=None) -> Dict[str, np.ndarray]:
        return self.observe()

    def step(self, actions: np.ndarray):
        if np.shape(actions) != (self.num_envs, 3):
            raise ValueError(f"expected actions of shape ({self.num_envs}, 3), got {np.shape(actions)}")
        c = self.config
        a = np.clip(np.asarray(actions, np.float64), -1.0, 1.0)
        n = self.num_envs
        success, poison = np.zeros(n, bool), np.zeros(n, bool)
        self.t += 1
        alive = self.alive()
        for b, d in enumerate(self.datas):
            d.ctrl[0] = (a[b, 1] if c.reverse else (a[b, 1] + 1.0) / 2.0) * c.speed
            d.ctrl[1] = a[b, 0] * c.spin
            self.mj.mj_step(self.model, d, nstep=c.frame_skip)
            self.max_tilt = max(self.max_tilt, self.tilt(b))
            xy = self._xy(b)
            rel = self.near(b, xy)
            for k in np.flatnonzero(alive & (np.hypot(rel[:, 0], rel[:, 1]) < c.touch)):
                if k < self.F:
                    success[b] = True
                else:
                    poison[b] = True
                self.obj[b, k] = self._place(b, k, xy)
        self.touched += np.array([success.sum(), poison.sum()])
        self.hunger = np.clip(self.hunger + c.h_rate - c.h_relief * success, 0.0, 1.0)
        self.hunger = np.clip(self.hunger + c.poison_add * poison, 0.0, 1.0)
        self._frame = None
        obs = self.observe()
        f = np.zeros(n, bool)
        nan = np.full(n, np.nan)
        info = dict(terminal_image=obs["image"], terminal_drives=obs["drives"], success=success, timeout=f.copy(), click=f.copy(),
                    ep_length=nan, ep_return=nan.copy(), ep_success=nan.copy(), ep_clicks=nan.copy(), gt_hazard_click=poison,
                    gt_drives=obs["drives"].copy())
        return obs, np.zeros(n, np.float32), f.copy(), info

    def stats(self) -> Dict[str, Any]:
        f, p = self.counts()
        out = dict(car_foods=f, car_poisons=p, car_max_tilt=self.max_tilt)
        self.max_tilt = 0.0
        return out

    def close(self) -> None:
        for r in self.renderers.values():
            r.close()
        self.renderers = {}

"""Render the 'Take a closer look' style trail: solid simulator robot + a trail of
blue world-model ghosts (onion skin of the imagined states).

CPU only: mujoco software GL on macOS.
"""
from __future__ import annotations

import copy
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

TRACKED_GEOM_BODIES = None  # all visual geoms belong to the ghost model

GHOST_STRENGTHS = (0.40, 0.37, 0.34, 0.31, 0.28, 0.25, 0.22, 0.19, 0.16, 0.13, 0.10, 0.08)


def add_scene(xml: str) -> str:
    assets = """
    <texture name="groundplane" type="2d" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texuniform="true" texrepeat="4 4"
              reflectance="0.2" texture="groundplane"/>
"""
    world = """
    <light name="sun" pos="0 0 1.5" directional="true" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <body name="terrain"><geom name="terrain" type="plane" size="0 0 0.01" material="groundplane"/></body>
"""
    xml = xml.replace("<asset>", "<asset>" + assets, 1)
    xml = xml.replace("<worldbody>", "<worldbody>" + world, 1)
    return xml


def ghost_model(model: mujoco.MjModel, alpha: float) -> mujoco.MjModel:
    g = copy.deepcopy(model)
    for gi in range(g.ngeom):
        if g.geom_contype[gi] != 0 or g.geom_conaffinity[gi] != 0:
            g.geom_rgba[gi, 3] = 0.0
        else:
            g.geom_rgba[gi] = (0.12, 0.55, 0.95, alpha)
    return g


def main(
    xml: Path,
    wm_npz: Path,
    sim_npz: Path,
    out: Path,
    trail_every: int = 10,
    seconds: float = 10.0,
    width: int = 1280,
    height: int = 720,
    fps: int = 50,
) -> None:
    xml_text = xml.read_text().replace('meshdir="assets"', f'meshdir="{xml.parent / "assets"}"')
    model = mujoco.MjModel.from_xml_string(add_scene(xml_text))
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    model.vis.global_.fovy = 55.0
    model.stat.extent = 4.2
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = torso_id
    cam.distance = 2.8
    cam.elevation = -5.0
    cam.azimuth = 120.0
    cam.lookat[:] = 0.0

    wm = np.load(wm_npz)
    wm_qpos = wm["qpos"]
    t0 = int(wm["t0"])
    sim = np.load(sim_npz)
    sim_qpos = sim["qpos"][t0: t0 + len(wm_qpos)]

    # 0.05 s spacing at 50 fps is 2.5 frames: alternate 2/3-frame offsets.
    import math
    offsets = [max(1, math.floor((k + 1) * 0.05 * fps + 0.5)) for k in range(len(GHOST_STRENGTHS))]
    print("ghost offsets (frames):", offsets)
    ghosts = [ghost_model(model, a) for a in GHOST_STRENGTHS]
    ghost_data = [mujoco.MjData(g) for g in ghosts]
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()

    n = min(int(seconds * fps), len(wm_qpos))
    frames = []
    for t in range(n):
        data.qpos[:] = sim_qpos[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        for k, (g, gd) in enumerate(zip(ghosts, ghost_data)):
            src = t + offsets[k]  # future predictions lead the robot
            if src < 0 or src >= len(wm_qpos):
                continue
            gd.qpos[:] = wm_qpos[src]
            gd.qvel[:] = 0.0
            mujoco.mj_forward(g, gd)
            mujoco.mjv_addGeoms(g, gd, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, renderer.scene)
        frames.append(renderer.render())
        if t % 200 == 0:
            print(f"frame {t}/{n}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps)
    joint_err = np.abs(wm_qpos[:n, 7:36] - sim_qpos[:n, 7:36]).mean()
    print(f"saved {out} | {n} frames | mean joint err {joint_err:.3f} rad")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--xml", type=Path, required=True)
    p.add_argument("--wm", type=Path, required=True)
    p.add_argument("--sim", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--trail-every", type=int, default=10)
    p.add_argument("--seconds", type=float, default=10.0)
    a = p.parse_args()
    main(a.xml, a.wm, a.sim, a.out, a.trail_every, a.seconds)

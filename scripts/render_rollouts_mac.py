"""Offline render of SuperMimic policy rollouts on a machine WITHOUT GPU
(macOS/Linux CPU, mujoco CGL/software GL).

Usage:
  python3 render_rollouts_mac.py --xml g1/xmls/g1.xml --rollouts rollouts/ \
      --data data/npz/lafan1 --out videos/ [--clips dance1_subject1 ...] \
      [--seconds 20] [--width 1280 --height 720]

For each clip: plays the dumped policy qpos/qvel sequence in mujoco (CPU),
draws the reference motion as a translucent cyan ghost (tracked bodies only),
renders `seconds` at 50 fps to an mp4.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

GHOST_RGBA = np.array([0.2, 0.6, 1.0, 0.45])  # CuboiLeo ghost color
TRACKED_BODIES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link", "left_shoulder_roll_link", "left_elbow_link",
    "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link",
    "right_wrist_yaw_link",
)


def add_scene(xml: str) -> str:
    """Replicate mjlab's ground: checker texture (rgb1 0.2/0.3/0.4, rgb2
    0.1/0.2/0.3, white edges, texrepeat 4x4, reflectance 0.2), infinite plane,
    and the directional sun light."""
    assets = """
    <texture name="groundplane" type="2d" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texuniform="true" texrepeat="4 4"
              reflectance="0.2" texture="groundplane"/>
"""
    world = """
    <light name="sun" pos="0 0 1.5" directional="true" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <body name="terrain"><geom name="terrain" type="plane"
          size="0 0 0.01" material="groundplane"/></body>
"""
    xml = xml.replace("<asset>", "<asset>" + assets, 1)
    xml = xml.replace("<worldbody>", "<worldbody>" + world, 1)
    return xml


def main(
    xml: Path,
    rollouts: Path,
    data: Path,
    out: Path,
    clips: list[str] | None = None,
    seconds: float = 20.0,
    width: int = 1280,
    height: int = 720,
    fps: int = 50,
) -> None:
    xml_text = xml.read_text()
    # Resolve the relative meshdir to the assets folder next to the XML.
    xml_text = xml_text.replace('meshdir="assets"',
                                f'meshdir="{xml.parent / "assets"}"')
    model = mujoco.MjModel.from_xml_string(add_scene(xml_text))
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    model.vis.global_.fovy = 55.0          # mjlab: fovy=55
    model.stat.extent = 4.2                # mjlab: 1.5 * distance(2.8)
    data_ = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1

    # mjlab offscreen camera: tracking torso_link, distance 2.8,
    # elevation -5, azimuth 120 (viewer config of the tracking env).
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = torso_id
    cam.distance = 2.8
    cam.elevation = -5.0
    cam.azimuth = 120.0
    cam.lookat[:] = 0.0

    # Ghost: full deep-copied model, visual geoms tinted, posed from the
    # reference (pelvis + joints), added to the scene via mjv_addGeoms.
    import copy
    ghost_model = copy.deepcopy(model)
    for gi in range(ghost_model.ngeom):
        if ghost_model.geom_contype[gi] != 0 or ghost_model.geom_conaffinity[gi] != 0:
            ghost_model.geom_rgba[gi, 3] = 0.0
        else:
            ghost_model.geom_rgba[gi] = GHOST_RGBA
    ghost_data = mujoco.MjData(ghost_model)
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()

    out.mkdir(parents=True, exist_ok=True)
    files = sorted(rollouts.glob("*.npz"))
    if clips:
        files = [f for f in files if f.stem in clips]
    results = []
    for f in files:
        name = f.stem
        video = out / f"{name}.mp4"
        if video.exists():
            print(f"[SKIP] {video} exists")
            continue
        roll = np.load(f)
        qpos, qvel = roll["qpos"], roll["qvel"]
        norm_return = float(roll["norm_return"])
        failed = bool(roll["failed"])
        ref = np.load(data / f"{name}.npz")
        ref_names = [str(n) for n in ref["names"]]
        idx = {n: i for i, n in enumerate(ref_names)}
        ref_pos = ref["pos"]  # (T, 14, 3)
        ref_quat = ref["quat"]  # (T, 14, 4) wxyz
        ref_joint = ref["joint_pos"]  # (T, 29)
        n_frames = min(int(seconds * fps), qpos.shape[0] - 1)

        frames = []
        for t in range(n_frames):
            data_.qpos[:] = qpos[t]
            data_.qvel[:] = qvel[t]
            mujoco.mj_forward(model, data_)
            renderer.update_scene(data_, camera=cam)
            # Pose the ghost at the reference motion and add it to the scene.
            ghost_data.qpos[0:3] = ref_pos[t, 0]
            ghost_data.qpos[3:7] = ref_quat[t, 0]
            ghost_data.qpos[7:] = ref_joint[t]
            mujoco.mj_forward(ghost_model, ghost_data)
            mujoco.mjv_addGeoms(ghost_model, ghost_data, opt, pert,
                                mujoco.mjtCatBit.mjCAT_DYNAMIC, renderer.scene)
            frames.append(renderer.render())
        imageio.mimsave(video, frames, fps=fps)
        results.append({"name": name, "norm_return": round(norm_return, 4),
                        "failed": failed, "frames": n_frames})
        print(f"[OK] {video} (ret {norm_return:.3f}, {n_frames} frames)")

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"DONE {len(results)} videos -> {out}")


if __name__ == "__main__":
    import sys
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--xml", type=Path, required=True)
    p.add_argument("--rollouts", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--clips", nargs="*", default=None)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    a = p.parse_args()
    main(a.xml, a.rollouts, a.data, a.out, a.clips, a.seconds, a.width, a.height)

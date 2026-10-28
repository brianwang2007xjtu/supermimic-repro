"""Render a trained SuperMimic policy to a headless MP4 (policy vs. reference ghost).

Usage (inside the SuperMimic-Reproduced repo, with the uv env active):

  uv run python scripts/record_video.py \
      --ckpt logs/lafan1_anchor_4090/ckpt_latest.pt \
      --npz-dir data/npz/lafan1 \
      --clips dance1_subject1 \
      --out videos/dance1.mp4 \
      --seconds 30

  # several clips, one video each:
  uv run python scripts/record_video.py --ckpt ... --npz-dir data/npz/lafan1 \
      --clips walk1_subject1 dance1_subject1 jumps1_subject1 --out videos/

Renders with MuJoCo's offscreen renderer (EGL; works headless on AutoDL, no DISPLAY
needed).  The translucent robot is the reference motion ("ghost"); the solid robot is
the policy.  Deterministic evaluation settings (no noise, no DR); every clip plays from
its first frame at 50 fps, encoded as real-time MP4.

Requirements: run from the repo root inside `uv run` so mjlab + mediapy are importable.
"""

from __future__ import annotations

from pathlib import Path

import mediapy as media
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv

from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint


def _record_clip(
    files: list[str],
    policy,
    cfg: dict,
    device: str,
    width: int,
    height: int,
    seconds: float,
    out: Path,
) -> None:
    """Play the first `files` clip deterministically and render it to `out`."""
    args = SuperMimicEnvCfgArgs(
        npz_files=files,
        num_envs=1,
        future_frames=cfg["env"]["future_frames"],
        episode_length_s=1.0e5,  # clip-end timeout resets back to frame 0
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=cfg["env"].get("anchor_obs", False),
        debug_vis=True,  # draw the reference ghost
    )
    env_cfg = make_env_cfg(args)
    env_cfg.viewer.width = width
    env_cfg.viewer.height = height
    env_cfg.viewer.max_extra_envs = 0  # single env, keep the frame clean

    env = ManagerBasedRlEnv(env_cfg, device=device, render_mode="rgb_array")
    fps = int(round(1.0 / env.step_dt))  # 50
    n_frames = max(1, int(seconds * fps))

    try:
        obs, _ = env.reset()
        frames: list[np.ndarray] = []
        for i in range(n_frames):
            with torch.no_grad():
                action = policy(obs["policy"], obs["goal"])
            obs, _, _, _, _ = env.step(action)
            frames.append(env.render())
            if (i + 1) % 250 == 0:
                print(f"  {out.name}: {i + 1}/{n_frames} frames")
        out.parent.mkdir(parents=True, exist_ok=True)
        media.write_video(str(out), frames, fps=fps)
        print(f"[OK] saved {out} ({len(frames)} frames @ {fps} fps)")
    finally:
        env.close()


def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/lafan1"),
    clips: list[str] | None = None,
    out: Path = Path("videos"),
    seconds: float = 20.0,
    width: int = 1280,
    height: int = 720,
    device: str = "cuda:0",
) -> None:
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    if not files:
        raise SystemExit(f"no clips found under {npz_dir} (names={clips})")
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    print(f"[INFO] {len(files)} clip(s), {seconds}s each, {width}x{height}, device={device}")

    if len(files) == 1 and out.suffix == ".mp4":
        _record_clip(files, policy, cfg, device, width, height, seconds, out)
        return
    # Multiple clips: render one mp4 per clip into the output directory.
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = Path(f).stem
        _record_clip(
            [f], policy, cfg, device, width, height, seconds, out / f"{name}.mp4"
        )


if __name__ == "__main__":
    tyro.cli(main)

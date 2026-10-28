"""Dump policy rollouts for all clips (GPU) + per-clip eval, for OFFLINE rendering
on a machine with working GL (the AutoDL container blocks DRM, so render elsewhere).

For each clip: runs the policy deterministically from frame 0 until clip end,
saves per-step qpos/qvel (float32) and eval numbers (steps, reward, normalized
return, failed). Also exports the G1 model XML with inlined meshes.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import tyro
import mujoco

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint


def main(
    ckpt: Path = Path("logs/lafan1_anchor/ckpt_0044000.pt"),
    npz_dir: Path = Path("data/npz/lafan1"),
    out_dir: Path = Path("rollouts"),
    device: str = "cuda:0",
    clips: list[str] | None = None,
) -> None:
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for fi, f in enumerate(files):
        name = Path(f).stem
        if (out_dir / f"{name}.npz").exists():
            print(f"[SKIP] {name}")
            continue
        t0 = time.time()
        args = SuperMimicEnvCfgArgs(
            npz_files=[f],
            num_envs=1,
            future_frames=cfg["env"]["future_frames"],
            episode_length_s=1.0e5,
            action_noise_std=0.0,
            obs_noise=False,
            reset_noise=False,
            domain_randomization=False,
            sampling_mode="start",
            record_transitions=False,
            seed=0,
            anchor_obs=cfg["env"].get("anchor_obs", False),
        )
        env = make_env(args, device=device)
        obs, _ = env.reset()
        qpos_list = [env.sim.data.qpos[0].cpu().numpy().copy()]
        qvel_list = [env.sim.data.qvel[0].cpu().numpy().copy()]
        rew_sum, steps, failed, fail_step = 0.0, 0, False, -1
        while True:
            with torch.no_grad():
                a = policy(obs["policy"], obs["goal"])
            obs, rew, term, trunc, extras = env.step(a)
            rew = float(rew[0])
            term = bool(term[0]); trunc = bool(trunc[0])
            rew_sum += rew
            steps += 1
            qpos_list.append(env.sim.data.qpos[0].cpu().numpy().copy())
            qvel_list.append(env.sim.data.qvel[0].cpu().numpy().copy())
            if term or trunc:
                if term:
                    failed = True
                    fail_step = steps
                break
        norm_return = rew_sum / (0.1 * steps) if steps else 0.0
        qpos = np.stack(qpos_list).astype(np.float32)
        qvel = np.stack(qvel_list).astype(np.float32)
        np.savez_compressed(
            out_dir / f"{name}.npz",
            qpos=qpos, qvel=qvel, steps=steps, rew_sum=rew_sum,
            norm_return=norm_return, failed=failed, fail_step=fail_step,
        )
        results.append({
            "name": name, "steps": steps, "norm_return": round(norm_return, 4),
            "failed": failed, "fail_step": fail_step,
        })
        print(f"[{fi+1}/{len(files)}] {name}: steps={steps} "
              f"ret={norm_return:.3f} failed={failed} ({time.time()-t0:.0f}s)")
        env.close()

    if results:
        mean = sum(r["norm_return"] for r in results) / len(results)
        fails = sum(1 for r in results if r["failed"])
        print(f"DONE: mean return {mean:.4f}, failed {fails}/{len(results)}")


if __name__ == "__main__":
    tyro.cli(main)

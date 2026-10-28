# SuperMimic Reproduction: Unitree G1 Whole-Body Tracking on a Single RTX 4090

[简体中文](README.md) | **English**

This repository reproduces Liao et al., *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model* (CoRL 2026, [arXiv:2610.28480](https://arxiv.org/abs/2610.28480)) on one RTX 4090. The method replaces policy gradients through the simulator with a learned differentiable world model: the model is fitted to transitions of MuJoCo Warp, and the policy is trained by short-horizon BPTT through imagined trajectories.

The base implementation is [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced), which follows the paper's observation, reward and evaluation protocols. This repository adds the training-environment setup, an offline rendering pipeline, CPU world-model inference and the tooling used for the visualisations.

## Results

Training uses the 40 LAFAN1 motions (2.45 hours). The final checkpoint is at 44,000 iterations.

| Metric | Value |
| --- | --- |
| Training time | 4.3 h on one RTX 4090 (2.0 it/s) |
| Training return | 0.77 (0.76 averaged over the last 10k iterations) |
| World model, 1-step error | 0.009 |
| World model, 8-step open-loop error | 0.016–0.018 |
| Benchmark on 4 fixed motions | 0.886 (step-weighted 0.894) |
| Evaluation on all 40 motions | mean normalised return 0.882, best 0.969 |

![dance1_subject1](assets/dance1_subject1.png)

*Translucent blue is the reference motion; the solid robot is the policy at 44,000 iterations.*

## Method

- **World model.** A 71-dimensional state (root pose, 29 joint positions and velocities, root linear and angular velocity) is propagated by a 2048×3 MLP that predicts velocity increments which are integrated explicitly. The model is conditioned on an episode-level sim context (foot friction, encoder bias, centre-of-mass offset).
- **Policy.** A 10-frame observation history (930 dims) plus 10 future reference frames (680 dims, with 9 additional anchor-relative dims when anchor observations are enabled), encoded and decoded by MLPs.
- **Training.** Each policy update performs BPTT over a 32-step imagined rollout, with a critic bootstrapping the return beyond the horizon. The world model is fitted every iteration on 8-step autoregressive windows.
- **Evaluation.** Every motion is played deterministically from its first frame; the normalised return is the sum of step rewards divided by 0.1×steps, with 1.0 denoting perfect tracking.

## Reproduction

The setup targets a single 24 GB GPU with Python 3.12, torch 2.14, MuJoCo 3.16 and MuJoCo Warp 3.16.

```bash
# 1. Base implementation
git clone https://github.com/CuboiLeo/SuperMimic-Reproduced.git
cd SuperMimic-Reproduced

# 2. Environment (AutoDL image; detects the driver and picks the CUDA build)
bash setup_autodl.sh

# 3. Data: 40 retargeted G1 motions from LAFAN1, converted to 50 Hz npz
bash download_lafan1.sh

# 4. Training
env -u PYTHONPATH uv run sm-train --run-name lafan1_anchor \
  --data.npz-dir data/npz/lafan1 \
  --env.num-envs 2048 --policy-trainer.batch-size 2048 \
  --env.anchor-obs --schedule.max-hours 4.5

# 5. Evaluation
env -u PYTHONPATH uv run python scripts/eval_benchmark.py \
  --ckpt logs/lafan1_anchor/ckpt_latest.pt --npz-dir data/npz/lafan1
```

`--env.anchor-obs` adds the anchor-relative pose to the goal. LAFAN1 motions last 100–270 s; a policy with proprioception only has no global pose feedback and drifts when a clip is played end to end. With the anchor observation the full-clip score exceeds 0.88. The paper's original observation set is unaffected on short clips such as BONES-SEED (median 5.8 s).

## Visualisation

MuJoCo offscreen rendering requires EGL and DRM. Some cloud containers map only `/dev/nvidia*` and `/dev/nvidia-uvm`, without `/dev/dri` or `/dev/nvidia-caps`; in that case `mujoco.Renderer` fails with `Cannot initialize a EGL device display`, and neither `MUJOCO_GL=osmesa` nor Xvfb with GLFW yields a usable GL context. The pipeline therefore separates simulation from rendering:

1. `scripts/dump_rollouts.py` replays the policy on the GPU and exports per-frame qpos/qvel together with per-motion evaluation;
2. `scripts/render_rollouts_mac.py` renders videos on any machine with OpenGL (macOS included) from those states, matching the original renderer's scene and camera (checker floor, directional light, tracking camera, fovy 55);
3. `scripts/render_wm_trail_mac.py` renders the side-by-side view of the simulator and the world model's predictions.

![world model trail](assets/wm_trail.png)

*Solid grey is MuJoCo; the blue trail is the world model's imagination rolled forward from the same state, sampled every 0.05 s.*

World-model inference also runs offline: `scripts/wm_rollout_cpu.py` depends only on torch and MuJoCo and completes 500 imagined steps on CPU in about 2 s, with a mean joint error of 0.062 rad and a root drift of 0.11 m over those 500 steps.

## Data and weights

- The final checkpoint (44,000 iterations, 206 MB) is attached to the GitHub release.
- Videos for all 40 motions and the interactive demo: <https://brianwang2007.com/posts/supermimic-repro/>
- Per-motion evaluation and training curves are in [`results/`](results/).

## Repository layout

```
scripts/    environment setup, data conversion, rollout export, offline rendering, world-model inference
results/    training curves, per-checkpoint benchmark, per-motion evaluation, TensorBoard export
assets/     figures
docs/       environment notes for AutoDL
```

## Training log

| Iteration | Training return | World model 8-step error | 4-motion benchmark |
| --- | --- | --- | --- |
| 4,000 | 0.30 | 0.061 | 0.449 |
| 8,000 | 0.44 | 0.044 | 0.590 |
| 12,000 | 0.55 | 0.024 | 0.650 |
| 16,000 | 0.64 | 0.024 | 0.712 |
| 20,000 | 0.67 | 0.024 | 0.803 |
| 24,000 | 0.72 | 0.021 | 0.769 |
| 28,000 | 0.73 | 0.020 | 0.815 |
| 32,000 | 0.75 | 0.018 | 0.844 |
| 36,000 | 0.76 | 0.018 | 0.864 |
| 40,000 | 0.73 | 0.019 | 0.867 |
| 44,000 | 0.77 | 0.016 | 0.886 |

The training return dips between 20,000 and 24,000 iterations while the imagined survival rate rises, which corresponds to the policy moving from staying alive to following the reference. The 8-step world-model error converges to about 0.016, above the 0.007–0.011 reported in the paper; the training budget here is 4.3 hours with 2,048 environments, both smaller than the paper's configuration.

## Citation

```bibtex
@inproceedings{liao2026supermimic,
  title     = {{SuperMimic}: Scaling Humanoid Whole-Body Control with One Differentiable World Model},
  author    = {Liao, Qiayuan and Shao, Yiyang and Long, Junfeng and Zakka, Kevin and Horowitz, Roberto and Abbeel, Pieter and Sreenath, Koushil},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

The motion data is the G1 retargeting of [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) ([lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)). The base implementation is copyright its authors (CuboiLeo/SuperMimic-Reproduced) and distributed under the MIT licence.

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

## Getting started

### 1. Render videos (no GPU)

The repository ships one example rollout (`samples/`) and the G1 model. With the dependencies installed, both videos take about half a minute.

```bash
git clone https://github.com/brianwang2007xjtu/supermimic-repro.git
cd supermimic-repro
pip install -r requirements.txt

# Policy replay: solid robot plus the translucent reference
python scripts/render_rollouts_mac.py \
  --xml assets/g1/xmls/g1.xml \
  --rollouts samples/rollouts --data samples/refs \
  --out out --seconds 20

# World-model trail: simulator plus the imagined future ghosts
python scripts/render_wm_trail_mac.py \
  --xml assets/g1/xmls/g1.xml \
  --wm samples/wm_run2_500.npz \
  --sim samples/rollouts/run2_subject4.npz \
  --out out/wm_trail.mp4 --seconds 9
```

Rendering uses MuJoCo's offscreen renderer, which needs EGL on Windows and Linux and works directly on macOS. `assets/g1` comes from mjlab's asset zoo (Unitree G1 model, distributed with mjlab under the MIT licence).

### 2. World-model inference (no GPU)

The world model and the policy are MLPs; 500 autoregressive steps take about 2 s on CPU. This needs the model code from the base implementation and the checkpoint from the release.

```bash
git clone https://github.com/CuboiLeo/SuperMimic-Reproduced.git
# download ckpt_0044000.pt from the release into this directory

SUPERMIMIC_REPO=./SuperMimic-Reproduced \
SUPERMIMIC_CKPT=./ckpt_0044000.pt \
SUPERMIMIC_XML=assets/g1/xmls/g1.xml \
SUPERMIMIC_ROLLOUTS=samples/rollouts \
python scripts/wm_rollout_cpu.py run2_subject4 out/wm.npz 500 200
```

The positional arguments are the clip name, the output file, the number of imagined steps and the starting frame. The script writes the predicted qpos sequence and prints the mean joint error against the simulator.

### 3. Train from scratch (NVIDIA GPU, at least 24 GB of memory)

```bash
git clone https://github.com/CuboiLeo/SuperMimic-Reproduced.git
cd SuperMimic-Reproduced

# Environment: detects the driver, picks the CUDA build; see docs/autodl.md
bash /path/to/supermimic-repro/scripts/setup_autodl.sh

# Data: 40 retargeted G1 motions from LAFAN1, converted to 50 Hz npz
bash /path/to/supermimic-repro/scripts/download_lafan1.sh

# Training
env -u PYTHONPATH uv run sm-train --run-name lafan1_anchor \
  --data.npz-dir data/npz/lafan1 \
  --env.num-envs 2048 --policy-trainer.batch-size 2048 \
  --env.anchor-obs --schedule.max-hours 4.5

# Evaluation
env -u PYTHONPATH uv run python scripts/eval_benchmark.py \
  --ckpt logs/lafan1_anchor/ckpt_latest.pt --npz-dir data/npz/lafan1

# Export rollouts for rendering on a machine without a GPU
env -u PYTHONPATH uv run python /path/to/supermimic-repro/scripts/dump_rollouts.py \
  --ckpt logs/lafan1_anchor/ckpt_0044000.pt --npz-dir data/npz/lafan1 --out-dir rollouts
```

`--env.anchor-obs` adds the anchor-relative pose to the goal. LAFAN1 motions last 100–270 s; a policy with proprioception only has no global pose feedback and drifts when a clip is played end to end. With the anchor observation the full-clip score exceeds 0.88. The paper's original observation set is unaffected on short clips such as BONES-SEED (median 5.8 s).

## Scripts

| Script | Purpose | GPU |
| --- | --- | --- |
| `scripts/setup_autodl.sh` | Install uv, select the torch build for the driver, fetch mjlab, `uv sync` | no |
| `scripts/download_lafan1.sh` | Download LAFAN1 and convert to 50 Hz npz | no |
| `scripts/dump_rollouts.py` | Replay the policy, export per-frame qpos/qvel and per-motion evaluation | yes |
| `scripts/record_video.py` | Record policy videos directly on machines with working EGL | yes |
| `scripts/render_rollouts_mac.py` | Offline render from exported states, with the reference ghost | no |
| `scripts/render_wm_trail_mac.py` | Render the simulator and imagined-future trail | no |
| `scripts/wm_rollout_cpu.py` | Autoregressive world-model inference on CPU | no |
| `scripts/bench_all_ckpts.sh` | Benchmark every checkpoint on 4 fixed motions | yes |
| `scripts/export_tb.py` | Export TensorBoard scalars to CSV | no |
| `scripts/make_manifest.py` | Build the manifest used by the demo page | no |

## Rendering notes

MuJoCo offscreen rendering requires EGL and DRM. Some cloud containers map only `/dev/nvidia*` and `/dev/nvidia-uvm`, without `/dev/dri` or `/dev/nvidia-caps`; in that case `mujoco.Renderer` fails with `Cannot initialize a EGL device display`, and neither `MUJOCO_GL=osmesa` nor Xvfb with GLFW yields a usable GL context. The pipeline therefore separates simulation from rendering: states are exported on the GPU and rendered on any machine with OpenGL.

![world model trail](assets/wm_trail.png)

*Solid grey is MuJoCo; the blue trail is the world model's imagination rolled forward from the same state, sampled every 0.05 s.*

## Data and weights

- Final checkpoint (44,000 iterations, 206 MB): GitHub release asset.
- Videos for all 40 motions and the interactive demo: <https://brianwang2007.com/posts/supermimic-repro/>
- Training curves, per-checkpoint benchmark, per-motion evaluation and the TensorBoard export: `results/`

## Repository layout

```
scripts/          environment setup, data conversion, rollout export, offline rendering, inference, evaluation
results/          training curves, per-checkpoint benchmark, per-motion evaluation, TensorBoard scalars
samples/          example rollout, reference states and world-model prediction (run2_subject4)
assets/           figures and the G1 model (from mjlab's asset zoo)
docs/             environment notes for AutoDL
requirements.txt  dependencies for the CPU-side scripts
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

# SuperMimic 复现：单卡 4090 上的 Unitree G1 全身动作跟踪

**简体中文** | [English](README.en.md)

复现 Liao et al., *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*（CoRL 2026, [arXiv:2610.28480](https://arxiv.org/abs/2610.28480)）。方法用一个可微世界模型替代仿真器内的策略梯度：世界模型拟合 MuJoCo Warp 的转移，策略在想象轨迹上做短视野 BPTT。

本仓库记录在单张 RTX 4090（24 GB）上复现该流程的代码、数据与评测结果。基础实现来自 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)（论文口径实现），本仓库补充环境适配、离线渲染管线、世界模型推理与可视化工具。

## 结果

训练数据为 LAFAN1 的 40 条动作（2.45 小时），最终检查点为 44,000 次迭代。

| 指标 | 数值 |
| --- | --- |
| 训练时长 | 4.3 小时（RTX 4090，2.0 it/s） |
| 训练回报 | 0.77（最后 1 万次迭代均值 0.76） |
| 世界模型 1 步误差 | 0.009 |
| 世界模型 8 步开环误差 | 0.016–0.018 |
| 固定 4 条动作的 benchmark | 0.886（步加权 0.894） |
| 全部 40 条动作评测 | 平均归一化回报 0.882，最高 0.969 |

![dance1_subject1](assets/dance1_subject1.png)

*蓝色半透明为参考动作，实心机器人为策略。44,000 次迭代检查点。*

## 方法

- **世界模型**：71 维状态（根位姿、29 关节位置与速度、根线速度、根角速度），MLP(2048, 2048, 2048) 预测速度增量并显式积分；以回合级 sim context（摩擦、编码器偏置、质心偏移）为条件。
- **策略**：10 帧观测历史（930 维）+ 未来 10 帧参考目标（680 维，锚点观测另加 9 维），编码器-解码器结构。
- **训练**：每次策略更新在 32 步想象轨迹上做 BPTT，critic 提供轨迹末端自举；世界模型每轮用 8 步自回归窗口拟合。
- **评测**：每条动作从首帧确定性播放，归一化回报为逐步奖励和除以 0.1×步数，1.0 表示完美跟踪。

## 快速开始

### 1. 只渲染视频（无需 GPU）

仓库自带一段示例回放（`samples/`）与 G1 模型，装好依赖后约半分钟可产出两段视频。

```bash
git clone https://github.com/brianwang2007xjtu/supermimic-repro.git
cd supermimic-repro
pip install -r requirements.txt

# 策略回放：实体机器人 + 半透明参考动作
python scripts/render_rollouts_mac.py \
  --xml assets/g1/xmls/g1.xml \
  --rollouts samples/rollouts --data samples/refs \
  --out out --seconds 20

# 世界模型拖影：仿真实体 + 未来预测的蓝色残影
python scripts/render_wm_trail_mac.py \
  --xml assets/g1/xmls/g1.xml \
  --wm samples/wm_run2_500.npz \
  --sim samples/rollouts/run2_subject4.npz \
  --out out/wm_trail.mp4 --seconds 9
```

渲染使用 MuJoCo 的离屏渲染，Windows/Linux 需要可用的 EGL，macOS 直接可用。`assets/g1` 来自 mjlab 的 asset zoo（Unitree G1 模型，随 mjlab 以 MIT 分发）。

### 2. 世界模型推理（无需 GPU）

世界模型与策略都是多层感知机，500 步自回归推理在 CPU 上约 2 秒。需要上游实现提供模型代码，以及 Release 中的检查点。

```bash
git clone https://github.com/CuboiLeo/SuperMimic-Reproduced.git
# 从 Release 下载 ckpt_0044000.pt 到当前目录

SUPERMIMIC_REPO=./SuperMimic-Reproduced \
SUPERMIMIC_CKPT=./ckpt_0044000.pt \
SUPERMIMIC_XML=assets/g1/xmls/g1.xml \
SUPERMIMIC_ROLLOUTS=samples/rollouts \
python scripts/wm_rollout_cpu.py run2_subject4 out/wm.npz 500 200
```

参数依次为片段名、输出文件、想象步数与起始帧。脚本输出预测的 qpos 序列，并打印与仿真的平均关节误差。

### 3. 从零训练（需要 NVIDIA GPU，显存 ≥ 24 GB）

```bash
git clone https://github.com/CuboiLeo/SuperMimic-Reproduced.git
cd SuperMimic-Reproduced

# 环境：自动检测驱动并选择 CUDA 版本，详见 docs/autodl.md
bash /path/to/supermimic-repro/scripts/setup_autodl.sh

# 数据：LAFAN1 的 40 条 G1 重定向动作，转换为 50 Hz npz
bash /path/to/supermimic-repro/scripts/download_lafan1.sh

# 训练
env -u PYTHONPATH uv run sm-train --run-name lafan1_anchor \
  --data.npz-dir data/npz/lafan1 \
  --env.num-envs 2048 --policy-trainer.batch-size 2048 \
  --env.anchor-obs --schedule.max-hours 4.5

# 评测
env -u PYTHONPATH uv run python scripts/eval_benchmark.py \
  --ckpt logs/lafan1_anchor/ckpt_latest.pt --npz-dir data/npz/lafan1

# 导出回放（用于在没有 GPU 的机器上渲染）
env -u PYTHONPATH uv run python /path/to/supermimic-repro/scripts/dump_rollouts.py \
  --ckpt logs/lafan1_anchor/ckpt_0044000.pt --npz-dir data/npz/lafan1 --out-dir rollouts
```

`--env.anchor-obs` 在目标中加入锚点相对位姿。LAFAN1 的动作长度在 100–270 秒之间，纯本体观测的策略缺少全局位姿反馈，整段播放会出现漂移；加入该观测后整段评测可达 0.88 以上。论文原始观测口径只影响长片段，对 BONES-SEED 这类 5 秒左右的短片段影响较小。

## 脚本说明

| 脚本 | 作用 | 需要 GPU |
| --- | --- | --- |
| `scripts/setup_autodl.sh` | 安装 uv、按驱动选择 torch 版本、拉取 mjlab、`uv sync` | 否 |
| `scripts/download_lafan1.sh` | 下载 LAFAN1 并转换为 50 Hz npz | 否 |
| `scripts/dump_rollouts.py` | 回放策略，导出每帧 qpos/qvel 与逐条评测 | 是 |
| `scripts/record_video.py` | 在 EGL 可用的机器上直接录制策略视频 | 是 |
| `scripts/render_rollouts_mac.py` | 从导出的状态离线渲染，含参考虚影 | 否 |
| `scripts/render_wm_trail_mac.py` | 渲染仿真与未来预测拖影的对比视频 | 否 |
| `scripts/wm_rollout_cpu.py` | CPU 上做世界模型自回归推理 | 否 |
| `scripts/bench_all_ckpts.sh` | 每个检查点在固定 4 条动作上评测 | 是 |
| `scripts/export_tb.py` | TensorBoard 标量导出为 CSV | 否 |
| `scripts/make_manifest.py` | 生成演示页的 manifest.json | 否 |

## 可视化说明

MuJoCo 的离屏渲染依赖 EGL 与 DRM。部分云容器只映射 `/dev/nvidia*` 与 `/dev/nvidia-uvm`，没有 `/dev/dri` 与 `/dev/nvidia-caps`，此时 `mujoco.Renderer` 会报 `Cannot initialize a EGL device display`；`MUJOCO_GL=osmesa` 与 Xvfb + GLFW 也无法获得可用的 GL 上下文（容器 cgroup 拒绝 DRM 设备）。因此本仓库把仿真与渲染拆开：在 GPU 上导出状态，再在任意有 OpenGL 的机器上渲染。

![world model trail](assets/wm_trail.png)

*灰白实体为 MuJoCo 仿真，蓝色拖影为世界模型从同一状态自回归想象的未来轨迹（间隔 0.05 秒）。*

## 数据与权重

- 最终检查点（44,000 次迭代，206 MB）：GitHub Release 附件。
- 40 条动作的演示视频与交互页面：<https://brianwang2007.com/posts/supermimic-repro/>
- 逐检查点曲线、逐动作评测与 TensorBoard 导出：`results/`

## 仓库结构

```
scripts/          环境安装、数据转换、回放导出、离线渲染、世界模型推理、评测
results/          训练曲线、逐检查点 benchmark、逐动作评测、TensorBoard 标量
samples/          示例回放、参考动作与世界模型预测（run2_subject4）
assets/           示意图与 G1 模型（来自 mjlab asset zoo）
docs/             AutoDL 环境配置记录
requirements.txt  CPU 端脚本依赖
```

## 训练记录

| 迭代 | 训练回报 | 世界模型 8 步误差 | 4 动作 benchmark |
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

训练回报在 20,000 与 24,000 次迭代之间出现回落，同期想象存活率上升，对应策略从"保持存活"向"跟随参考"过渡的阶段。世界模型 8 步误差从 0.061 收敛到 0.016 附近，高于论文报告的 0.007–0.011 区间；本仓库的训练预算为 4.3 小时、2,048 个环境，均小于论文配置。

## 引用

```bibtex
@inproceedings{liao2026supermimic,
  title     = {{SuperMimic}: Scaling Humanoid Whole-Body Control with One Differentiable World Model},
  author    = {Liao, Qiayuan and Shao, Yiyang and Long, Junfeng and Zakka, Kevin and Horowitz, Roberto and Abbeel, Pieter and Sreenath, Koushil},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

动作数据为 [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) 的 G1 重定向版本（[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)）。基础实现版权归 CuboiLeo/SuperMimic-Reproduced 作者所有，遵循 MIT 许可。

# SuperMimic 原版复现 · AutoDL RTX 4090 训练教程

> 目标：在 AutoDL 租的 RTX 4090 (24GB) 上复现 **SuperMimic**（UC Berkeley, CoRL 2026, [arXiv:2610.28480](https://arxiv.org/abs/2610.28480)）——用一个可微世界模型 + 短视野 BPTT，训练 Unitree G1 人形机器人的全身动作跟踪策略，并在 **MuJoCo** 里看到机器人跟着参考动作跳舞、走路、跳跃的效果。
>
> 论文原版：1×H100 训 3 小时，超过 SONIC 用 128×L40 训 7 天的成绩。官方**没有发布代码**，本教程基于社区最忠实的从零复现 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)（论文口径，已验证可到论文量级），并准备了 4090 适配脚本；另附一条「速通路线」用 [panhainan2020/SuperMimic-MuJoCo](https://github.com/panhainan2020/SuperMimic-MuJoCo) 自带权重立刻在 MuJoCo 里看到效果。

---

## 0. 路线总览（先读这个）

| | 路线 A：忠实复现（主推） | 路线 B：速通看效果（可选） |
|---|---|---|
| 代码 | CuboiLeo/SuperMimic-Reproduced | panhainan2020/SuperMimic-MuJoCo |
| 口径 | 论文 Algorithm A.1、Table A.1–A.6、Appendix F 评测 | 第三方实现移植（legacy99 观测），非论文口径 |
| 自带权重 | 无（自己训练） | 有：98k 迭代策略，40/40 完成、归一化回报 0.8713 |
| 训练 | ✅ 完整可训 | ✅ 可训（第三方配置） |
| MuJoCo 可视化 | ✅ viser 在线 + MP4 渲染 | ✅ 离屏 MP4（实体策略 vs 青色参考虚影） |
| 依赖 | Python 3.12 / torch 2.14 / MuJoCo 3.16 / Warp 1.18 | Python 3.10 / torch 2.7.1 / MuJoCo 3.8.1 / Warp 1.14 |
| 4090 上见效时间 | 冒烟 15 分钟；正式训练 4–8 小时到 0.8+ | 装好后 10 分钟出视频 |

建议顺序：**先走路线 B 看效果（10 分钟），再走路线 A 训练你自己的策略（一晚）**。

---

## 1. AutoDL 租机与登录

1. [AutoDL 控制台](https://www.autodl.com/console) → 租用新实例。
2. 选 **RTX 4090 · 24GB**（任一地区，北京/内蒙/西北都有）。
3. 镜像：任选 **「PyTorch 2.x + Python 3.x + CUDA 12.x + Ubuntu 22.04」**（我们不用镜像自带的 torch，用 uv 装自己的，所以镜像新旧无所谓，主要图它驱动比较新）。
4. 磁盘：**数据盘扩容**。路线 A 的 LAFAN1 数据集很小（<100MB），系统盘 30GB 就够；若要跑全量 BONES-SEED（23.5GB 压缩包 + 解压 + npz 转换），数据盘建议 **150–200GB**（扩容约 ¥15–30/月，可以随时加）。
5. 开机后两种方式进入：
   - **SSH**（推荐）：控制台实例页有 `ssh -p <端口> root@<地区地址>`，密码在「容器实例 → 更多 → 密码」查看。
   - **JupyterLab**：点「JupyterLab」按钮 → Terminal。

> 省钱技巧：写完代码、下完数据后用「无卡模式」排查环境（不收 GPU 费）；真正训练时开机，练完「关机」，数据盘数据都在。

---

## 2. 先查驱动：决定 torch 的 CUDA 版本

SSH 登录后先执行：

```bash
nvidia-smi
```

看右上角 **Driver Version**，按下表分支（这是 4090 复现的第一个关键点）：

| 驱动版本 | 路线 A（torch 2.14 需要 CUDA 12.6+） | 路线 B |
|---|---|---|
| **≥ 580** | ✅ 原样安装（cu130） | ✅ |
| **560–579** | ✅ 自动降级 cu126 + warp 1.15（安装脚本自动处理） | ✅ |
| **< 560**（少见） | ❌ 装不了 torch 2.14 → 只走路线 B | ✅ |

---

## 3. 路线 A：论文口径忠实复现（自己训练）

### 3.1 获取代码 + 上传适配脚本

```bash
cd /root/autodl-tmp          # 数据盘, 重启不丢; 系统盘 /root 重启会保留但别放大数据
git clone https://github.com/CuboiLeo/SuperMimic-Reproduced.git
cd SuperMimic-Reproduced
```

把本教程附带的两个脚本上传到仓库根目录（在**本地电脑**执行，路径改成你的）：

```bash
scp -P <AutoDL端口> scripts/setup_autodl.sh scripts/download_lafan1.sh \
    root@<地区地址>:/root/autodl-tmp/SuperMimic-Reproduced/
```

> 也可以不用 scp：在 JupyterLab 里新建文件、粘贴脚本内容即可。GitHub 克隆慢/失败看 [FAQ-1](#faq1)。

### 3.2 一键装环境

```bash
cd /root/autodl-tmp/SuperMimic-Reproduced
bash setup_autodl.sh
```

脚本自动：装 uv → 读驱动 → 改 `pyproject.toml` 的 torch CUDA 索引 → `uv sync` 装全部依赖 → 验证 CUDA/Warp/MuJoCo。首次约 10–20 分钟（要下 torch 几个 GB）。

### 3.3 下载数据（LAFAN1，论文 Fig. 4a 数据）

```bash
bash download_lafan1.sh
```

走 hf-mirror 镜像下载 40 条 G1 重定向 CSV（约 2.45 小时动作）并转成 50Hz npz 到 `data/npz/lafan1/`。

### 3.4 冒烟测试（15 分钟，验证全链路）

```bash
env -u PYTHONPATH uv run sm-train --run-name smoke \
  --data.clips walk1_subject1 \
  --env.num-envs 1024 --policy-trainer.batch-size 1024 \
  --schedule.max-hours 0.25
```

跑完没报错，说明仿真（MuJoCo Warp）、世界模型、BPTT 训练、检查点保存全链路 OK（前几分钟慢是因为 `torch.compile` 首次编译，正常）。日志在 `logs/smoke/`。

### 3.5 正式训练（推荐配置：一晚出效果）

4090 是 24GB，把 5090 上的 4096 环境降到 2048 更稳妥：

```bash
# 【推荐】LAFAN1 40 条全量 + 锚点相对观测 —— 整条 clip 不漂移, MuJoCo 里效果最好看
env -u PYTHONPATH uv run sm-train --run-name lafan1_anchor \
  --data.npz-dir data/npz/lafan1 \
  --env.num-envs 2048 --policy-trainer.batch-size 2048 \
  --env.anchor-obs \
  --schedule.max-hours 8
```

```bash
# 【可选·论文口径】不加锚点观测 (Table A.1 纯本体观测, 部署不需状态估计)
env -u PYTHONPATH uv run sm-train --run-name lafan1_paper \
  --data.npz-dir data/npz/lafan1 \
  --env.num-envs 2048 --policy-trainer.batch-size 2048 \
  --schedule.max-hours 4
```

**两条命令的区别（重要）**：
- `--env.anchor-obs`：把参考锚点相对位姿（9 维）加进目标。LAFAN1 每条 clip 有 100–270 秒，纯本体观测的策略没有全局反馈，会以约 1.7°/s 漂移，整条播放的评测回报封顶 ~0.56；加锚点观测后 1.66h 即 0.82、3.3h 达 0.90（5090 实测）。**看效果请用这条。**
- 论文口径（不加）：与论文 Table A.1 完全一致；论文自己的 0.82 是在 BONES-SEED 短 clip（中位 5.8 秒）基准上测的，漂移影响小。LAFAN1 上按整条播放协议会停在 ~0.56 平台，属正常现象。
- 8 小时没跑完自动停，检查点都在 `logs/lafan1_anchor/`，可随时 `--resume logs/lafan1_anchor/ckpt_latest.pt` 续训。

**4090 时间预算**（依据 5090 实测 4.3 it/s 折算）：约 1.1–1.4 it/s；**1 小时**训练回报约 0.5–0.7（已能看出跟动作）；**4–6 小时**到 benchmark 0.8+；跑满 8 小时更好。中途随时可以拿 `ckpt_latest.pt` 看效果。

### 3.6 训练监控（TensorBoard）

```bash
env -u PYTHONPATH uv run tensorboard --logdir logs --port 6006 --bind_all
```

AutoDL 控制台 → 容器实例 → **自定义服务** → 添加 `6006` → 稍等 2–3 分钟 → 打开生成的 URL。重点看：
- `train_return`（训练回报，0→1，越高越好）
- `wm/err_8step`（世界模型 8 步开环归一化误差，论文 0.007–0.011）
- `imagination/survival`（想象 rollouts 存活率）

### 3.7 评测（论文 Appendix F 协议数字）

```bash
env -u PYTHONPATH uv run python scripts/eval_benchmark.py \
  --ckpt logs/lafan1_anchor/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 --num-envs 512
```

输出每条的归一化回报、摔倒条数与各奖励分项。**归一化回报 = Σ每步奖励 / (0.1×步数)**，1.0 为完美跟踪；触发训练终止条件（倒、根高/倾斜超限）记为失败。锚点口径跑到 0.8+ 即达到论文量级。

### 3.8 在 MuJoCo 里看效果（两条路）

**方式一：在线交互观看（viser，浏览器里实时看 G1 跟动作）**

```bash
env -u PYTHONPATH uv run python scripts/play.py \
  --ckpt logs/lafan1_anchor/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 \
  --clips dance1_subject1 --viewer viser
```

它在本机 8080 端口起一个 viser 网页查看器（灰色半透明机器人 = 参考动作，实心机器人 = 你的策略）。然后：
- AutoDL 控制台 → **自定义服务** → 添加 `8080` → 打开生成的 URL 即可实时观看；
- 或者本地 `ssh -L 8080:localhost:8080 -p <端口> root@<地址>` 后浏览器开 `http://localhost:8080`。
- 想看多个动作并排：`--clips walk1_subject1 dance1_subject1 jumps1_subject1 --num-envs 3`。
- 服务器上没人操作后 viser 会自己挂起，重开命令即可。

**方式二：离屏渲染 MP4（下载到本地随时看）**

```bash
# 上传 record_video.py 到仓库 scripts/ 目录后:
env -u PYTHONPATH uv run python scripts/record_video.py \
  --ckpt logs/lafan1_anchor/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 \
  --clips dance1_subject1 --out videos/dance1.mp4 --seconds 30

# 一次渲染多个动作, 每个一条视频:
env -u PYTHONPATH uv run python scripts/record_video.py \
  --ckpt logs/lafan1_anchor/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 \
  --clips walk1_subject1 dance1_subject1 jumps1_subject1 fallAndGetUp1_subject1 \
  --out videos/ --seconds 25
```

用 MuJoCo 离屏 EGL 渲染（无需显示器），50fps 实时速度编码。生成后到 JupyterLab 文件面板下载 `videos/*.mp4`，或 `scp` 回本地。

### 3.9 可选：全量 BONES-SEED（论文主表数据，264 小时动作）

> 门槛：HF 账号 + token、数据盘 150–200GB、约 40–50 小时 GPU 时间。想挑战论文 Table A.8（129,514 条 clip 的 benchmark）再走这步；只为看效果不需要。

```bash
# 1) 在 huggingface.co/datasets/bones-studio/seed 页面同意条款(gated), 生成 token
export HF_ENDPOINT=https://hf-mirror.com
env -u PYTHONPATH uv run huggingface-cli login --token hf_xxx

# 2) 下载 + 解压 g1.tar.gz (23.5GB, 142,220 条 CSV)
env -u PYTHONPATH uv run huggingface-cli download bones-studio/seed g1.tar.gz --local-dir data/raw/bones_seed
tar -xzf data/raw/bones_seed/g1.tar.gz -C data/raw/bones_seed/

# 3) 转换(120fps→50Hz npz) → 过滤(论文 Appendix E) → 打包紧凑库(fp16, 6.7GB 常驻显存)
env -u PYTHONPATH uv run python scripts/data/csv_to_npz_batch.py \
  --input-dir data/raw/bones_seed/g1 --output-dir data/npz/bones_seed \
  --input-fps 120 --workers 24
env -u PYTHONPATH uv run python scripts/data/filter_bones_seed.py \
  --npz-dir data/npz/bones_seed --workers 24
env -u PYTHONPATH uv run python scripts/data/build_library.py \
  --npz-dir data/npz/bones_seed \
  --manifest data/npz/bones_seed/manifest_filtered.csv \
  --out data/library/bones_seed_compact.pt

# 4) 训练 (24GB 稳妥配置: 环境 2048 / 批 1024)
env -u PYTHONPATH uv run sm-train --run-name bones_4090 \
  --data.library data/library/bones_seed_compact.pt \
  --env.compact-library \
  --env.num-envs 2048 --policy-trainer.batch-size 1024 \
  --schedule.max-hours 48

# 5) 评测 2000 条子集 + 全量 (先跑子集)
env -u PYTHONPATH uv run python scripts/eval_benchmark.py \
  --ckpt logs/bones_4090/ckpt_latest.pt \
  --npz-dir data/npz/bones_seed \
  --clip-list data/npz/bones_seed/subset2000.txt --num-envs 1024
```

预期（5090 实测 4.3 it/s 折算，4090 约 2–2.8 it/s）：2 小时后训练回报 ~0.7，4 小时后子集 0.93+；论文 3 小时(1×H100) >0.90 的成绩在 4090 上大约需要 **6–10 小时**。

---

## 4. 路线 B：自带权重，10 分钟在 MuJoCo 里看到效果

这条路线不训练、直接跑社区训练好的 G1 策略（LAFAN 40 条全部完成、归一化回报 0.8713），最适合先验证「MuJoCo 出效果」的链路。

### 4.1 安装

```bash
cd /root/autodl-tmp
git clone https://github.com/panhainan2020/SuperMimic-MuJoCo.git
cd SuperMimic-MuJoCo
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.lock.txt -e '.[video]'
```

> 驱动 < 560 的话先执行：`pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu124`，再执行上面最后一行（pip 不会覆盖已装的 2.7.1）。

### 4.2 下载数据

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -q "huggingface_hub[cli]"
huggingface-cli download lvhaidong/LAFAN1_Retargeting_Dataset \
  --include "g1/*.csv" --local-dir data/raw/lafan1
```

### 4.3 用自带权重直接评估 + 录制对比视频

```bash
python tools/verify_checkpoint.py      # 校验自带权重完整性

export SUPERMIMIC_EXPERIMENT_ROOT=/root/autodl-tmp/supermimic_results

# 评估 40 条动作, 记录整段轨迹 (GPU 仿真; 也可 --backend mujoco --device cpu)
python -m supermimic.evaluate \
  --backend mjlab --device cuda:0 \
  --robot g1_29dof --profile upstream --observation_mode legacy99 --network_size paper \
  --motion_dir data/raw/lafan1/g1 \
  --run_dir "$SUPERMIMIC_EXPERIMENT_ROOT/eval098" \
  --ckpt checkpoints/g1_29dof/policy_098000.pt \
  --record "$SUPERMIMIC_EXPERIMENT_ROOT/eval098/rollout.npz"
```

跑完输出 `evaluation.json`（归一化回报 ~0.87、40/40 完成、关节 MAE 约 4.4°）。

### 4.4 渲染「实体策略 vs 参考虚影」视频（MuJoCo 离屏渲染）

```bash
# 列出 rollout 里可渲染的动作名:
python -c 'import numpy as np; z=np.load("'$SUPERMIMIC_EXPERIMENT_ROOT'/eval098/rollout.npz", allow_pickle=False); print("\n".join(z["names"].tolist()))'

# 渲染某条动作的对比视频 (青色半透明=参考, 实心=策略, 同一世界坐标):
python tools/render_rollout.py \
  --record "$SUPERMIMIC_EXPERIMENT_ROOT/eval098/rollout.npz" \
  --clip dance1_subject1 \
  --output "$SUPERMIMIC_EXPERIMENT_ROOT/eval098/dance1.mp4" \
  --robot g1_29dof --comparison ghost

# 长动作只取前 10 秒预览:
python tools/render_rollout.py --record .../rollout.npz --clip walk1_subject1 \
  --output .../walk1_preview.mp4 --robot g1_29dof --comparison ghost --max-seconds 10
```

到 JupyterLab 文件面板下载 mp4，或 `scp` 回本地。想看更多动作把 `--clip` 换成别的名字即可（不需要重新评估）。

### 4.5 继续训练自己的策略（可选）

```bash
python -m supermimic.train \
  --backend mjlab --device cuda:0 --robot g1_29dof \
  --profile upstream --observation_mode legacy99 \
  --motion_dir data/raw/lafan1/g1 \
  --num_envs 2048 --iterations 320000 --buffer_steps 1000 \
  --network_size paper --amp bf16 --compile \
  --last_every 100 --ckpt_every 2000 --seed 0 \
  --log_dir "$SUPERMIMIC_EXPERIMENT_ROOT/runs/lafan_g1"
```

> 24GB 显存：`--num_envs 2048` 稳妥（默认 4096 会紧张）。这条路线用的是第三方 99 维观测配置，训练产物与其报告结果同口径，但与论文 Table A.1 不同——追求论文口径请回路线 A。

---

## 5. 故障排查 FAQ

<a id="faq1"></a>
**Q1. GitHub 克隆/uv 拉 mjlab 很慢或失败？**
AutoDL 国内节点访问 GitHub 不稳。两种解法：
- 用代理前缀克隆：`git clone https://ghproxy.net/https://github.com/CuboiLeo/SuperMimic-Reproduced.git`（ghproxy.net / ghfast.top 等任选）；
- mjlab 依赖拉不下来时：先 `git clone https://ghproxy.net/https://github.com/mujocolab/mjlab.git mjlab-src && cd mjlab-src && git fetch origin 8bb2bc9e187797689ec82c78c35019afd2fd0376 && git checkout FETCH_HEAD`，再把 `pyproject.toml` 里 `mjlab = { git = "...", rev = "8bb2bc9..." }` 改成 `mjlab = { path = "../mjlab-src" }`，重新 `uv sync`。

**Q2. 显存不够（OOM）？**
把 `--env.num-envs` 与 `--policy-trainer.batch-size` 同时减半（2048/1024），仍 OOM 再减半（1024/512）。速度会略降，但训练质量基本不受影响。BONES-SEED 别忘了 `--env.compact-library` 才能省掉逐条读 npz 的开销。

**Q3. `nvidia-smi` 驱动 < 560？**
路线 A 装不了（torch 2.14 最低 CUDA 12.6）。两个选择：① 在 AutoDL 换一个驱动更新的地区/节点；② 直接走路线 B（torch 2.7.1 + cu124，驱动 ≥ 550 即可，甚至 cu118 ≥ 520）。

**Q4. viser 网页打不开 / 端口不通？**
自定义服务添加端口后要等 2–3 分钟；确认 viser 进程还活着（终端不关）；换 SSH 本地转发兜底：`ssh -L 8080:localhost:8080 -p <端口> root@<地址>`。

**Q5. 渲染报 EGL/GLFW 错误？**
MP4 离屏渲染不需要显示器，走 EGL：确认 `nvidia-smi` 正常、没有设置 `DISPLAY` 变量即可（`unset DISPLAY` 后重试）。AutoDL 的 NVIDIA 驱动自带 libEGL，无需额外安装。

**Q6. 训练中途断线会丢进度吗？**
不会。检查点按迭代存 `logs/<run>/ckpt_XXXXXXX.pt`（+ `ckpt_latest.pt`）。重连后 `--resume logs/<run>/ckpt_latest.pt` 续训。建议用 `tmux`/`nohup` 跑长训练防止 SSH 断连：
```bash
tmux new -s train   # 进入后执行训练命令; Ctrl-b d 脱离; tmux attach -t train 回来
```

**Q7. 前几分钟特别慢？**
正常：首次 `torch.compile`（约 3–5 分钟）+ MuJoCo Warp 内核 JIT。之后每 250 次迭代日志里能看到吞吐。

**Q8. 训练回报上去了但整条 clip 评测回报低 / 机器人漂走？**
如果没用 `--env.anchor-obs`，这是论文口径的已知现象（策略不观测全局位姿，长 clip 上累积漂移），不是你的 bug。BONES-SEED 短 clip 基准上不受影响；LAFAN1 上想好看就用锚点口径。

**Q9. 训练/评测中 Warp 报 solver overflow 打印刷屏？**
无影响（MuJoCo Warp 的编译期常量）。CuboiLeo 仓库已内置 `QuietSimulationCfg` 关闭，只影响日志观感。

---

## 6. 时间与成本预算（4090）

| 步骤 | 时间 | 费用 |
|---|---|---|
| 路线 B：装环境 + 出第一个效果视频 | ~30 分钟（下载为主） | ~¥1 |
| 路线 A：装环境 + 数据 | 30–40 分钟 | ~¥1.5 |
| 路线 A：冒烟测试 | 15 分钟 | ~¥0.5 |
| 路线 A：LAFAN1 正式训练（锚点口径） | 4–8 小时 | ¥8–16 |
| 路线 A：评测 + 渲染视频 | ~20 分钟 | ~¥0.7 |
| 可选：BONES-SEED 全量 | 40–50 小时 | ¥80–100 |
| 数据盘 150–200GB | 按月计 | ~¥15–30/月 |

> 4090 按量约 ¥2/小时（随地区浮动）。无卡模式不计 GPU 费，装环境、写代码阶段可用。

---

## 7. 指标速查

| 指标 | 含义 | 参考值 |
|---|---|---|
| train_return | 训练中归一化回报 | 到 0.7+ 就算跟踪上了 |
| benchmark 归一化回报 | Σ每步奖励/(0.1×步数)，1.0=完美 | 论文 3h >0.90 (BONES-SEED) |
| failed / 摔倒 | 触发终止条件（倒/根高/倾斜超限）的 clip | 越少越好 |
| WM 8 步误差 | 世界模型 8 步开环归一化预测误差 | 论文 0.007–0.011 |
| 关节 MAE（路线 B） | 29 关节角平均绝对误差 | ~4.4°（训练集） |

## 8. 参考链接

- 论文：[arXiv:2610.28480](https://arxiv.org/abs/2610.28480)；项目页（含浏览器 demo）：[supermimic.github.io](https://supermimic.github.io/)
- 路线 A 代码：[CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)（含论文规格逐条转录 `docs/PAPER_SPEC.md`、实验日志 `docs/EXPERIMENTS.md`）
- 路线 B 代码：[panhainan2020/SuperMimic-MuJoCo](https://github.com/panhainan2020/SuperMimic-MuJoCo)（自带权重 + [在线评估结果页](https://supermimic-mujoco-results.zippyaspen7.chatgpt.site)）
- 数据：LAFAN1 重定向集 [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)（公开）；BONES-SEED [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed)（gated，需同意条款）

# 环境配置记录（AutoDL 单卡 4090）

训练在 AutoDL 的 RTX 4090（24 GB）实例上完成，镜像为 Ubuntu 22.04 + Python 3.12，实例自带驱动 580.76.05 与 570.124.04。

## 驱动与 CUDA 版本

基础实现默认使用 `torch 2.14.1 + cu130`，要求驱动不低于 580。若实例驱动为 570，需要改用 cu126 构建的 torch，并把 `warp-lang` 降到 1.15（Warp 1.18 依赖 CUDA 13）。两者都通过 `uv` 管理：

```bash
# 驱动 >= 580（默认）
uv sync --group dev

# 驱动 560-579：torch 换 cu126，warp 降到 1.15
uv pip install "warp-lang==1.15.0"
uv pip install --reinstall torch==2.14.1 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

cu126 的 torch 不含 Blackwell（sm_120）内核，在 5090 上会报 `no kernel image is available`；4090（sm_89）不受影响。

## 依赖安装

`uv sync` 会从 GitHub 拉取 mjlab（固定在 commit `8bb2bc9`）。国内直连 GitHub 常失败，可先克隆到本地再改为 path 源：

```bash
git clone https://ghproxy.net/https://github.com/mujocolab/mjlab.git mjlab-src
cd mjlab-src && git fetch --depth 1 origin 8bb2bc9e187797689ec82c78c35019afd2fd0376 && git checkout FETCH_HEAD
# pyproject.toml: mjlab = { path = "mjlab-src" }
```

PyPI 与 PyTorch 索引在部分节点上较慢。实测清华 PyPI 单流约 1.8 MB/s，阿里云 PyPI 与阿里云 pytorch-wheels 约 9.5 MB/s。注意阿里云的 `pytorch-wheels` 不支持 uv 所需的 PEP 503 子路径（`/cu130/torch/` 返回 404），只能作为 pip 的 find-links 或直接下载 wheel 安装。

uv 默认会对所有平台求解依赖。本项目只在 Linux 上训练，可在 `pyproject.toml` 中限制环境，避免解析 Windows 分支时报错：

```toml
[tool.uv]
environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]
```

## 数据

LAFAN1 的 G1 重定向版本来自 HuggingFace 的 `lvhaidong/LAFAN1_Retargeting_Dataset`，国内可设置 `HF_ENDPOINT=https://hf-mirror.com`。40 条 CSV（30 fps）转换为 50 Hz npz 约需 10 秒。

## 离屏渲染

MuJoCo 的离屏渲染依赖 EGL 与 DRM。部分云容器只映射 `/dev/nvidia*` 与 `/dev/nvidia-uvm`，没有 `/dev/dri` 与 `/dev/nvidia-caps`，此时 `mujoco.Renderer` 会报 `Cannot initialize a EGL device display`；`MUJOCO_GL=osmesa` 与 Xvfb + GLFW 也无法获得可用的 GL 上下文（容器 cgroup 拒绝 DRM 设备）。

解决办法是把仿真与渲染分离：在 GPU 实例上导出每帧状态与逐条评测，再在任意有 OpenGL 的机器上渲染。`scripts/dump_rollouts.py` 与 `scripts/render_rollouts_mac.py` 即为此流程，渲染端只需要 MuJoCo 与 imageio。

## 训练耗时

4090 上 2048 环境、批 2048 时约为 2.0–2.4 it/s，其中约 5 分钟用于首次 `torch.compile` 与 MuJoCo Warp 内核编译。4.5 小时的训练上限对应约 45,000 次迭代。

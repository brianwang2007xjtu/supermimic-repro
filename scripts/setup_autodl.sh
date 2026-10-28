#!/usr/bin/env bash
# ============================================================
# SuperMimic (CuboiLeo/SuperMimic-Reproduced) AutoDL 4090 一键环境安装
#
# 用法: 在仓库根目录执行  bash setup_autodl.sh
# 作用:
#   1) 安装 uv (若没有)
#   2) 读取 nvidia-smi 驱动版本, 自动决定 torch 的 CUDA 变体:
#        驱动 >= 580  -> cu130 (仓库默认, CUDA 13.0)
#        560 <= 驱动 < 580 -> cu126 (CUDA 12.6), 同时把 warp-lang 降到 1.15.0
#        驱动 < 560  -> 报错并给出指引 (请改用 panhainan 仓库方案 B, 见教程)
#   3) 把 pyproject.toml 里的 torch/warp 源改到对应版本
#   4) uv sync 安装全部依赖 (自动下载 Python 3.12, 无需系统 Python)
# ============================================================
set -euo pipefail

# ---------- 0. 安装 uv ----------
if ! command -v uv >/dev/null 2>&1; then
  echo "[1/6] installing uv ..."
  if command -v pip >/dev/null 2>&1; then
    pip install -q uv
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
export PATH="$HOME/.local/bin:$PATH"
echo "[1/6] uv: $(uv --version)"

# ---------- 1. 检测驱动 ----------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "!! nvidia-smi 不存在: 请确认已开机并租用了 GPU 实例 (无卡模式看不到 GPU)。" >&2
  exit 1
fi
DRV_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
echo "[2/6] NVIDIA 驱动主版本 = ${DRV_MAJOR}"

# ---------- 2. 决定 CUDA 变体 ----------
if   [ "$DRV_MAJOR" -ge 580 ]; then CUDA_VARIANT="cu130"; WARP_PIN=""
elif [ "$DRV_MAJOR" -ge 560 ]; then CUDA_VARIANT="cu126"; WARP_PIN="warp-lang==1.15.0"
else
  echo "!! 驱动 ${DRV_MAJOR} < 560, 无法安装 torch 2.14 (需要 CUDA 12.6+)。" >&2
  echo "   请改走方案 B (panhainan2020/SuperMimic-MuJoCo, torch 2.7.1), 见教程第 5 节。" >&2
  exit 1
fi
echo "[3/6] 使用 torch +${CUDA_VARIANT}"

# ---------- 3. 改写 pyproject.toml ----------
python3 - "$CUDA_VARIANT" "$WARP_PIN" <<'PYEOF'
import pathlib, sys
variant, warp_pin = sys.argv[1], sys.argv[2]
p = pathlib.Path("pyproject.toml")
s = p.read_text()
s = s.replace('index = "pytorch-cu130"', f'index = "pytorch-{variant}"')
s = s.replace('url = "https://download.pytorch.org/whl/cu130"',
              f'url = "https://download.pytorch.org/whl/{variant}"')
if warp_pin:
    s = s.replace('"warp-lang>=1.15.0"', f'"{warp_pin}"')
p.write_text(s)
print(f"    pyproject.toml 已改写 -> {variant}" + (f", {warp_pin}" if warp_pin else ""))
PYEOF

# ---------- 4. 处理 mjlab git 依赖 (国内直连 GitHub 经常失败, 自动换镜像) ----------
MJLAB_REV="8bb2bc9e187797689ec82c78c35019afd2fd0376"
git config --global http.version HTTP/1.1 2>/dev/null || true

have_mjlab=0
if [ -d mjlab-src/src/mjlab ]; then
  have_mjlab=1
  echo "[4/6] 发现本地 mjlab-src, 直接使用 (跳过 GitHub)"
else
  for url in \
    "https://ghproxy.net/https://github.com/mujocolab/mjlab.git" \
    "https://ghfast.top/https://github.com/mujocolab/mjlab.git" \
    "https://gitclone.com/github.com/mujocolab/mjlab.git" \
    "https://github.com/mujocolab/mjlab.git"; do
    echo "[4/6] 尝试克隆 mjlab: $url"
    if git clone --depth 1 "$url" mjlab-src 2>/dev/null \
       && (cd mjlab-src && git fetch --depth 1 origin "$MJLAB_REV" 2>/dev/null && git checkout -q FETCH_HEAD); then
      have_mjlab=1
      break
    fi
    rm -rf mjlab-src
  done
fi

if [ "$have_mjlab" = "1" ]; then
  python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("pyproject.toml")
s = p.read_text()
s = s.replace('mjlab = { git = "https://github.com/mujocolab/mjlab", rev = "8bb2bc9e187797689ec82c78c35019afd2fd0376" }',
              'mjlab = { path = "mjlab-src" }')
p.write_text(s)
print("    pyproject.toml: mjlab 改为本地 path 源")
PYEOF
else
  echo "!! 无法从 GitHub/镜像获取 mjlab。可从本地电脑打包上传 mjlab 源码到仓库根目录的 mjlab-src/ 后重跑本脚本。" >&2
  exit 1
fi

# ---------- 5. uv sync (首次约 10~20 分钟, 取决于网络) ----------
# uv 需要 Python 3.12: 系统有就直接用, 没有则走 ghproxy 镜像下载 (直连 GitHub 会超时)
if command -v python3.12 >/dev/null 2>&1; then
  echo "[5/6] 系统已有 python3.12, uv 直接使用"
else
  export UV_PYTHON_INSTALL_MIRROR="https://ghproxy.net/https://github.com/astral-sh/python-build-standalone/releases/download"
  echo "[5/6] 未发现系统 python3.12, uv 将经 ghproxy 镜像下载"
fi
echo "[5/6] uv sync ..."
if ! uv sync --group dev; then
  echo "!! uv sync 失败。常见原因与处理:" >&2
  echo "   - uv 下载 Python 3.12 失败(GitHub releases 被墙): 用 conda 建环境兜底:" >&2
  echo "       conda create -n sm312 python=3.12 -y && conda activate sm312" >&2
  echo "       uv venv --python \$(which python3.12) && uv sync --group dev" >&2
  echo "   - torch/warp 索引下载慢: 重跑本脚本, 或按教程 FAQ 换源。" >&2
  exit 1
fi

# ---------- 6. 冒烟验证 ----------
echo "[6/6] 验证 CUDA / Warp / MuJoCo ..."
env -u PYTHONPATH uv run python - <<'PYEOF'
import torch, mujoco, warp as wp
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("mujoco:", mujoco.__version__)
wp.init()
print("warp:", wp.config.version, "| device:", wp.get_device())
PYEOF

echo
echo "全部完成! 下一步: bash download_lafan1.sh 下载数据, 然后按教程开始训练。"

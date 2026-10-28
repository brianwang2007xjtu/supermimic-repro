#!/usr/bin/env bash
# ============================================================
# SuperMimic AutoDL 4090 一体化安装: 依赖 + 数据
# 幂等可重跑: 断线后重跑同一命令即可, 已下载内容全部续传
# 日志: 同目录 setup.log
# ============================================================
set -euo pipefail
cd /root/autodl-tmp/SuperMimic-Reproduced
export PATH="$HOME/.local/bin:/usr/local/bin:$HOME/miniconda3/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------- 1. uv ----------
if ! command -v uv >/dev/null 2>&1; then
  log "安装 uv ..."
  command -v pip >/dev/null 2>&1 && pip install -q uv
fi
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
log "uv: $(uv --version)"

# ---------- 2. 驱动检查 + 索引自愈 (torch 走默认 PyPI 镜像, 移除不可用的自定义索引) ----------
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$DRV" -ge 560 ] || { log "!! 驱动 $DRV < 560, 无法安装 torch 2.14"; exit 1; }
log "驱动 $DRV OK -> torch 2.14 走清华 PyPI 镜像"
python3 - <<'PYEOF'
import pathlib, re
p = pathlib.Path("pyproject.toml"); s = p.read_text()
# 移除 torch/warp 的自定义索引绑定与索引块 (阿里云 pytorch-wheels 不支持 uv 的 PEP503 子路径)
s = re.sub(r'^torch = \{ index = "pytorch-cu130".*\n', '', s, flags=re.M)
s = re.sub(r'^warp-lang = \{ index = "nvidia".*\n', '', s, flags=re.M)
s = re.sub(r'\n?\[\[tool\.uv\.index\]\]\nname = "pytorch-cu130"\nurl = "[^"]+"\nexplicit = true\n', '', s)
s = re.sub(r'\n?\[\[tool\.uv\.index\]\]\nname = "nvidia"\nurl = "[^"]+"\nexplicit = true\n', '', s)
if "\n[tool.uv]\n" not in s:
    s += "\n[tool.uv]\nenvironments = [\"sys_platform == 'linux' and platform_machine == 'x86_64'\"]\n"
p.write_text(s)
print("pyproject ok (indexes self-healed)")
PYEOF

# ---------- 3. mjlab 本地源 (镜像链) ----------
MJLAB_REV=8bb2bc9e187797689ec82c78c35019afd2fd0376
if [ ! -d mjlab-src/src/mjlab ]; then
  log "克隆 mjlab ..."
  ok=0
  for url in \
    "https://ghproxy.net/https://github.com/mujocolab/mjlab.git" \
    "https://ghfast.top/https://github.com/mujocolab/mjlab.git" \
    "https://gitclone.com/github.com/mujocolab/mjlab.git" \
    "https://github.com/mujocolab/mjlab.git"; do
    log "尝试: $url"
    if git clone --depth 1 "$url" mjlab-src 2>/dev/null \
       && (cd mjlab-src && git fetch --depth 1 origin "$MJLAB_REV" 2>/dev/null && git checkout -q FETCH_HEAD); then
      ok=1; break
    fi
    rm -rf mjlab-src
  done
  [ "$ok" = "1" ] || { log "!! mjlab 获取失败"; exit 1; }
fi
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("pyproject.toml"); s = p.read_text()
s = s.replace('mjlab = { git = "https://github.com/mujocolab/mjlab", rev = "8bb2bc9e187797689ec82c78c35019afd2fd0376" }',
              'mjlab = { path = "mjlab-src" }')
p.write_text(s)
print("mjlab -> path 源")
PYEOF

# ---------- 4. 依赖 ----------
log "uv sync (缓存续传, 首次 5~15 分钟) ..."
uv sync --group dev

# ---------- 5. 验证 ----------
log "验证 CUDA / Warp / MuJoCo ..."
env -u PYTHONPATH uv run python - <<'PYEOF'
import torch, mujoco, warp as wp
print("torch:", torch.__version__, "| cuda:", torch.version.cuda,
      "| ok:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("mujoco:", mujoco.__version__)
wp.init()
print("warp:", wp.config.version, "| device:", wp.get_device())
PYEOF

# ---------- 6. 数据 ----------
log "下载 LAFAN1 (hf-mirror) ..."
mkdir -p data/raw/lafan1
env -u PYTHONPATH uv run python - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    "lvhaidong/LAFAN1_Retargeting_Dataset",
    repo_type="dataset",
    allow_patterns="g1/*.csv",
    local_dir="data/raw/lafan1",
)
PYEOF
log "CSV 数量: $(ls data/raw/lafan1/g1/*.csv | wc -l)"
log "转换 50Hz npz ..."
env -u PYTHONPATH uv run python scripts/data/csv_to_npz_batch.py \
  --input-dir data/raw/lafan1/g1 --output-dir data/npz/lafan1 \
  --input-fps 30 --workers 16
log "npz 数量: $(ls data/npz/lafan1/*.npz | wc -l)"

log "=== 环境与数据全部就绪, 可以开始训练 ==="

#!/usr/bin/env bash
# ============================================================
# LAFAN1 数据下载 + 50Hz npz 转换 (SuperMimic-Reproduced 仓库内执行)
#
# 数据集: lvhaidong/LAFAN1_Retargeting_Dataset (公开, 无需 token)
#   40 条 G1 重定向 CSV (30 fps, 共约 2.45 小时) —— 论文 Fig. 4a 所用数据
# 通过 hf-mirror.com 镜像下载, 国内速度稳定。
# ============================================================
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com

echo "[1/3] 下载 LAFAN1 G1 CSV (约几十 MB) ..."
# huggingface-hub 已是仓库依赖, 用 uv run 调用项目环境里的 CLI
mkdir -p data/raw/lafan1
env -u PYTHONPATH uv run huggingface-cli download lvhaidong/LAFAN1_Retargeting_Dataset \
  --include "g1/*.csv" \
  --local-dir data/raw/lafan1
ls data/raw/lafan1/g1/ | head
echo "CSV 数量: $(ls data/raw/lafan1/g1/*.csv | wc -l)"

echo "[2/3] 转换为 50Hz npz (论文口径) ..."
env -u PYTHONPATH uv run python scripts/data/csv_to_npz_batch.py \
  --input-dir data/raw/lafan1/g1 \
  --output-dir data/npz/lafan1 \
  --input-fps 30 --workers 16

echo "[3/3] 完成:"
ls data/npz/lafan1 | head
echo "npz 数量: $(ls data/npz/lafan1/*.npz | wc -l)"

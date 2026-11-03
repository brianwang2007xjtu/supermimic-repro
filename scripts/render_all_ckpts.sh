#!/usr/bin/env bash
# 训练结束后执行: 为每个检查点渲染同一条 dance1 视频 + 4 条动作子集评测
# 输出: videos/ckpt_progress/<ckpt>.mp4 与 progress.csv 对照表
set -euo pipefail
cd /root/autodl-tmp/SuperMimic-Reproduced
export PATH="$HOME/.local/bin:/usr/local/bin:$HOME/miniconda3/bin:$PATH"
mkdir -p videos/ckpt_progress
echo "ckpt,subset_return,failed" > videos/ckpt_progress/progress.csv

CKPTS="$(ls logs/lafan1_anchor/ckpt_*.pt 2>/dev/null | grep -v latest | sort -t_ -k2 -n) logs/lafan1_anchor/ckpt_latest.pt"

for ckpt in $CKPTS; do
  name=$(basename "$ckpt" .pt)
  echo "===== $name ====="
  env -u PYTHONPATH uv run python scripts/record_video.py \
    --ckpt "$ckpt" --npz-dir data/npz/lafan1 \
    --clips dance1_subject1 --out "videos/ckpt_progress/${name}.mp4" --seconds 20 \
    || { echo "$name 渲染失败" >&2; continue; }
  line=$(env -u PYTHONPATH uv run python scripts/eval_benchmark.py \
    --ckpt "$ckpt" --npz-dir data/npz/lafan1 \
    --clips dance1_subject1 walk1_subject1 jumps1_subject1 fallAndGetUp1_subject1 \
    --num-envs 512 --out "videos/ckpt_progress/${name}_bench.json" 2>/dev/null \
    | grep -E "clips \|" || true)
  ret=$(echo "$line" | sed -E 's/.*mean norm return ([0-9.]+).*/\1/')
  failed=$(echo "$line" | sed -E 's/.*failed ([0-9]+).*/\1/')
  echo "${name},${ret:-NA},${failed:-NA}" >> videos/ckpt_progress/progress.csv
done

echo "===== 汇总 ====="
cat videos/ckpt_progress/progress.csv

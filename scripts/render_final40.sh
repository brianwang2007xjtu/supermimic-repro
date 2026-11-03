#!/usr/bin/env bash
# 最终检查点(44k)渲染全部 40 个动作, 每个 20 秒, 输出 videos/final40/
# 幂等: 已存在的视频自动跳过
set -euo pipefail
cd /root/autodl-tmp/SuperMimic-Reproduced
export PATH="/root/miniconda3/bin:$PATH"
mkdir -p videos/final40

for f in data/npz/lafan1/*.npz; do
  name=$(basename "$f" .npz)
  out="videos/final40/${name}.mp4"
  if [ -s "$out" ]; then echo "SKIP $name"; continue; fi
  echo "===== $name ====="
  .venv/bin/python scripts/record_video.py \
    --ckpt logs/lafan1_anchor/ckpt_0044000.pt --npz-dir data/npz/lafan1 \
    --clips "$name" --out "$out" --seconds 20 || echo "FAIL $name"
done
echo "===== 完成: $(ls videos/final40/*.mp4 2>/dev/null | wc -l)/40 ====="

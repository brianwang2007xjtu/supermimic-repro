#!/usr/bin/env bash
# 每个检查点在固定 4 条动作上的 benchmark (回报 vs 迭代 曲线数据)
# 输出: bench_by_ckpt.csv
set -euo pipefail
cd /root/autodl-tmp/SuperMimic-Reproduced
export PATH="/root/miniconda3/bin:$PATH"
echo "ckpt,mean_norm_return,step_weighted,failed" > bench_by_ckpt.csv

for ckpt in $(ls logs/lafan1_anchor/ckpt_*.pt | grep -v latest | sort -t_ -k2 -n); do
  name=$(basename "$ckpt" .pt)
  echo "===== $name ====="
  line=$(.venv/bin/python scripts/eval_benchmark.py \
    --ckpt "$ckpt" --npz-dir data/npz/lafan1 \
    --clips dance1_subject1 walk1_subject1 jumps1_subject1 fallAndGetUp1_subject1 \
    --num-envs 512 2>/dev/null | grep -E "clips \|" || true)
  ret=$(echo "$line" | sed -E 's/.*mean norm return ([0-9.]+).*/\1/')
  sw=$(echo "$line" | sed -E 's/.*step-weighted ([0-9.]+).*/\1/')
  failed=$(echo "$line" | sed -E 's/.*failed ([0-9]+).*/\1/')
  echo "${name},${ret:-NA},${sw:-NA},${failed:-NA}" >> bench_by_ckpt.csv
done
cat bench_by_ckpt.csv

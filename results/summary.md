# SuperMimic 复现 · 图表数据包

复现对象：*SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*（Liao et al., CoRL 2026, arXiv:2610.28480）
代码：[CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)（论文口径从零复现）
数据：LAFAN1 40 条动作（2.45 h）；训练设备：RTX 4090 24GB；最终检查点 44,000 迭代。

## 文件说明

| 文件 | 内容 | 适合画的图 |
|---|---|---|
| `training_series.csv` | 逐 50 迭代训练指标（968 行）：iteration/hours/it_per_s/updates/train_ret/wm_nse1/wm_nse8/imag_ret/imag_surv/t_sim/t_wm/t_policy | 训练回报曲线、世界模型 1/8 步误差曲线、吞吐与耗时分解 |
| `tb_scalars.csv` | TensorBoard 全量标量导出（37 个 tag，13,801 点，含 `wm/*`、`policy/*` 等） | 损失/梯度/想象指标等细节曲线 |
| `bench_by_ckpt.csv` | 每个检查点在固定 4 条动作上的 benchmark（11 个点） | **回报 vs 训练迭代**（核心 training curve） |
| `per_clip_eval.csv` | 最终 44k 检查点在全部 40 条动作上的整段评测 | 每动作回报分布（柱状图/散点） |

## 关键数字（44k 检查点）

- 训练回报：**0.752**（LAFAN1, 4.3 h, 2.0 it/s）
- 世界模型 8 步开环误差：**0.0096**（论文 0.007–0.011）
- 世界模型 1 步误差：**0.0088**
- 动作子集 benchmark：**0.8856**（步加权 0.8942，4 条动作，1/4 失败）
- 全量 40 动作评测：**平均归一化回报 0.8815**，最高 0.969，最低 0.661，失败 9/40
- 训练曲线（4 动作子集）：4k 0.449 → 8k 0.590 → 12k 0.650 → 16k 0.712 → 20k 0.803 → 24k 0.769 → 28k 0.815 → 32k 0.844 → 36k 0.864 → 40k 0.867 → **44k 0.886**
- 吞吐：训练稳态 2.0–2.4 it/s（RTX 4090, 2048 envs, batch 2048）；渲染 40 条 20s 视频共约 15 分钟（CPU 离屏）

## 建议的论文式配图

1. **Fig. A — 训练回报 & 世界模型误差**（双 y 轴）：`training_series.csv` 的 `train_ret` 与 `wm_nse8`；
2. **Fig. B — 回报 vs 迭代**：`bench_by_ckpt.csv`（最能体现"小算力快速收敛"）；
3. **Fig. C — 每动作回报分布**：`per_clip_eval.csv` 按动作类型分组画箱线图；
4. **Fig. D — 耗时分解**：`t_sim/t_wm/t_policy` 堆叠面积图；
5. **Fig. E — 想象 rollouts 指标**：`imag_ret` 与 `imag_surv`。

> 注：训练在 4.3 小时处被实例关机中断（计划 4.5 h），最后保存的检查点为 44k；曲线在 24k 处的小回落来自该阶段策略从"活得久"向"跟得准"的过渡，属复现仓库文档记录过的正常现象。

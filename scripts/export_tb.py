"""Export all TensorBoard scalars of the training run to CSV (for blog charts)."""
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator("logs/lafan1_anchor/tb", size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags().get("scalars", [])
with open("tb_scalars.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tag", "step", "wall_time", "value"])
    n = 0
    for tag in tags:
        for ev in ea.Scalars(tag):
            w.writerow([tag, ev.step, ev.wall_time, ev.value])
            n += 1
print(f"{len(tags)} tags, {n} points -> tb_scalars.csv")
for t in tags:
    print(" -", t)

"""Build manifest.json for the blog demo from rollout npz files."""
import json
import sys
from pathlib import Path

import numpy as np

rollouts = Path(sys.argv[1] if len(sys.argv) > 1 else "rollouts")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "manifest.json")

GROUPS = ["walk", "dance", "run", "sprint", "fight", "jumps", "fallAndGetUp"]


def group_of(name: str) -> str:
    n = name.lower()
    for g in GROUPS:
        if n.startswith(g.lower()):
            return g
    return "other"


clips = []
for f in sorted(rollouts.glob("*.npz")):
    d = np.load(f)
    clips.append({
        "name": f.stem,
        "group": group_of(f.stem),
        "src": f"{f.stem}.mp4",
        "duration": round(int(d["steps"]) / 50.0, 1),
        "norm_return": round(float(d["norm_return"]), 3),
        "failed": bool(d["failed"]),
    })

clips.sort(key=lambda c: (GROUPS.index(c["group"]) if c["group"] in GROUPS else 99, c["name"]))
manifest = {"checkpoint": "44000", "fps": 50, "clips": clips}
out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
ret = [c["norm_return"] for c in clips]
print(f"{len(clips)} clips -> {out} | mean return {sum(ret)/len(ret):.4f} | failed {sum(c['failed'] for c in clips)}/{len(clips)}")

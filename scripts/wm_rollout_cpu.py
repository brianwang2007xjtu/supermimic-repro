"""CPU-only world-model imagination rollout (for the blog trail video).

Loads the trained checkpoint, seeds the world model from the simulator rollout
of one clip, and rolls the policy + world model forward autonomously (no MuJoCo
in the loop). Saves predicted MuJoCo-layout qpos/qvel sequences.

Reference for the policy's goal window is taken from the simulator rollout
itself (the policy tracks it closely), so this visualises what the world model
imagines from a real state under the same motion.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path("/Users/master/Desktop/supermimic-repro/SuperMimic-Reproduced")
sys.path.insert(0, str(REPO / "src"))


# ---------------------------------------------------------------- mjlab shim
def _quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _quat_inv(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1) / (q * q).sum(-1, keepdim=True).clamp_min(1e-9)


def _quat_apply(q, v):
    qv, w = q[..., 1:], q[..., :1]
    t = torch.cross(qv, v, dim=-1) * 2
    return v + w * t + torch.cross(qv, t, dim=-1)


def _quat_box_minus(a, b):
    d = _quat_mul(a, _quat_inv(b))
    w = d[..., :1].clamp(-1.0, 1.0)
    v = d[..., 1:]
    ang = 2 * torch.atan2(v.norm(dim=-1, keepdim=True), w)
    n = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return v / n * ang


def _yaw_quat(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    out = torch.zeros_like(q)
    out[..., 0] = torch.cos(yaw / 2)
    out[..., 3] = torch.sin(yaw / 2)
    return out



def _matrix_from_quat(q):
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ], dim=-1)
    return m.reshape(*q.shape[:-1], 3, 3)


def _quat_apply_inverse(q, v):
    return _quat_apply(_quat_conjugate(q), v)


def _quat_conjugate(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_error_magnitude(a, b):
    d = (a * b).sum(-1).abs().clamp(max=1.0)
    return 2 * torch.acos(d)


def _quat_from_angle_axis(angle, axis):
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    half = angle / 2
    return torch.cat([torch.cos(half).unsqueeze(-1), axis * torch.sin(half).unsqueeze(-1)], dim=-1)

_math = types.ModuleType("mjlab.utils.lab_api.math")
_math.quat_apply = _quat_apply
_math.matrix_from_quat = _matrix_from_quat
_math.quat_apply_inverse = _quat_apply_inverse
_math.quat_conjugate = _quat_conjugate
_math.quat_error_magnitude = _quat_error_magnitude
_math.quat_from_angle_axis = _quat_from_angle_axis
_math.quat_box_minus = _quat_box_minus
_math.quat_inv = _quat_inv
_math.quat_mul = _quat_mul
_math.yaw_quat = _yaw_quat
for name, mod in [("mjlab", types.ModuleType("mjlab")), ("mjlab.utils", types.ModuleType("mjlab.utils")),
                  ("mjlab.utils.lab_api", types.ModuleType("mjlab.utils.lab_api")),
                  ("mjlab.utils.lab_api.math", _math)]:
    sys.modules[name] = mod

# The env package __init__ imports mjlab; skip it by registering the packages
# with their real __path__ so submodules load directly (rewards.py only needs
# the mjlab math shim above).
import types as _t  # noqa: E402
for _name, _sub in [("supermimic.envs", "supermimic/envs"), ("supermimic.envs.mdp", "supermimic/envs/mdp")]:
    if _name not in sys.modules:
        _m = _t.ModuleType(_name)
        _m.__path__ = [str(REPO / "src" / _sub)]
        sys.modules[_name] = _m

from supermimic.kinematics.rotations import (  # noqa: E402
    projected_gravity,
    quat_apply_inverse,
    yaw_quat,
)


def goal_features(ref, default_joint_pos):
    """Copy of motion_library.goal_features (68-d per-frame goal features)."""
    q = ref.root_quat_w
    grav = projected_gravity(q)
    ang_b = quat_apply_inverse(q, ref.root_ang_vel_w)
    lin_h = quat_apply_inverse(yaw_quat(q), ref.root_lin_vel_w)
    height = ref.root_pos_w[..., 2:3]
    return torch.cat([ref.joint_pos - default_joint_pos, ref.joint_vel, grav, ang_b, lin_h, height], dim=-1)

from supermimic.policy.nets import Policy, PolicyCfg  # noqa: E402
from supermimic.world_model.heads import (  # noqa: E402
    analytic_observation,
    anchor_relative_obs,
    flatten_history_term_major,
)
from supermimic.world_model.world_model import WorldModel, WorldModelCfg  # noqa: E402

TRACKED = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link", "left_shoulder_roll_link", "left_elbow_link",
    "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link",
    "right_wrist_yaw_link",
)
ANCHOR = "torso_link"
JOINTS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]


def default_joint_pos() -> torch.Tensor:
    q0 = np.zeros(29, dtype=np.float32)
    for name, val in {
        "hip_pitch": -0.312, "knee": 0.669, "ankle_pitch": -0.363, "elbow": 0.6,
        "left_shoulder_roll": 0.2, "left_shoulder_pitch": 0.2,
        "right_shoulder_roll": -0.2, "right_shoulder_pitch": 0.2,
    }.items():
        for i, j in enumerate(JOINTS):
            if j == name or j.endswith(name):
                q0[i] = val
    return torch.from_numpy(q0)


class AnchorFK:
    """torso_link world pose from a MuJoCo-layout qpos (CPU mujoco)."""

    def __init__(self, xml_path: Path):
        import mujoco
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ANCHOR)

    def __call__(self, qpos: np.ndarray):
        self.data.qpos[:] = qpos
        self._mujoco.mj_forward(self.model, self.data)
        return (torch.from_numpy(self.data.xpos[self.body_id].copy()).float(),
                torch.from_numpy(self.data.xquat[self.body_id].copy()).float())


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else "run2_subject4"
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else f"wm_{clip}.npz")
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    t0 = int(sys.argv[4]) if len(sys.argv) > 4 else 200

    torch.manual_seed(0)
    device = "cpu"

    ckpt_path = REPO.parent / "checkpoints_backup/ckpt_0044000.pt"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]

    from supermimic.world_model.state_spec import StateSpecCfg

    wm_raw = dict(cfg["world_model"])
    wm_raw["state_spec"] = StateSpecCfg(**wm_raw["state_spec"])
    wm_cfg = WorldModelCfg(**wm_raw)
    pcfg = PolicyCfg(**cfg["policy"])

    class DummyHeads:
        anchor_index = TRACKED.index(ANCHOR)
        default_joint_pos = None

    heads = DummyHeads()
    heads.default_joint_pos = default_joint_pos()
    wm = WorldModel(wm_cfg, heads, device)
    wm.load_state_dict(ck["wm"])
    wm.eval()
    policy = Policy(pcfg)
    policy.load_state_dict(ck["policy"])
    policy.eval()

    q0 = heads.default_joint_pos
    xi = torch.zeros(1, 33)
    xi[0, 0] = 1.0  # nominal foot friction 1.0, zero encoder bias / COM offset

    d = np.load(Path("rollouts") / f"{clip}.npz")
    sim_qpos, sim_qvel = d["qpos"], d["qvel"]
    n_frames = min(horizon + t0, len(sim_qpos) - 11)

    fk = AnchorFK(Path("g1/xmls/g1.xml"))

    # ---- proxy reference features from the simulator trajectory
    def ref_state(qpos_row, qvel_row):
        root_q = torch.from_numpy(qpos_row[3:7]).float()
        ang_w = _quat_apply(root_q, torch.from_numpy(qvel_row[3:6]).float())
        return types.SimpleNamespace(
            root_pos_w=torch.from_numpy(qpos_row[0:3]).float(),
            root_quat_w=root_q,
            joint_pos=torch.from_numpy(qpos_row[7:36]).float(),
            root_lin_vel_w=torch.from_numpy(qvel_row[0:3]).float(),
            root_ang_vel_w=ang_w,
            joint_vel=torch.from_numpy(qvel_row[6:35]).float(),
        )

    goals_feat = []
    for t in range(len(sim_qpos)):
        r = ref_state(sim_qpos[t], sim_qvel[t])
        goals_feat.append(goal_features(r, q0).numpy())
    goals_feat = np.stack(goals_feat)  # (T, 68)
    ref_anchor = [fk(sim_qpos[t]) for t in range(len(sim_qpos))]

    def goal_at(t, robot_anchor):
        idx = [min(t + k, len(sim_qpos) - 1) for k in range(10)]
        g = torch.from_numpy(goals_feat[idx].reshape(1, -1)).float()
        rel = anchor_relative_obs(robot_anchor[0][None], robot_anchor[1][None],
                                  ref_anchor[t][0][None], ref_anchor[t][1][None])
        return torch.cat([g, rel], dim=-1)

    # ---- seed state + observation history from the simulator
    s = torch.from_numpy(np.concatenate([sim_qpos[t0], sim_qvel[t0]]))[None].float()
    hist = []
    for t in range(t0 - 9, t0 + 1):
        st = torch.from_numpy(np.concatenate([sim_qpos[t], sim_qvel[t]]))[None].float()
        hist.append(analytic_observation(st, torch.zeros(1, 29), xi, q0))
    hist = torch.stack(hist)  # (10,1,93)

    pred_qpos = [s[0, :36].numpy().copy()]
    a_prev = torch.zeros(1, 29)
    for step in range(horizon):
        t = t0 + step
        anchor = fk(s[0, :36].numpy())
        goal = goal_at(t, anchor)
        obs_hist = flatten_history_term_major(hist)
        with torch.no_grad():
            a = policy(obs_hist, goal)
            s_next = wm.step(s, a, xi)
        obs = analytic_observation(s_next, a, xi, q0)
        hist = torch.cat([hist[1:], obs[None]], dim=0)
        s = s_next
        pred_qpos.append(s[0, :36].numpy().copy())
        a_prev = a
        if step % 100 == 0:
            print(f"step {step}/{horizon}", flush=True)

    pred_qpos = np.stack(pred_qpos).astype(np.float32)
    np.savez_compressed(out_path, qpos=pred_qpos, t0=t0, clip=clip)
    joint_err = np.abs(pred_qpos[:, 7:36] - sim_qpos[t0:t0 + len(pred_qpos), 7:36]).mean()
    root_err = np.abs(pred_qpos[:, 0:3] - sim_qpos[t0:t0 + len(pred_qpos), 0:3]).mean()
    print(f"saved {out_path}: {len(pred_qpos)} frames | mean joint err {joint_err:.3f} rad | root err {root_err:.3f} m")


if __name__ == "__main__":
    main()

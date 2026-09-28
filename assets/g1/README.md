# assets/g1 — IK kinematic model

This directory is the **runtime Pinocchio/CasADi IK source** for G1-D Dex1 teleop
and policy handoff (`teleop/robot_control/robot_arm_ik.py`).

| File | Role |
|---|---|
| `g1_body29_hand14.urdf` | Dual-arm (+ hand links) URDF used by `G1_29_ArmIK` |
| `g1_29_model_cache.pkl` | Prebuilt Pinocchio reduced model cache |
| `meshes/` | Mesh assets referenced by the URDF |

Product name **G1-D** still uses this 29-DoF arm chain for teleop IK.
See `assets/g1_D/` for a machine-reference URDF that is **not** loaded by runtime.

`g1_body29_hand14` is modified from
[g1_29dof_with_hand_rev_1_0](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf).

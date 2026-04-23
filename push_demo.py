"""
push_demo.py — SO-ARM100 推方块 Demo

轨迹:
    Home(workspace 中心, z=0.1) → Descend(z=PUSH_Z) → BackPos(方块背后) → PushEnd(目标+超调) → Home

运行: mjpython push_demo_single.py
依赖: pip install mujoco opencv-python numpy
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mujoco
import mujoco.viewer
import numpy as np

# ─────────────────────── 场景配置 ───────────────────────

XML_PATH        = "./push_scene.xml"
FIXED_TIP_GEOM  = "fixed_jaw_pad_1"
MOVING_TIP_GEOM = "moving_jaw_pad_1"
CUBE_BODY       = "cube"
CUBE_JOINT      = "cube_free"
TARGET_BODY     = "target"
WRIST_CAM       = "wrist_cam"

# workspace 黄框(与 push_scene.xml 中 ws_edge_* 对应)
WS_X = (0.20, 0.30)
WS_Y = (-0.09, 0.0)
WS_CENTER = np.array([(WS_X[0] + WS_X[1]) / 2, (WS_Y[0] + WS_Y[1]) / 2])  # (0.235, -0.01)
TARGET_XY = np.array([0.29, 0.00])   # 固定目标位置

# 推动参数
HOME_Z          = 0.1    # 起始 Home 高度
PUSH_Z          = 0.03  # 末端下降后保持的推送高度
CUBE_SIZE       = 0.03
PRE_BACK_DIST   = 0.06    # 从方块背后退多少
OVERSHOOT       = 0.01    # 推过目标的余量

# Home 位置: workspace 中心, 起始高度 HOME_Z
HOME_XYZ = np.array([*WS_CENTER, HOME_Z])
HOME_FALLBACK_QPOS = np.array([1.6, -2.2, 1.8, 1.5, -1.5, 0.0])
HOME_IK_SEEDS = [
    [0.0, -1.2, 1.8, 1.8, -1.57, 0.0],
    [0.0, -1.0, 1.5, 1.8, -1.57, 0.0],
    [0.0, -1.4, 2.0, 1.6, -1.57, 0.0],
    [0.0, -1.2, 1.8, 1.7, -1.57, 0.0],
    [0.0, -1.6, 2.2, 1.5, -1.57, 0.0],
    [0.0, -0.8, 1.3, 1.9, -1.57, 0.0],
]

# 任务参数
TARGET_SUCCESSES  = 5
ROUND_TIME_LIMIT  = 10.0   # 单轮最长时间(秒)
MAX_ATTEMPTS      = 4      # 单轮最多推几次
SUCCESS_TOL       = 0.04   # cube 离 target 多近算成功(米)
SETTLE_TOL        = 0.012  # 末端稳定判定(米)
SETTLE_TIMEOUT    = 2.5    # 最长等待稳定时间(秒)
MIN_DIST, MAX_DIST = 0.06, 0.18   # cube-to-target 合法距离范围

# IK 参数
IK_DAMPING        = 0.06
IK_MAX_ITERS      = 800
IK_STEP_SCALE     = 0.3
IK_TOL            = 5e-4
IK_ACCEPT         = 0.01   # 10mm,接受阈值
IK_ACCEPT_LO      = 0.03   # 30mm,宽松阈值
IK_RANDOM_RESTARTS = 3
IK_RANDOM_STD     = 0.15

# 相机
CAM_W, CAM_H   = 640, 480
CAM_EVERY_N    = 3     # 每 N 步刷一次手腕相机窗口


# ─────────────────────── IK ───────────────────────

def tip_mid(data, g1, g2):
    return 0.5 * (data.geom_xpos[g1] + data.geom_xpos[g2])


def solve_ik(model, data, g1, g2, target, seed, rng, jaw=0.0):
    """阻尼最小二乘 IK,多初始姿态 + 自适应阻尼/步长。返回 (qpos, err)。"""
    has_jaw = data.ctrl.shape[0] >= 7
    backup  = data.qpos.copy()
    best_q, best_err = seed.copy(), np.inf

    for noise in (0.0, 0.1, 0.2):
        q = seed + rng.normal(0, noise, 6) if noise else seed.copy()
        err_norm = np.inf

        for it in range(IK_MAX_ITERS):
            data.qpos[:6] = q
            if has_jaw:
                data.qpos[6] = jaw
            mujoco.mj_forward(model, data)

            err = target - tip_mid(data, g1, g2)
            err_norm = np.linalg.norm(err)
            if err_norm < IK_TOL:
                break

            jacp1 = np.zeros((3, model.nv))
            jacp2 = np.zeros((3, model.nv))
            mujoco.mj_jacGeom(model, data, jacp1, np.zeros((3, model.nv)), g1)
            mujoco.mj_jacGeom(model, data, jacp2, np.zeros((3, model.nv)), g2)
            J = 0.5 * (jacp1[:, :6] + jacp2[:, :6])

            d = IK_DAMPING * (2.0 if it > IK_MAX_ITERS // 2 else
                              0.5 if err_norm > 0.1 else 1.0)
            JJT = J @ J.T + (d ** 2) * np.eye(3)
            dq  = J.T @ np.linalg.solve(JJT, err)
            q   = q + min(IK_STEP_SCALE, err_norm * 0.5) * dq

            for j in range(6):
                lo, hi = model.jnt_range[j]
                m = (hi - lo) * 0.05
                q[j] = np.clip(q[j], lo + m, hi - m)

        if err_norm < best_err:
            best_err, best_q = err_norm, q.copy()

    data.qpos[:] = backup
    mujoco.mj_forward(model, data)
    return best_q, best_err


def solve_ik_robust(model, data, g1, g2, target, seed, rng):
    """带偏移兜底的 IK。返回 (qpos, err) 或在完全失败时返回 None。"""
    # 扫描几个夹爪开合度,选误差最小且相机不遮挡的
    has_jaw = data.ctrl.shape[0] >= 7
    jaws = np.linspace(0.01, 0.05, 7) if has_jaw else [0.0]
    best_q, best_err, best_jaw = seed.copy(), np.inf, 0.0

    for jaw in jaws:
        q, err = solve_ik(model, data, g1, g2, target, seed, rng, jaw)
        if err < best_err:
            best_err, best_q, best_jaw = err, q, jaw
        if err < IK_ACCEPT:
            break

    if best_err < IK_ACCEPT:
        return best_q, best_err

    # 位置偏移兜底
    offsets = [
        [0.01, 0, 0], [-0.01, 0, 0], [0, 0.01, 0], [0, -0.01, 0],
        [0.01, 0.01, 0], [0.01, -0.01, 0], [-0.01, 0.01, 0], [-0.01, -0.01, 0],
        [0, 0, 0.02], [0, 0, -0.02]
    ]
    for off in offsets:
        q, err = solve_ik(model, data, g1, g2, target + np.array(off), seed, rng)
        if err < IK_ACCEPT:
            return q, err
        if err < best_err:
            best_err, best_q = err, q

    # 随机重启：如果目标在可达范围但解算陷入局部最优，尝试更多起始点
    jnt_lo = model.jnt_range[:6, 0]
    jnt_hi = model.jnt_range[:6, 1]
    for _ in range(IK_RANDOM_RESTARTS):
        random_seed = seed + rng.normal(0, IK_RANDOM_STD, 6)
        random_seed = np.clip(random_seed, jnt_lo, jnt_hi)
        for jaw in jaws:
            q, err = solve_ik(model, data, g1, g2, target, random_seed, rng, jaw)
            if err < IK_ACCEPT:
                return q, err
            if err < best_err:
                best_err, best_q, best_jaw = err, q, jaw

    if best_err < IK_ACCEPT_LO:
        return best_q, best_err

    return None


# ─────────────────────── 场景控制 ───────────────────────

def send_ctrl(data, arm_qpos, jaw=0.0):
    data.ctrl[:6] = arm_qpos
    if data.ctrl.shape[0] >= 7:
        data.ctrl[6] = jaw


def reset_cube(model, data, cube_xy):
    jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT)
    qadr = model.jnt_qposadr[jid]
    data.qpos[qadr:qadr+3] = [cube_xy[0], cube_xy[1], CUBE_SIZE / 2 + 0.002]
    data.qpos[qadr+3]      = 1.0   # quaternion w
    data.qpos[qadr+4:qadr+7] = 0.0
    data.qvel[model.jnt_dofadr[jid]:model.jnt_dofadr[jid]+6] = 0.0
    mujoco.mj_forward(model, data)


def sample_cube(rng):
    """随机采样使 cube-to-target 距离在 [MIN_DIST, MAX_DIST] 内。"""
    for _ in range(500):
        cube = np.array([rng.uniform(*WS_X), rng.uniform(*WS_Y)])
        if MIN_DIST <= np.linalg.norm(cube - TARGET_XY) <= MAX_DIST:
            return cube
    return np.array([0.20, -0.05])


# ─────────────────────── 运动执行 ───────────────────────

_cam_step = 0

def sim_step(model, data, viewer, renderer, cam_id, info=""):
    global _cam_step
    mujoco.mj_step(model, data)
    viewer.sync()
    _cam_step += 1
    if _cam_step % CAM_EVERY_N == 0:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam_id)
        frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        for i, line in enumerate(info.split("\n")):
            cv2.putText(frame, line, (8, 20 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.imshow("Wrist Camera", frame)
        cv2.waitKey(1)


def execute_segment(model, data, viewer, renderer, cam_id,
                    g1, g2, q_from, q_to, target_tip,
                    duration, name="", info="", jaw_from=0.0, jaw_to=0.0):
    """关节空间线性插值到目标,然后等末端稳定。返回末端位置,viewer关闭则返回None。"""
    has_jaw = data.ctrl.shape[0] >= 7
    dt      = model.opt.timestep
    steps   = int(duration / dt)

    for i in range(steps):
        if not viewer.is_running():
            return None
        a = (i + 1) / steps
        send_ctrl(data,
                  q_from + a * (q_to - q_from),
                  jaw_from + a * (jaw_to - jaw_from) if has_jaw else 0.0)
        sim_step(model, data, viewer, renderer, cam_id, f"{info}\n{name}")
        time.sleep(dt)

    t0 = time.time()
    while viewer.is_running():
        send_ctrl(data, q_to, jaw_to if has_jaw else 0.0)
        sim_step(model, data, viewer, renderer, cam_id, f"{info}\n{name} settling")
        time.sleep(dt)
        if np.linalg.norm(tip_mid(data, g1, g2) - target_tip) < SETTLE_TOL:
            break
        if time.time() - t0 > SETTLE_TIMEOUT:
            break

    return tip_mid(data, g1, g2)


# ─────────────────────── 推动规划 & 执行 ───────────────────────

def execute_push(model, data, viewer, renderer, cam_id,
                 g1, g2, home_qpos, cube_xy, rng, info=""):
    """
    Home → BackPos → Descend → PushEnd → Home.
    先移动到背后点的 XY 平面位置再下降到 PUSH_Z，避免在 z 高度直接撞到方块。
    返回 (success, ik_failed)。
    """
    direction = TARGET_XY - cube_xy
    dist = np.linalg.norm(direction)
    if dist < 1e-4:
        return True, False
    d_hat = direction / dist

    def pt(xy, z=PUSH_Z):
        return np.array([xy[0], xy[1], z])

    back_xy = cube_xy - d_hat * PRE_BACK_DIST
    waypoints = [
        ("BackPos", pt(back_xy, HOME_Z),                 1.2),
        ("Descend", pt(back_xy, PUSH_Z),                  0.8),
        ("PushEnd", pt(TARGET_XY + d_hat * OVERSHOOT),    max(0.8, dist * 6.0)),
        ("Home",    HOME_XYZ,                             1.2),
    ]

    # 预解所有 IK
    plan = []
    seed = home_qpos.copy()
    for name, target, dur in waypoints:
        result = solve_ik_robust(model, data, g1, g2, target, seed, rng)
        if result is None:
            print(f"    [{name}] IK 失败,放弃")
            return False, True
        q, err = result
        print(f"    [{name}] IK err={err*1000:.1f}mm")
        plan.append((name, q, target, dur))
        seed = q

    # 逐段执行
    q_prev, jaw_prev = home_qpos.copy(), 0.0
    for name, q, tip_target, dur in plan:
        result = execute_segment(
            model, data, viewer, renderer, cam_id, g1, g2,
            q_prev, q, tip_target, dur, name, info,
            jaw_from=jaw_prev, jaw_to=0.0,
        )
        if result is None:
            return False, False
        q_prev = q

    return True, False


# ─────────────────────── 单轮 ───────────────────────

def run_round(model, data, viewer, renderer, cam_id,
              g1, g2, cube_id, home_qpos, cube_xy, rng, round_idx):
    print(f"\n{'='*55}")
    print(f"第 {round_idx+1} 轮  cube={cube_xy.round(3)}  target={TARGET_XY}")
    print(f"{'='*55}")

    t0 = time.time()
    attempt = 1
    while attempt <= MAX_ATTEMPTS:
        elapsed   = time.time() - t0
        cube_now  = data.xpos[cube_id][:2].copy()
        err       = np.linalg.norm(cube_now - TARGET_XY)

        if err < SUCCESS_TOL:
            print(f"  ✅ 成功 attempt={attempt-1}  {elapsed:.1f}s  err={err*1000:.0f}mm")
            return True
        if ROUND_TIME_LIMIT - elapsed < 2.0:
            print(f"  ❌ 超时  {elapsed:.1f}s  err={err*1000:.0f}mm")
            return False

        info = f"R{round_idx+1} #{attempt} err={err*1000:.0f}mm"
        print(f"  第{attempt}次  err={err*1000:.0f}mm  剩余{ROUND_TIME_LIMIT-elapsed:.1f}s")
        ok, ik_failed = execute_push(model, data, viewer, renderer, cam_id,
                                     g1, g2, home_qpos, cube_now, rng, info)
        if ok:
            return True
        if ik_failed:
            print("  ⚠ IK 失败，重置本轮并重新尝试")
            reset_cube(model, data, cube_xy)
            data.qpos[:6] = home_qpos
            send_ctrl(data, home_qpos)
            mujoco.mj_forward(model, data)
            attempt += 1
            continue

        return False

    err = np.linalg.norm(data.xpos[cube_id][:2] - TARGET_XY)
    ok  = err < SUCCESS_TOL
    print(f"  {'✅' if ok else '❌'} 用尽  err={err*1000:.0f}mm")
    return ok


# ─────────────────────── main ───────────────────────

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # 检查必要对象
    for name, typ in [(FIXED_TIP_GEOM,  mujoco.mjtObj.mjOBJ_GEOM),
                      (MOVING_TIP_GEOM, mujoco.mjtObj.mjOBJ_GEOM),
                      (CUBE_BODY,       mujoco.mjtObj.mjOBJ_BODY),
                      (CUBE_JOINT,      mujoco.mjtObj.mjOBJ_JOINT),
                      (WRIST_CAM,       mujoco.mjtObj.mjOBJ_CAMERA)]:
        if mujoco.mj_name2id(model, typ, name) < 0:
            raise RuntimeError(f"XML 中缺少 '{name}'")

    g1      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FIXED_TIP_GEOM)
    g2      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, MOVING_TIP_GEOM)
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY)
    cam_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAM)

    rng = np.random.default_rng()
    renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    cv2.namedWindow("Wrist Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Wrist Camera", CAM_W, CAM_H)

    # 固定 Home 姿态
    print("=== 使用固定 Home 姿态 ===")
    home_qpos = HOME_FALLBACK_QPOS.copy()
    print(f"  Home qpos = {home_qpos}")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance  = 0.7
            viewer.cam.azimuth   = 135
            viewer.cam.elevation = -35
            viewer.cam.lookat[:] = [0.235, -0.01, 0.05]

            results = []
            successes = 0
            rnd = 0
            while viewer.is_running() and successes < TARGET_SUCCESSES:
                if not viewer.is_running():
                    break

                cube_xy = sample_cube(rng)
                reset_cube(model, data, cube_xy)
                data.qpos[:6] = home_qpos
                send_ctrl(data, home_qpos)
                mujoco.mj_forward(model, data)

                # 物理稳定 0.5s
                end = time.time() + 0.5
                while viewer.is_running() and time.time() < end:
                    send_ctrl(data, home_qpos)
                    sim_step(model, data, viewer, renderer, cam_id,
                             f"Round {rnd+1} READY")
                    time.sleep(model.opt.timestep)

                ok = run_round(model, data, viewer, renderer, cam_id,
                               g1, g2, cube_id, home_qpos, cube_xy, rng, rnd)
                results.append(ok)
                successes += int(ok)

                # 轮间暂停
                end = time.time() + 1.0
                while viewer.is_running() and time.time() < end:
                    sim_step(model, data, viewer, renderer, cam_id,
                             f"Round {rnd+1} {'OK' if ok else 'FAIL'}")
                    time.sleep(model.opt.timestep)
                rnd += 1

            n, w = len(results), sum(results)
            print(f"\n{'='*55}")
            print(f"累计成功 {w}/{TARGET_SUCCESSES} | 总轮数 {n} | 成功率 {w/max(n,1)*100:.0f}%")
            print(f"{'='*55}")

            end = time.time() + 3
            while viewer.is_running() and time.time() < end:
                sim_step(model, data, viewer, renderer, cam_id, "DONE")
                time.sleep(model.opt.timestep)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

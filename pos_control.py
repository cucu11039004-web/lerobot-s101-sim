"""
pos_control.py — SO-ARM100 末端位置 (xyz) 控制

在 qpos_control.QposControl 基础上增加:
    - forward kinematics: 末端 "tip" 位置 = 两个夹爪 pad 的几何中点
    - inverse kinematics: 阻尼最小二乘 + 自适应阻尼/步长 + 多 seed / 偏移兜底 / 随机重启
    - set_ee_pose(xyz)     解 IK 并瞬间重置到该末端位置
    - move_ee_to(xyz, dur) 解 IK 后关节空间线性插值, 末端稳定判据

夹爪始终保持关闭 (jaw=0.0)。

运行: mjpython pos_control.py  (独立 demo: 画 xy 小方形轨迹)
依赖: 继承 qpos_control.py
"""

from typing import Optional, Tuple

import mujoco
import numpy as np

from qpos_control import QposControl


# ─────────────────────── 默认配置 ───────────────────────

DEFAULT_FIXED_TIP_GEOM  = "fixed_jaw_pad_1"
DEFAULT_MOVING_TIP_GEOM = "moving_jaw_pad_1"

# IK 参数 (与原 push_demo 保持一致)
IK_DAMPING          = 0.06
IK_MAX_ITERS        = 800
IK_STEP_SCALE       = 0.3
IK_TOL              = 5e-4
IK_ACCEPT           = 0.01   # 10mm, 可接受阈值
IK_ACCEPT_LO        = 0.03   # 30mm, 宽松阈值
IK_RANDOM_RESTARTS  = 3
IK_RANDOM_STD       = 0.15

SETTLE_TOL       = 0.012    # 末端稳定判据 (米)
SETTLE_TIMEOUT   = 2.5      # 最长等待时间 (秒)

JAW_CLOSED = 0.0            # 夹爪始终关闭


class PosControl(QposControl):
    """
    末端位置 (xyz) 控制器。夹爪始终关闭。
    "末端" 定义为两个 jaw pad geom 的几何中点 (tip_mid), 与原 push_demo 完全一致。
    """

    def __init__(
        self,
        fixed_tip_geom:  str = DEFAULT_FIXED_TIP_GEOM,
        moving_tip_geom: str = DEFAULT_MOVING_TIP_GEOM,
        rng: Optional[np.random.Generator] = None,
        **qpos_kwargs,
    ):
        super().__init__(**qpos_kwargs)

        g1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, fixed_tip_geom)
        g2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, moving_tip_geom)
        if g1 < 0 or g2 < 0:
            raise RuntimeError(
                f"tip geom 未找到: {fixed_tip_geom}={g1}, {moving_tip_geom}={g2}"
            )
        self.g1 = g1
        self.g2 = g2

        self.rng = rng if rng is not None else np.random.default_rng()

        # 默认 IK seed (可用 set_ik_seed 覆盖)
        self._default_seed = np.array([0.0, -1.2, 1.8, 1.8, -1.57, 0.0])

    # ─────────────── 正运动学 ───────────────

    def tip_mid(self) -> np.ndarray:
        """当前末端 (两 pad 中点) 的世界坐标 xyz。"""
        return 0.5 * (self.data.geom_xpos[self.g1] + self.data.geom_xpos[self.g2])

    def get_ee_xyz(self) -> np.ndarray:
        return self.tip_mid().copy()

    # ─────────────── IK ───────────────

    def set_ik_seed(self, seed: np.ndarray):
        self._default_seed = np.asarray(seed, dtype=float).reshape(-1).copy()

    def _solve_ik_single(
        self,
        target_xyz: np.ndarray,
        seed: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """阻尼最小二乘 IK, 多 noise 扰动, 返回 (qpos, err)。不改动 data 状态。"""
        backup = self.data.qpos.copy()
        best_q, best_err = seed.copy(), np.inf

        for noise in (0.0, 0.1, 0.2):
            q = seed + self.rng.normal(0, noise, self.arm_dim) if noise else seed.copy()
            err_norm = np.inf

            for it in range(IK_MAX_ITERS):
                self.data.qpos[:self.arm_dim] = q
                mujoco.mj_forward(self.model, self.data)

                err = target_xyz - self.tip_mid()
                err_norm = float(np.linalg.norm(err))
                if err_norm < IK_TOL:
                    break

                jacp1 = np.zeros((3, self.model.nv))
                jacp2 = np.zeros((3, self.model.nv))
                mujoco.mj_jacGeom(self.model, self.data, jacp1,
                                  np.zeros((3, self.model.nv)), self.g1)
                mujoco.mj_jacGeom(self.model, self.data, jacp2,
                                  np.zeros((3, self.model.nv)), self.g2)
                J = 0.5 * (jacp1[:, :self.arm_dim] + jacp2[:, :self.arm_dim])

                d = IK_DAMPING * (
                    2.0 if it > IK_MAX_ITERS // 2 else
                    0.5 if err_norm > 0.1 else
                    1.0
                )
                JJT = J @ J.T + (d ** 2) * np.eye(3)
                dq  = J.T @ np.linalg.solve(JJT, err)
                q   = q + min(IK_STEP_SCALE, err_norm * 0.5) * dq

                for j in range(self.arm_dim):
                    lo, hi = self.model.jnt_range[j]
                    m = (hi - lo) * 0.05
                    q[j] = np.clip(q[j], lo + m, hi - m)

            if err_norm < best_err:
                best_err, best_q = err_norm, q.copy()

        # 还原 data
        self.data.qpos[:] = backup
        mujoco.mj_forward(self.model, self.data)
        return best_q, best_err

    def solve_ik(
        self,
        target_xyz: np.ndarray,
        seed: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        鲁棒 IK: 多次尝试 + 位置偏移兜底 + 随机重启。
        成功返回 (qpos, err), 完全失败返回 None。
        """
        if seed is None:
            seed = self._default_seed.copy()
        seed = np.asarray(seed, dtype=float).reshape(-1).copy()

        best_q, best_err = seed.copy(), np.inf

        # 1) 初次求解
        q, err = self._solve_ik_single(target_xyz, seed)
        if err < best_err:
            best_err, best_q = err, q
        if best_err < IK_ACCEPT:
            return best_q, best_err

        # 2) 位置偏移兜底
        offsets = [
            [ 0.01, 0,    0], [-0.01, 0,    0],
            [ 0,    0.01, 0], [ 0,   -0.01, 0],
            [ 0.01, 0.01, 0], [ 0.01,-0.01, 0],
            [-0.01, 0.01, 0], [-0.01,-0.01, 0],
            [ 0,    0,    0.02], [ 0,    0,   -0.02],
        ]
        for off in offsets:
            q, err = self._solve_ik_single(target_xyz + np.array(off), seed)
            if err < IK_ACCEPT:
                return q, err
            if err < best_err:
                best_err, best_q = err, q

        # 3) 随机重启
        jnt_lo = self.model.jnt_range[:self.arm_dim, 0]
        jnt_hi = self.model.jnt_range[:self.arm_dim, 1]
        for _ in range(IK_RANDOM_RESTARTS):
            random_seed = seed + self.rng.normal(0, IK_RANDOM_STD, self.arm_dim)
            random_seed = np.clip(random_seed, jnt_lo, jnt_hi)
            q, err = self._solve_ik_single(target_xyz, random_seed)
            if err < IK_ACCEPT:
                return q, err
            if err < best_err:
                best_err, best_q = err, q

        if best_err < IK_ACCEPT_LO:
            return best_q, best_err
        return None

    # ─────────────── 末端位置控制 ───────────────

    def set_ee_pose(
        self,
        xyz: np.ndarray,
        seed: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """
        解 IK, 瞬间重置机械臂到该末端位置。夹爪保持关闭。
        返回 IK 误差 (米); 失败返回 None。
        """
        result = self.solve_ik(np.asarray(xyz, dtype=float), seed=seed)
        if result is None:
            return None
        q, err = result
        self.reset_qpos(q, jaw=JAW_CLOSED, forward=True)
        return err

    def move_ee_to(
        self,
        xyz: np.ndarray,
        duration: float = 1.0,
        seed: Optional[np.ndarray] = None,
        info: str = "",
        realtime: bool = True,
        settle: bool = True,
    ) -> Optional[np.ndarray]:
        """
        关节空间线性插值, 让末端从当前位置运动到 xyz。夹爪始终关闭。
        过程:
            1) 解 IK 得到目标 qpos (从当前 qpos 作为 seed)
            2) 从当前 qpos 线性插值到目标 qpos, 时长 duration
            3) 可选: 在 SETTLE_TIMEOUT 内等待末端收敛到 SETTLE_TOL
        返回:
            - 成功: 最终末端 xyz (np.ndarray)
            - viewer 被关闭: None
            - IK 失败: None
        """
        target_xyz = np.asarray(xyz, dtype=float).reshape(3)
        if seed is None:
            seed = self.get_qpos()
        result = self.solve_ik(target_xyz, seed=seed)
        if result is None:
            print(f"[PosControl] IK 失败 target={target_xyz}")
            return None
        q_target, ik_err = result

        q_from = self.get_qpos()
        ok = self.move_qpos_linear(
            q_to=q_target, duration=duration,
            q_from=q_from, jaw_from=JAW_CLOSED, jaw_to=JAW_CLOSED,
            info=info, realtime=realtime,
        )
        if not ok:
            return None

        # settle
        if settle:
            import time as _time
            t0 = _time.time()
            while self.is_running:
                self.set_qpos(q_target, JAW_CLOSED)
                self.step(f"{info} settling")
                if realtime:
                    _time.sleep(self.dt)
                if np.linalg.norm(self.tip_mid() - target_xyz) < SETTLE_TOL:
                    break
                if _time.time() - t0 > SETTLE_TIMEOUT:
                    break
            if not self.is_running:
                return None

        return self.tip_mid()


# ─────────────────────── Demo ───────────────────────

def _demo():
    """独立运行: 末端画一个小方形 (在 workspace 中心上方)。"""
    WS_CENTER = np.array([0.235, -0.01])
    Z_HI = 0.12
    Z_LO = 0.05

    corners = [
        [WS_CENTER[0] - 0.03, WS_CENTER[1] - 0.03, Z_HI],
        [WS_CENTER[0] + 0.03, WS_CENTER[1] - 0.03, Z_HI],
        [WS_CENTER[0] + 0.03, WS_CENTER[1] + 0.03, Z_HI],
        [WS_CENTER[0] - 0.03, WS_CENTER[1] + 0.03, Z_HI],
        [WS_CENTER[0] - 0.03, WS_CENTER[1] - 0.03, Z_LO],  # 下降
        [WS_CENTER[0] - 0.03, WS_CENTER[1] - 0.03, Z_HI],  # 回升
    ]

    home_qpos = np.array([1.6, -2.2, 1.8, 1.5, -1.5, 0.0])

    with PosControl() as robot:
        robot.reset_qpos(home_qpos, jaw=JAW_CLOSED)
        robot.hold(0.5, info="READY")
        for i, xyz in enumerate(corners):
            if not robot.is_running:
                break
            print(f"[demo] move -> corner {i}: {xyz}")
            res = robot.move_ee_to(np.array(xyz), duration=1.2, info=f"corner {i}")
            if res is None:
                print("  viewer closed 或 IK 失败")
                break
            print(f"  tip xyz = {res.round(4)}")
        robot.hold(1.0, info="DONE")


if __name__ == "__main__":
    _demo()
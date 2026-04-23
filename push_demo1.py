"""
push_demo.py — SO-ARM100 推方块任务 (Gym 风格)

基于 pos_control.PosControl, 把原推方块 demo 重构成标准 RL 风格的接口:

    env = PushEnv()
    obs, info = env.reset()
    while not done:
        action = env.plan_push(obs)     # 或外部策略给出的 action
        obs, reward, terminated, truncated, info = env.step(action)
    env.close()

action 语义 (连续, 任务空间 waypoint):
    一个 "推一次" 的完整动作被拆成若干 waypoint, 每次 step() 执行 1 个 waypoint:
        action = np.array([x, y, z, duration])    (最小形式)
    即末端运动到该 xyz 位置, 耗时 duration 秒。
    这样外部策略 / 脚本都可以逐 waypoint 发送。

也提供一个高层辅助:
    env.plan_push_waypoints(cube_xy) -> list of waypoints
    env.execute_push(cube_xy)         -> 整条推送序列, 内部循环调 step()

观测:
    dict(
        qpos       : (6,)    机械臂关节
        ee_xyz     : (3,)    末端位置
        cube_xy    : (2,)    方块 xy
        cube_xyz   : (3,)    方块 xyz
        target_xy  : (2,)    目标 xy
        cube_to_target : float  方块到目标的 xy 距离
    )

奖励:
    reward = prev_cube_to_target - cube_to_target      (推近目标为正)
    额外在成功时给 +1, 时间惩罚 -0.01 / step。

运行: mjpython push_demo.py
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from pos_control import PosControl


# ─────────────────────── 场景 / 任务常量 ───────────────────────

XML_PATH    = "./push_scene.xml"
CUBE_BODY   = "cube"
CUBE_JOINT  = "cube_free"
TARGET_BODY = "target"

WS_X = (0.20, 0.30)
WS_Y = (-0.09, 0.00)
WS_CENTER = np.array([(WS_X[0] + WS_X[1]) / 2, (WS_Y[0] + WS_Y[1]) / 2])
TARGET_XY = np.array([0.29, 0.00])

# 推送几何参数
HOME_Z        = 0.10
PUSH_Z        = 0.03
CUBE_SIZE     = 0.03
PRE_BACK_DIST = 0.06
OVERSHOOT     = 0.01

HOME_XYZ           = np.array([*WS_CENTER, HOME_Z])
HOME_FALLBACK_QPOS = np.array([1.6, -2.2, 1.8, 1.5, -1.5, 0.0])

# 任务参数
TARGET_SUCCESSES = 5
ROUND_TIME_LIMIT = 10.0
MAX_ATTEMPTS     = 4
SUCCESS_TOL      = 0.04
MIN_DIST, MAX_DIST = 0.06, 0.18


@dataclass
class PushConfig:
    xml_path:         str   = XML_PATH
    target_xy:        np.ndarray = field(default_factory=lambda: TARGET_XY.copy())
    home_qpos:        np.ndarray = field(default_factory=lambda: HOME_FALLBACK_QPOS.copy())
    success_tol:      float = SUCCESS_TOL
    round_time_limit: float = ROUND_TIME_LIMIT
    max_attempts:     int   = MAX_ATTEMPTS
    seed:             Optional[int] = None


# ─────────────────────── PushEnv ───────────────────────

class PushEnv(PosControl):
    """
    推方块环境. 继承 PosControl -> QposControl, 所以也可以直接用 set_qpos / set_ee_pose 做调试。

    典型用法 (脚本式):
        env = PushEnv()
        env.reset()
        env.execute_push_task(num_rounds=5)

    典型用法 (Gym 风格):
        env = PushEnv()
        obs, info = env.reset()
        for wp in env.plan_push_waypoints(obs["cube_xy"]):
            obs, r, done, trunc, info = env.step(wp)
    """

    def __init__(self, config: Optional[PushConfig] = None, **kwargs):
        self.cfg = config or PushConfig()
        rng = np.random.default_rng(self.cfg.seed)

        # 交给父类: 加载模型 / viewer / 相机 / IK
        kwargs.setdefault("xml_path", self.cfg.xml_path)
        super().__init__(rng=rng, **kwargs)

        # 任务对象 id
        self.cube_body_id  = self._require_id(mujoco.mjtObj.mjOBJ_BODY,  CUBE_BODY)
        self.cube_joint_id = self._require_id(mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT)
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY
        )

        # 每轮状态
        self._round_idx      = 0
        self._attempt        = 0
        self._round_start    = 0.0
        self._prev_dist      = None
        self._ep_step        = 0
        self._initial_cube_xy = None

        # 任务参数
        self.target_xy = np.asarray(self.cfg.target_xy, dtype=float).reshape(2)
        self.home_qpos = np.asarray(self.cfg.home_qpos, dtype=float).reshape(self.arm_dim)

    def _require_id(self, obj_type, name: str) -> int:
        i = mujoco.mj_name2id(self.model, obj_type, name)
        if i < 0:
            raise RuntimeError(f"XML 中缺少 '{name}'")
        return i

    # ─────────────── 观测 / 奖励 ───────────────

    def _cube_xyz(self) -> np.ndarray:
        return self.data.xpos[self.cube_body_id].copy()

    def _cube_xy(self) -> np.ndarray:
        return self._cube_xyz()[:2]

    def _cube_to_target(self) -> float:
        return float(np.linalg.norm(self._cube_xy() - self.target_xy))

    def _get_obs(self) -> Dict[str, Any]:
        cube_xyz = self._cube_xyz()
        return {
            "qpos":           self.get_qpos(),
            "ee_xyz":         self.get_ee_xyz(),
            "cube_xy":        cube_xyz[:2].copy(),
            "cube_xyz":       cube_xyz,
            "target_xy":      self.target_xy.copy(),
            "cube_to_target": self._cube_to_target(),
        }

    def _compute_reward(self, prev_dist: float, new_dist: float,
                        success: bool) -> float:
        # 方块靠近目标的 xy 位移增量 (正值 = 靠近)
        shaping = prev_dist - new_dist
        time_pen = -0.01
        bonus    = 1.0 if success else 0.0
        return float(shaping + time_pen + bonus)

    # ─────────────── 场景重置 ───────────────

    def _reset_cube(self, cube_xy: np.ndarray):
        qadr = self.model.jnt_qposadr[self.cube_joint_id]
        self.data.qpos[qadr:qadr + 3] = [cube_xy[0], cube_xy[1], CUBE_SIZE / 2 + 0.002]
        self.data.qpos[qadr + 3]      = 1.0
        self.data.qpos[qadr + 4:qadr + 7] = 0.0
        dof = self.model.jnt_dofadr[self.cube_joint_id]
        self.data.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _sample_cube_xy(self) -> np.ndarray:
        for _ in range(500):
            xy = np.array([
                self.rng.uniform(*WS_X),
                self.rng.uniform(*WS_Y),
            ])
            if MIN_DIST <= np.linalg.norm(xy - self.target_xy) <= MAX_DIST:
                return xy
        return np.array([0.20, -0.05])

    def reset(
        self,
        cube_xy: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Gym 风格 reset: 重置 cube 位置 + 机械臂回到 home qpos + 物理稳定 0.5s.
        返回 (obs, info).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if cube_xy is None:
            cube_xy = self._sample_cube_xy()
        cube_xy = np.asarray(cube_xy, dtype=float).reshape(2)

        self._reset_cube(cube_xy)
        self.reset_qpos(self.home_qpos, jaw=0.0, forward=True)
        # 物理稳定
        self.hold(0.5, info=f"R{self._round_idx + 1} READY")

        self._initial_cube_xy = cube_xy.copy()
        self._round_start     = time.time()
        self._attempt         = 0
        self._ep_step         = 0
        self._prev_dist       = self._cube_to_target()

        obs  = self._get_obs()
        info = {
            "round":      self._round_idx,
            "cube_xy":    cube_xy,
            "target_xy":  self.target_xy.copy(),
        }
        return obs, info

    # ─────────────── step (Gym 风格) ───────────────

    def step(
        self,
        action: Any = None,
        info: str = "",
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]] | None:
        """
        执行一个 waypoint-style action.
            action: [x, y, z, duration]  (duration 可选, 默认 1.0)
        过程:
            1) 解 IK, 关节插值到该 ee_xyz, 带末端稳定
            2) 读新观测, 计算 shaping reward
            3) 判 success / 时限 / IK 失败
        返回 (obs, reward, terminated, truncated, info).
        """
        # 兼容父类 QposControl.step(info="...") / hold() / move_qpos_linear() 的调用方式。
        if action is None or isinstance(action, str):
            super().step(info=action if isinstance(action, str) else info)
            return None

        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape[0] < 3:
            raise ValueError(f"action 至少需要 [x,y,z], 实际 {action.shape}")
        xyz      = action[:3]
        duration = float(action[3]) if action.shape[0] >= 4 else 1.0

        self._ep_step += 1
        info: Dict[str, Any] = {
            "ep_step":  self._ep_step,
            "action":   action.copy(),
            "ik_ok":    True,
            "viewer_closed": False,
        }

        # 运动
        tip = self.move_ee_to(
            xyz=xyz,
            duration=duration,
            info=f"R{self._round_idx + 1} step {self._ep_step}",
        )
        if tip is None:
            # 要么 viewer 关了, 要么 IK 失败. 用 is_running 区分.
            if not self.is_running:
                info["viewer_closed"] = True
                obs = self._get_obs()
                return obs, 0.0, False, True, info
            info["ik_ok"] = False
            obs = self._get_obs()
            reward = self._compute_reward(self._prev_dist, obs["cube_to_target"], False)
            self._prev_dist = obs["cube_to_target"]
            return obs, reward, False, False, info

        # 新观测
        obs = self._get_obs()
        new_dist = obs["cube_to_target"]
        success  = new_dist < self.cfg.success_tol
        elapsed  = time.time() - self._round_start
        truncated = elapsed > self.cfg.round_time_limit

        reward = self._compute_reward(self._prev_dist, new_dist, success)
        self._prev_dist = new_dist

        info.update({
            "cube_to_target": new_dist,
            "success":        success,
            "elapsed":        elapsed,
            "tip_xyz":        tip,
        })
        return obs, reward, success, truncated, info

    # ─────────────── 高层推送规划 ───────────────

    def plan_push_waypoints(self, cube_xy: np.ndarray) -> List[np.ndarray]:
        """
        规划一次 "推方块" 的 waypoint 序列 (action 列表, 长度 4: [x,y,z,dur]).
        顺序: BackPos(高) -> Descend(低) -> PushEnd(低, 过目标) -> Home(高)
        """
        cube_xy = np.asarray(cube_xy, dtype=float).reshape(2)
        direction = self.target_xy - cube_xy
        dist = float(np.linalg.norm(direction))
        if dist < 1e-4:
            return []
        d_hat = direction / dist

        back_xy  = cube_xy - d_hat * PRE_BACK_DIST
        push_end = self.target_xy + d_hat * OVERSHOOT

        waypoints = [
            # [x, y, z, duration]
            np.array([back_xy[0], back_xy[1], HOME_Z,  1.2]),
            np.array([back_xy[0], back_xy[1], PUSH_Z,  0.8]),
            np.array([push_end[0], push_end[1], PUSH_Z, max(0.8, dist * 6.0)]),
            np.array([HOME_XYZ[0], HOME_XYZ[1], HOME_XYZ[2], 1.2]),
        ]
        return waypoints

    def execute_push(self, cube_xy: np.ndarray) -> Tuple[bool, bool, Dict[str, Any]]:
        """
        执行一次完整推送 (多个 step). 返回 (success, ik_failed, info).

        与原 push_demo 的细微区别:
            - 原版: 预解所有 4 个 waypoint 的 IK, 全部成功后再逐段执行.
            - 本版: 逐 waypoint 在线解 IK (seed = 当前 qpos) + 执行.
          在线式更契合 Gym step 语义, 一般不影响成功率; 且每个 step()
          都能输出真实观测 + reward, 方便外部策略接入。
        """
        waypoints = self.plan_push_waypoints(cube_xy)
        names = ["BackPos", "Descend", "PushEnd", "Home"]
        last_info: Dict[str, Any] = {}

        for name, wp in zip(names, waypoints):
            obs, reward, terminated, truncated, info = self.step(wp)
            last_info = info
            last_info["waypoint"] = name
            if info.get("viewer_closed"):
                return False, False, last_info
            if not info.get("ik_ok", True):
                print(f"    [{name}] IK 失败, 放弃")
                return False, True, last_info
            print(f"    [{name}] reward={reward:+.3f}  d={info['cube_to_target']*1000:.0f}mm")
            if terminated:   # success
                return True, False, last_info
            if truncated:
                return False, False, last_info
        # 全部 waypoint 执行完也算一次推送结束; 成功与否看最后观测
        success = self._cube_to_target() < self.cfg.success_tol
        return success, False, last_info

    # ─────────────── 整套任务循环 ───────────────

    def run_round(self, cube_xy: np.ndarray) -> bool:
        """完整跑一个 round, 最多 max_attempts 次推送. 返回是否成功。"""
        print(f"\n{'='*55}")
        print(f"第 {self._round_idx + 1} 轮  cube={cube_xy.round(3)}  target={self.target_xy}")
        print(f"{'='*55}")

        while self._attempt < self.cfg.max_attempts and self.is_running:
            elapsed  = time.time() - self._round_start
            cur_xy   = self._cube_xy()
            err      = np.linalg.norm(cur_xy - self.target_xy)

            if err < self.cfg.success_tol:
                print(f"  ✅ 成功 attempt={self._attempt}  "
                      f"{elapsed:.1f}s  err={err*1000:.0f}mm")
                return True
            if self.cfg.round_time_limit - elapsed < 2.0:
                print(f"  ❌ 超时  {elapsed:.1f}s  err={err*1000:.0f}mm")
                return False

            self._attempt += 1
            print(f"  第{self._attempt}次  err={err*1000:.0f}mm  "
                  f"剩余{self.cfg.round_time_limit - elapsed:.1f}s")

            success, ik_failed, _info = self.execute_push(cur_xy)

            if success:
                return True
            if _info.get("viewer_closed"):
                return False
            if ik_failed:
                print("  ⚠ IK 失败, 重置本轮并重试")
                self._reset_cube(cube_xy)
                self.reset_qpos(self.home_qpos, jaw=0.0, forward=True)
                continue
            # 非 IK 失败但也没成功: 看有没有时间再试
            # 继续 while, 下一次循环会重新判定 err 和时限

        err = float(np.linalg.norm(self._cube_xy() - self.target_xy))
        ok  = err < self.cfg.success_tol
        print(f"  {'✅' if ok else '❌'} 用尽  err={err*1000:.0f}mm")
        return ok

    def execute_push_task(
        self,
        num_successes: int = TARGET_SUCCESSES,
    ):
        """
        跑完整任务: 重复 reset + run_round, 直到累计 num_successes 次成功或 viewer 关闭.
        """
        results: List[bool] = []
        successes = 0
        self._round_idx = 0

        while self.is_running and successes < num_successes:
            cube_xy = self._sample_cube_xy()
            self.reset(cube_xy=cube_xy)
            ok = self.run_round(self._initial_cube_xy)
            results.append(ok)
            successes += int(ok)

            # 轮间暂停
            self.hold(1.0, info=f"Round {self._round_idx + 1} {'OK' if ok else 'FAIL'}")
            self._round_idx += 1

        n = len(results)
        w = sum(results)
        print(f"\n{'='*55}")
        print(f"累计成功 {w}/{num_successes} | 总轮数 {n} | "
              f"成功率 {(w / max(n, 1)) * 100:.0f}%")
        print(f"{'='*55}")

        self.hold(3.0, info="DONE")


# ─────────────────────── main ───────────────────────

def main():
    env = PushEnv()
    try:
        env.execute_push_task(num_successes=TARGET_SUCCESSES)
    finally:
        env.close()


if __name__ == "__main__":
    main()

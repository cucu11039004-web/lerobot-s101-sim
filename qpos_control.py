"""
qpos_control.py — SO-ARM100 关节空间控制基类

提供最底层的 qpos 控制封装：
- 加载 MuJoCo 模型 / data / viewer / 手腕相机
- set_qpos(q, jaw) 直接设置 6 个关节 + 夹爪目标
- step() 推进一个仿真步, 同步 viewer, 可选刷手腕相机窗口
- reset_qpos(q) 瞬间重置机械臂到给定关节位置
- close() 关闭 viewer / 相机窗口

运行: mjpython qpos_control.py  (作为独立脚本时跑一个关节扫动 demo)
依赖: pip install mujoco opencv-python numpy
"""

import time
from typing import Optional

import cv2
import mujoco
import mujoco.viewer
import numpy as np


# ─────────────────────── 默认配置 ───────────────────────

DEFAULT_XML_PATH   = "/home/ubuntu/lerobot-s101-sim/push_scene.xml"
DEFAULT_WRIST_CAM  = "wrist_cam"

CAM_W, CAM_H       = 640, 480
CAM_EVERY_N        = 3        # 每 N 个仿真步刷一次手腕相机窗口

# 相机面板的默认视角
VIEW_CAM_DISTANCE  = 0.7
VIEW_CAM_AZIMUTH   = 135
VIEW_CAM_ELEVATION = -35
VIEW_CAM_LOOKAT    = (0.235, -0.01, 0.05)


class QposControl:
    """
    SO-ARM100 关节空间控制器。

    约定:
        - 机械臂 6 个关节 qpos/ctrl 索引 [0:6]:
              Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll  (前 5 个是 arm, 第 6 个是 Jaw)
          实际上 so_arm100.xml 里有 6 个非夹爪关节? 不 - 共 6 个关节, 其中最后一个是 Jaw。
          为兼容原 push_demo, 这里把「arm 关节」定义为 ctrl[:6] 的前 5 个 + Wrist_Roll 共 6 个? 
          -> 按原脚本: data.ctrl[:6] = arm_qpos, data.ctrl[6] = jaw, 所以 xml 有 7 个 actuator?
          实际 so_arm100.xml 只有 6 个 actuator (含 Jaw)。原脚本判断 data.ctrl.shape[0] >= 7,
          因此我们在这里统一按「arm_dim=6 (含 Wrist_Roll), jaw 视 ctrl 长度决定」实现。

        - 实际上 so_arm100.xml 的 6 个 actuator 是:
              [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]
          所以 arm_qpos 指的是前 5 个关节, 而 jaw = ctrl[5]。
          为了保持与原 push_demo 接口兼容, 这里做一层适配:
              set_qpos(arm_qpos_6) 里前 5 个写 ctrl[:5], 第 6 个写 ctrl[5] 当 jaw? 
          -> 看原代码: HOME_FALLBACK_QPOS = [1.6, -2.2, 1.8, 1.5, -1.5, 0.0] 是 6 个值,
             send_ctrl(data, arm_qpos=6, jaw=0) -> ctrl[:6] = arm_qpos (覆盖了 Jaw 那一位为 -1.5 再 0?)
          实际上 so_arm100.xml 的 6 个 actuator 顺序是 [Rot, Pit, Elb, WP, WR, Jaw],
          原脚本把 arm_qpos[5] 当 Wrist_Roll 写进 ctrl[5], 但 ctrl[5] 其实是 Jaw.
          这是原脚本的 bug / 兼容问题 - 我们保持完全一致的行为即可:
              ctrl[:6] <- arm_qpos[:6] 
          (原脚本在 data.ctrl.shape[0] >= 7 时才写 ctrl[6]=jaw, 但这里 shape==6 所以 jaw 被忽略)

    结论: 本类按 arm_qpos 长度 6 与原脚本保持一致, jaw 参数仅在 ctrl.shape[0] >= 7 时生效。
    """

    def __init__(
        self,
        xml_path: str = DEFAULT_XML_PATH,
        wrist_cam_name: Optional[str] = DEFAULT_WRIST_CAM,
        launch_viewer: bool = True,
        show_wrist_cam: bool = True,
    ):
        self.xml_path  = xml_path
        self.model     = mujoco.MjModel.from_xml_path(xml_path)
        self.data      = mujoco.MjData(self.model)
        self.dt        = self.model.opt.timestep

        # arm 关节数量 = ctrl 长度里除掉 jaw 的部分
        # 对 so_arm100: ctrl.shape = (6,), 其中最后一位是 Jaw, 但为与原 push_demo 兼容,
        # 我们把「arm_qpos」定义为长度 6 (前 6 个 qpos/ctrl).
        self.arm_dim   = 6
        self.has_jaw_ctrl = self.data.ctrl.shape[0] >= 7

        # 手腕相机
        self.show_wrist_cam = show_wrist_cam and wrist_cam_name is not None
        self.cam_id = -1
        self.renderer = None
        if self.show_wrist_cam:
            cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, wrist_cam_name
            )
            if cam_id < 0:
                print(f"[QposControl] WARN: camera '{wrist_cam_name}' not found, "
                      f"disabling wrist-cam window")
                self.show_wrist_cam = False
            else:
                self.cam_id   = cam_id
                self.renderer = mujoco.Renderer(self.model, height=CAM_H, width=CAM_W)
                cv2.namedWindow("Wrist Camera", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Wrist Camera", CAM_W, CAM_H)

        # viewer
        self._launch_viewer = launch_viewer
        self.viewer = None
        if launch_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance  = VIEW_CAM_DISTANCE
            self.viewer.cam.azimuth   = VIEW_CAM_AZIMUTH
            self.viewer.cam.elevation = VIEW_CAM_ELEVATION
            self.viewer.cam.lookat[:] = VIEW_CAM_LOOKAT

        self._cam_counter = 0
        self._last_info   = ""

    # ─────────────── 基本状态 ───────────────

    @property
    def is_running(self) -> bool:
        if self.viewer is None:
            return True
        return self.viewer.is_running()

    def get_qpos(self) -> np.ndarray:
        """返回当前 arm 的 6 维 qpos 副本。"""
        return self.data.qpos[:self.arm_dim].copy()

    def get_ctrl(self) -> np.ndarray:
        return self.data.ctrl.copy()

    # ─────────────── 控制 ───────────────

    def set_qpos(self, arm_qpos: np.ndarray, jaw: float = 0.0):
        """
        写入控制器目标 (不 step)。
            arm_qpos: 长度 6 的数组
            jaw:      夹爪目标 (仅当 ctrl.shape[0] >= 7 时写入)
        """
        arm_qpos = np.asarray(arm_qpos, dtype=float).reshape(-1)
        assert arm_qpos.shape[0] == self.arm_dim, \
            f"arm_qpos 长度应为 {self.arm_dim}, 实际 {arm_qpos.shape[0]}"
        self.data.ctrl[:self.arm_dim] = arm_qpos
        if self.has_jaw_ctrl:
            self.data.ctrl[self.arm_dim] = jaw

    def reset_qpos(self, arm_qpos: np.ndarray, jaw: float = 0.0, forward: bool = True):
        """
        瞬间将 arm 的 qpos 重置为给定值 (同时同步 ctrl), 并 mj_forward.
        用于任务初始化, 与 set_qpos+step 的渐进式不同。
        """
        arm_qpos = np.asarray(arm_qpos, dtype=float).reshape(-1)
        self.data.qpos[:self.arm_dim] = arm_qpos
        self.set_qpos(arm_qpos, jaw)
        if forward:
            mujoco.mj_forward(self.model, self.data)

    # ─────────────── 仿真推进 ───────────────

    def step(self, info: str = ""):
        """
        推进一个仿真步, 同步 viewer, 按节流刷手腕相机窗口。
        不做 ctrl 写入 - 调用方需先 set_qpos。
        """
        mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        self._cam_counter += 1
        if (self.show_wrist_cam
                and self.renderer is not None
                and self._cam_counter % CAM_EVERY_N == 0):
            mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(self.data, camera=self.cam_id)
            frame = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
            text = info or self._last_info
            for i, line in enumerate(str(text).split("\n")):
                cv2.putText(frame, line, (8, 20 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("Wrist Camera", frame)
            cv2.waitKey(1)

        self._last_info = info

    def move_qpos_linear(
        self,
        q_to: np.ndarray,
        duration: float,
        q_from: Optional[np.ndarray] = None,
        jaw_from: float = 0.0,
        jaw_to: float = 0.0,
        info: str = "",
        realtime: bool = True,
    ) -> bool:
        """
        关节空间线性插值从 q_from 到 q_to, 时长 duration 秒。
        q_from 默认取当前 arm_qpos。
        返回 True 表示完成, False 表示 viewer 被关闭中断。
        """
        q_to = np.asarray(q_to, dtype=float).reshape(-1)
        if q_from is None:
            q_from = self.get_qpos()
        q_from = np.asarray(q_from, dtype=float).reshape(-1)

        steps = max(1, int(duration / self.dt))
        for i in range(steps):
            if not self.is_running:
                return False
            a = (i + 1) / steps
            q_cmd = q_from + a * (q_to - q_from)
            jaw_cmd = jaw_from + a * (jaw_to - jaw_from)
            self.set_qpos(q_cmd, jaw_cmd)
            self.step(info)
            if realtime:
                time.sleep(self.dt)
        return True

    def hold(self, duration: float, info: str = "", realtime: bool = True) -> bool:
        """保持当前 ctrl 一段时间, 推进仿真。"""
        end = time.time() + duration
        while self.is_running and time.time() < end:
            self.step(info)
            if realtime:
                time.sleep(self.dt)
        return self.is_running

    # ─────────────── 关闭 ───────────────

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        if self.show_wrist_cam:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass
            self.show_wrist_cam = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ─────────────────────── Demo ───────────────────────

def _demo():
    """独立运行时: 让机械臂从 home 扫动到几个示例 qpos, 展示底层控制。"""
    home_qpos = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0])
    poses = [
        ("home",    home_qpos,                                 1.0),
        ("tilt_L",  np.array([ 0.8, -1.4, 1.5, 1.3, -1.57, 0.0]), 1.5),
        ("tilt_R",  np.array([-0.8, -1.4, 1.5, 1.3, -1.57, 0.0]), 1.5),
        ("reach",   np.array([ 0.0, -2.2, 1.8, 1.5, -1.57, 0.0]), 1.5),
        ("home",    home_qpos,                                 1.2),
    ]

    with QposControl() as robot:
        robot.reset_qpos(home_qpos)
        robot.hold(0.5, info="READY")
        for name, q, dur in poses:
            if not robot.is_running:
                break
            print(f"[demo] move -> {name}")
            robot.move_qpos_linear(q, duration=dur, info=f"DEMO {name}")
        robot.hold(1.0, info="DONE")

def demo_hold_action(action: Optional[np.ndarray] = None):
    """
    输入一个 6 维 action（qpos），机械臂移动过去并保持在那里。

    用法：
        demo_hold_action(np.array([...]))
    或运行时输入：
        mjpython qpos_control.py
        然后在终端输入 6 个值
    """

    # 默认 home
    home_qpos = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0])

    # 如果没传 action，就让用户输入
    action = np.array([0.0, -1.57, -1.57, 1.57, -1.57, 0.0])

    action = np.asarray(action, dtype=float).reshape(-1)
    assert action.shape[0] == 6, "action 必须是 6 维"

    with QposControl() as robot:
        # reset 到 home
        robot.reset_qpos(home_qpos)
        robot.hold(0.5, info="RESET -> HOME")

        # move 到目标
        print(f"[demo] move to action: {action}")
        ok = robot.move_qpos_linear(
            q_to=action,
            duration=2.0,
            info="MOVING TO TARGET"
        )

        if not ok:
            return

        # 一直保持
        print("[demo] holding position...")
        robot.hold(9999, info="HOLDING TARGET")
if __name__ == "__main__":
    demo_hold_action()

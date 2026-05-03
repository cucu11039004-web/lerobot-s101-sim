"""Qpos control for moving the arm in the MuJoCo scene while keeping cameras on."""

from pathlib import Path
from typing import Optional

import cv2
import mujoco
import mujoco.viewer
import numpy as np


# ─────────────────────── 默认配置 ───────────────────────

DEFAULT_XML_PATH   = str(Path(__file__).resolve().with_name("push_scene.xml"))
DEFAULT_WRIST_CAM  = "wrist_cam"

CAM_W, CAM_H       = 640, 480
CAM_EVERY_N        = 3        # 每 N 个仿真步刷一次手腕相机窗口

# 相机面板的默认视角
VIEW_CAM_DISTANCE  = 0.7
VIEW_CAM_AZIMUTH   = 135
VIEW_CAM_ELEVATION = -35
VIEW_CAM_LOOKAT    = (0.235, -0.01, 0.05)


class QposControl:
    """Minimal qpos controller with viewer and wrist camera."""

    def __init__(
        self,
        xml_path: str = DEFAULT_XML_PATH,
        wrist_cam_name: Optional[str] = DEFAULT_WRIST_CAM,
        launch_viewer: bool = True,
        show_wrist_cam: bool = True,
    ):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.qpos_dim = 6

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
                self.cam_id = cam_id
                self.renderer = mujoco.Renderer(self.model, height=CAM_H, width=CAM_W)
                cv2.namedWindow("Wrist Camera", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Wrist Camera", CAM_W, CAM_H)

        self.viewer = None
        if launch_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = VIEW_CAM_DISTANCE
            self.viewer.cam.azimuth = VIEW_CAM_AZIMUTH
            self.viewer.cam.elevation = VIEW_CAM_ELEVATION
            self.viewer.cam.lookat[:] = VIEW_CAM_LOOKAT

        self._cam_counter = 0

    @property
    def is_running(self) -> bool:
        if self.viewer is None:
            return True
        return self.viewer.is_running()

    def get_qpos(self) -> np.ndarray:
        return self.data.qpos[:self.qpos_dim].copy()

    def set_qpos(self, qpos: np.ndarray) -> None:
        """Set the 6-dim control target."""
        qpos = np.asarray(qpos, dtype=float).reshape(-1)
        if qpos.shape[0] != self.qpos_dim:
            raise ValueError(f"qpos length must be {self.qpos_dim}, got {qpos.shape[0]}")
        self.data.ctrl[:self.qpos_dim] = qpos

    def reset_qpos(self, qpos: np.ndarray) -> None:
        """Teleport the arm to a qpos and sync control target."""
        qpos = np.asarray(qpos, dtype=float).reshape(-1)
        if qpos.shape[0] != self.qpos_dim:
            raise ValueError(f"qpos length must be {self.qpos_dim}, got {qpos.shape[0]}")
        self.data.qpos[:self.qpos_dim] = qpos
        self.set_qpos(qpos)
        mujoco.mj_forward(self.model, self.data)

    def step(self, info: str = "") -> None:
        mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        self._cam_counter += 1
        if self.show_wrist_cam and self.renderer is not None and self._cam_counter % CAM_EVERY_N == 0:
            self.renderer.update_scene(self.data, camera=self.cam_id)
            frame = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
            for i, line in enumerate(str(info).split("\n")):
                cv2.putText(
                    frame, line, (8, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                )
            cv2.imshow("Wrist Camera", frame)
            cv2.waitKey(1)

    def run_steps(self, steps: int, info: str = "") -> bool:
        for _ in range(steps):
            if not self.is_running:
                return False
            self.step(info)
        return True

    def move_to(self, qpos: np.ndarray, steps: int = 200, info: str = "") -> bool:
        qpos = np.asarray(qpos, dtype=float).reshape(-1)
        q0 = self.get_qpos()
        for i in range(steps):
            alpha = (i + 1) / max(steps, 1)
            self.set_qpos(q0 + alpha * (qpos - q0))
            if not self.run_steps(1, info=info):
                return False
        return True

    def close(self) -> None:
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


def demo_hold_action(action: Optional[np.ndarray] = None) -> None:
    home_qpos = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0])
    if action is None:
        action = np.array([0.0, -2.2, 1.8, 1.5, -1.57, 0.0])
    action = np.asarray(action, dtype=float).reshape(-1)
    if action.shape[0] != 6:
        raise ValueError("action must be 6-dim")

    with QposControl() as robot:
        robot.reset_qpos(home_qpos)
        robot.run_steps(100, info="HOME")
        robot.move_to(action, steps=200, info="MOVE")
        robot.run_steps(1000000, info="HOLD")


if __name__ == "__main__":
    demo_hold_action()

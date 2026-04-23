"""
pos_control.py — SO-ARM100 末端位姿 (xyz + rpy) 控制

依赖: qpos_control.py
"""

from typing import Optional, Tuple
import mujoco
import numpy as np
from qpos_control import QposControl

DEFAULT_FIXED_TIP_GEOM  = "fixed_jaw_pad_1"
DEFAULT_MOVING_TIP_GEOM = "moving_jaw_pad_1"
DEFAULT_TIP_BODY        = "Fixed_Jaw"

IK_DAMPING=0.06; IK_MAX_ITERS=800; IK_STEP_SCALE=0.3
IK_TOL_POS=5e-4; IK_ACCEPT_POS=0.01; IK_ACCEPT_LO=0.03
IK_RANDOM_RESTARTS=3; IK_RANDOM_STD=0.15

POSE_RANDOM_SEEDS=300; POSE_REFINE_ITERS=400
POSE_REFINE_STEP=0.3; POSE_REFINE_DAMP=0.05
POSE_POS_W=1.0; POSE_ORI_W=0.15
POSE_ACCEPT_POS=0.015; POSE_ACCEPT_ORI=0.20
POSE_TOL_POS=5e-4; POSE_TOL_ORI=1e-2

SETTLE_TOL=0.012; SETTLE_TIMEOUT=2.5


def rpy_to_mat(rpy):
    r,p,y=float(rpy[0]),float(rpy[1]),float(rpy[2])
    cr,sr=np.cos(r),np.sin(r); cp,sp=np.cos(p),np.sin(p); cy,sy=np.cos(y),np.sin(y)
    return np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]]) @ \
           np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]]) @ \
           np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])

def mat_to_rpy(R):
    sy=np.sqrt(R[0,0]**2+R[1,0]**2)
    if sy>1e-6:
        return np.array([np.arctan2(R[2,1],R[2,2]),np.arctan2(-R[2,0],sy),np.arctan2(R[1,0],R[0,0])])
    return np.array([np.arctan2(-R[1,2],R[1,1]),np.arctan2(-R[2,0],sy),0.0])

def mat_to_quat(R):
    tr=R[0,0]+R[1,1]+R[2,2]
    if tr>0:
        s=0.5/np.sqrt(tr+1.0)
        return np.array([0.25/s,(R[2,1]-R[1,2])*s,(R[0,2]-R[2,0])*s,(R[1,0]-R[0,1])*s])
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=2.0*np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])
        return np.array([(R[2,1]-R[1,2])/s,0.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s])
    elif R[1,1]>R[2,2]:
        s=2.0*np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])
        return np.array([(R[0,2]-R[2,0])/s,(R[0,1]+R[1,0])/s,0.25*s,(R[1,2]+R[2,1])/s])
    else:
        s=2.0*np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])
        return np.array([(R[1,0]-R[0,1])/s,(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,0.25*s])

def quat_err_rotvec(q_cur,q_des):
    if np.dot(q_cur,q_des)<0: q_des=-q_des
    x=-q_des[0]*q_cur[1]+q_des[1]*q_cur[0]+q_des[2]*q_cur[3]-q_des[3]*q_cur[2]
    y=-q_des[0]*q_cur[2]-q_des[1]*q_cur[3]+q_des[2]*q_cur[0]+q_des[3]*q_cur[1]
    z=-q_des[0]*q_cur[3]+q_des[1]*q_cur[2]-q_des[2]*q_cur[1]+q_des[3]*q_cur[0]
    return 2.0*np.array([x,y,z])


class PosControl(QposControl):

    def __init__(self,fixed_tip_geom=DEFAULT_FIXED_TIP_GEOM,
                 moving_tip_geom=DEFAULT_MOVING_TIP_GEOM,
                 tip_body=DEFAULT_TIP_BODY,
                 rng=None,**qpos_kwargs):
        super().__init__(**qpos_kwargs)
        g1=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,fixed_tip_geom)
        g2=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,moving_tip_geom)
        if g1<0 or g2<0: raise RuntimeError("tip geom 未找到")
        self.g1,self.g2=g1,g2
        b=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,tip_body)
        if b<0: b=self.model.geom_bodyid[g1]
        self.tip_body_id=b
        print(f"[PosControl] tip_body='{tip_body}' id={b}")
        self.rng=rng if rng is not None else np.random.default_rng()
        self._default_seed=np.array([0.0,-1.2,1.8,1.8,-1.57,0.0])

    def tip_mid(self):
        return 0.5*(self.data.geom_xpos[self.g1]+self.data.geom_xpos[self.g2])
    def get_ee_xyz(self): return self.tip_mid().copy()
    def get_ee_rot(self): return self.data.xmat[self.tip_body_id].reshape(3,3).copy()
    def get_ee_rpy(self): return mat_to_rpy(self.get_ee_rot())

    def _split_pose_input(self,xyz,rpy=None):
        pose=np.asarray(xyz,dtype=float).reshape(-1)
        if pose.shape[0]==6:
            if rpy is not None: raise ValueError("xyz 已是 6D")
            return pose[:3].copy(),pose[3:].copy()
        if pose.shape[0]!=3: raise ValueError("xyz 应 3 或 6 维")
        if rpy is None: return pose.copy(),None
        return pose.copy(),np.asarray(rpy,dtype=float).reshape(3).copy()

    def set_ik_seed(self,seed):
        self._default_seed=np.asarray(seed,dtype=float).reshape(-1).copy()

    def _pos_ik_single(self,target_xyz,seed,jaw=0.0):
        has_jaw=self.data.ctrl.shape[0]>=7
        backup=self.data.qpos.copy()
        best_q,best_err=seed.copy(),np.inf
        for noise in (0.0,0.1,0.2):
            q=seed+self.rng.normal(0,noise,self.arm_dim) if noise else seed.copy()
            err_norm=np.inf
            for it in range(IK_MAX_ITERS):
                self.data.qpos[:self.arm_dim]=q
                if has_jaw: self.data.qpos[self.arm_dim]=jaw
                mujoco.mj_forward(self.model,self.data)
                err=target_xyz-self.tip_mid(); err_norm=float(np.linalg.norm(err))
                if err_norm<IK_TOL_POS: break
                j1=np.zeros((3,self.model.nv)); j2=np.zeros((3,self.model.nv))
                mujoco.mj_jacGeom(self.model,self.data,j1,np.zeros((3,self.model.nv)),self.g1)
                mujoco.mj_jacGeom(self.model,self.data,j2,np.zeros((3,self.model.nv)),self.g2)
                J=0.5*(j1[:,:self.arm_dim]+j2[:,:self.arm_dim])
                d=IK_DAMPING*(2.0 if it>IK_MAX_ITERS//2 else 0.5 if err_norm>0.1 else 1.0)
                dq=J.T@np.linalg.solve(J@J.T+(d**2)*np.eye(3),err)
                q=q+min(IK_STEP_SCALE,err_norm*0.5)*dq
                for j in range(self.arm_dim):
                    lo,hi=self.model.jnt_range[j]; m=(hi-lo)*0.05
                    q[j]=np.clip(q[j],lo+m,hi-m)
            if err_norm<best_err: best_err,best_q=err_norm,q.copy()
        self.data.qpos[:]=backup; mujoco.mj_forward(self.model,self.data)
        return best_q,best_err

    def _solve_pos_ik(self,target_xyz,seed):
        has_jaw=self.data.ctrl.shape[0]>=7
        jaws=np.linspace(0.01,0.05,7) if has_jaw else [0.0]
        jnt_lo=self.model.jnt_range[:self.arm_dim,0]
        jnt_hi=self.model.jnt_range[:self.arm_dim,1]
        best_q,best_err=seed.copy(),np.inf
        for jaw in jaws:
            q,e=self._pos_ik_single(target_xyz,seed,jaw)
            if e<best_err: best_err,best_q=e,q
            if e<IK_ACCEPT_POS: return q,e
        for off in ([.01,0,0],[-.01,0,0],[0,.01,0],[0,-.01,0],[0,0,.02],[0,0,-.02]):
            q,e=self._pos_ik_single(target_xyz+np.array(off),seed)
            if e<IK_ACCEPT_POS: return q,e
            if e<best_err: best_err,best_q=e,q
        for _ in range(IK_RANDOM_RESTARTS):
            rs=np.clip(seed+self.rng.normal(0,IK_RANDOM_STD,self.arm_dim),jnt_lo,jnt_hi)
            for jaw in jaws:
                q,e=self._pos_ik_single(target_xyz,rs,jaw)
                if e<IK_ACCEPT_POS: return q,e
                if e<best_err: best_err,best_q=e,q
        return (best_q,best_err) if best_err<IK_ACCEPT_LO else None

    def _pose_cost(self,target_xyz,target_quat):
        pe=float(np.linalg.norm(target_xyz-self.tip_mid()))
        oe=float(np.linalg.norm(quat_err_rotvec(
            mat_to_quat(self.data.xmat[self.tip_body_id].reshape(3,3)),target_quat)))
        return POSE_POS_W*pe+POSE_ORI_W*oe,pe,oe

    def _refine_pose(self,q_init,target_xyz,target_quat,jaw=0.0):
        has_jaw=self.data.ctrl.shape[0]>=7
        backup=self.data.qpos.copy()
        n=self.arm_dim
        jnt_lo=self.model.jnt_range[:n,0]; jnt_hi=self.model.jnt_range[:n,1]
        q=q_init.copy()
        best_q,best_cost,best_pe,best_oe=q.copy(),np.inf,np.inf,np.inf
        for it in range(POSE_REFINE_ITERS):
            self.data.qpos[:n]=q
            if has_jaw: self.data.qpos[n]=jaw
            mujoco.mj_forward(self.model,self.data)
            cost,pe,oe=self._pose_cost(target_xyz,target_quat)
            if cost<best_cost: best_cost,best_pe,best_oe,best_q=cost,pe,oe,q.copy()
            if pe<POSE_TOL_POS and oe<POSE_TOL_ORI: break
            j1=np.zeros((3,self.model.nv)); j2=np.zeros((3,self.model.nv))
            mujoco.mj_jacGeom(self.model,self.data,j1,np.zeros((3,self.model.nv)),self.g1)
            mujoco.mj_jacGeom(self.model,self.data,j2,np.zeros((3,self.model.nv)),self.g2)
            Jp=0.5*(j1[:,:n]+j2[:,:n])
            jacr=np.zeros((3,self.model.nv))
            mujoco.mj_jacBody(self.model,self.data,None,jacr,self.tip_body_id)
            Jr=jacr[:,:n]
            pev=target_xyz-self.tip_mid()
            oev=quat_err_rotvec(mat_to_quat(self.data.xmat[self.tip_body_id].reshape(3,3)),target_quat)
            err6=np.concatenate([POSE_POS_W*pev,POSE_ORI_W*oev])
            J6=np.vstack([POSE_POS_W*Jp,POSE_ORI_W*Jr])
            dq=J6.T@np.linalg.solve(J6@J6.T+(POSE_REFINE_DAMP**2)*np.eye(6),err6)
            step=min(POSE_REFINE_STEP,float(np.linalg.norm(err6))*0.5)
            q=q+step*dq
            q=np.clip(q,jnt_lo+(jnt_hi-jnt_lo)*0.05,jnt_hi-(jnt_hi-jnt_lo)*0.05)
        self.data.qpos[:]=backup; mujoco.mj_forward(self.model,self.data)
        return best_q,best_pe,best_oe

    def _solve_pose_ik(self,target_xyz,target_rot,seed):
        target_quat=mat_to_quat(target_rot)
        has_jaw=self.data.ctrl.shape[0]>=7
        jaws=np.linspace(0.01,0.05,3) if has_jaw else [0.0]
        jnt_lo=self.model.jnt_range[:self.arm_dim,0]
        jnt_hi=self.model.jnt_range[:self.arm_dim,1]
        backup=self.data.qpos.copy()

        # 随机粗搜索
        candidates=[]
        all_seeds=[seed.copy()]+[self.rng.uniform(jnt_lo,jnt_hi)
                                  for _ in range(POSE_RANDOM_SEEDS-1)]
        for s in all_seeds:
            self.data.qpos[:self.arm_dim]=s
            mujoco.mj_forward(self.model,self.data)
            cost,pe,oe=self._pose_cost(target_xyz,target_quat)
            if pe<0.06: candidates.append((cost,pe,oe,s.copy()))
        self.data.qpos[:]=backup; mujoco.mj_forward(self.model,self.data)

        candidates.sort(key=lambda x:x[0])

        # 加入纯位置 IK 结果
        pr=self._solve_pos_ik(target_xyz,seed)
        if pr: candidates.append((np.inf,np.inf,np.inf,pr[0]))

        # 精化 top-15
        best_q,best_pe,best_oe=seed.copy(),np.inf,np.inf
        for _,_,_,qi in candidates[:15]:
            for jaw in jaws:
                q,pe,oe=self._refine_pose(qi,target_xyz,target_quat,jaw)
                if pe<best_pe or (pe<POSE_ACCEPT_POS and oe<best_oe):
                    best_q,best_pe,best_oe=q.copy(),pe,oe
                if pe<POSE_ACCEPT_POS and oe<POSE_ACCEPT_ORI:
                    return best_q,best_pe,best_oe
        return best_q,best_pe,best_oe

    def solve_ik(self,target_xyz,seed=None,rpy=None):
        if seed is None: seed=self._default_seed.copy()
        seed=np.asarray(seed,dtype=float).reshape(-1).copy()
        if rpy is None: return self._solve_pos_ik(target_xyz,seed)
        target_rot=rpy_to_mat(np.asarray(rpy,dtype=float).reshape(3))
        q,pe,oe=self._solve_pose_ik(target_xyz,target_rot,seed)
        print(f"  [IK] pos={pe:.4f}m  ori={oe:.4f}rad  "
              f"({'✓' if pe<POSE_ACCEPT_POS else '✗pos'} "
              f"{'✓' if oe<POSE_ACCEPT_ORI else '✗ori'})")
        if pe>IK_ACCEPT_LO:
            fb=self._solve_pos_ik(target_xyz,seed)
            return fb
        return q,pe

    def set_ee_pose(self,xyz,rpy=None,seed=None,jaw=0.0):
        tx,tr=self._split_pose_input(xyz,rpy)
        r=self.solve_ik(tx,seed=seed,rpy=tr)
        if r is None: return None
        q,err=r; self.reset_qpos(q,jaw=jaw,forward=True); return err

    def move_ee_to(self,xyz,rpy=None,duration=1.0,seed=None,
                   jaw_from=0.0,jaw_to=0.0,info="",realtime=True,settle=True):
        tx,tr=self._split_pose_input(xyz,rpy)
        if seed is None: seed=self.get_qpos()
        r=self.solve_ik(tx,seed=seed,rpy=tr)
        if r is None:
            print(f"[PosControl] IK 失败 target={tx}"); return None
        q_target,_=r
        ok=self.move_qpos_linear(q_to=q_target,duration=duration,
            q_from=self.get_qpos(),jaw_from=jaw_from,jaw_to=jaw_to,
            info=info,realtime=realtime)
        if not ok: return None
        if settle:
            import time as _time
            t0=_time.time()
            while self.is_running:
                self.set_qpos(q_target,jaw_to if self.has_jaw_ctrl else 0.0)
                self.step(f"{info} settling")
                if realtime: _time.sleep(self.dt)
                if np.linalg.norm(self.tip_mid()-tx)<SETTLE_TOL: break
                if _time.time()-t0>SETTLE_TIMEOUT: break
            if not self.is_running: return None
        return self.tip_mid()


# ─────────────────────── Demo ───────────────────────

def _demo():
    WS_CENTER=np.array([0.235,-0.01]); Z_HI,Z_LO=0.12,0.05
    corners=[[WS_CENTER[0]-0.03,WS_CENTER[1]-0.03,Z_HI],
             [WS_CENTER[0]+0.03,WS_CENTER[1]-0.03,Z_HI],
             [WS_CENTER[0]+0.03,WS_CENTER[1]+0.03,Z_HI],
             [WS_CENTER[0]-0.03,WS_CENTER[1]+0.03,Z_HI],
             [WS_CENTER[0]-0.03,WS_CENTER[1]-0.03,Z_LO],
             [WS_CENTER[0]-0.03,WS_CENTER[1]-0.03,Z_HI]]
    home_qpos=np.array([1.6,-2.2,1.8,1.5,-1.5,0.0])
    with PosControl() as robot:
        robot.reset_qpos(home_qpos); robot.hold(0.5,info="READY")
        for i,xyz in enumerate(corners):
            if not robot.is_running: break
            res=robot.move_ee_to(np.array(xyz),duration=1.2,info=f"corner {i}")
            if res is None: break
            print(f"  corner {i} tip={res.round(4)}")
        robot.hold(1.0,info="DONE")


def demo_pose_control():
    """
    姿态控制 demo。

    工作原理:
      先到达目标位置, 读出"自然姿态" (natural_rpy)。
      再以自然姿态为基准, 演示 Wrist_Roll 能控制的方向。

    SO-ARM100 在给定位置的姿态自由度约 1 个 (Wrist_Roll),
    主要影响 rpy[0] (roll), 同时轻微影响 rpy[2] (yaw)。
    """
    home_qpos=np.array([0,-1.57,1.57,1.57,-1.57,0])
    target_xyz=np.array([0.30, 0.0, 0.25])   # 姿态自由度最大区域

    with PosControl() as robot:
        robot.reset_qpos(home_qpos); robot.hold(0.5,info="READY")

        # 先到达位置, 读自然姿态
        res=robot.move_ee_to(target_xyz,duration=1.5,info="goto_pos")
        if res is None: print("位置 IK 失败"); return
        natural=robot.get_ee_rpy()
        print(f"\n[info] 自然姿态 rpy={natural.round(4)}")
        print(f"[info] 以自然姿态为基准演示 Wrist_Roll 控制\n")
        robot.hold(1.0,info="natural")

        # 以自然姿态为基准, 在 roll 方向演示 ±1.2rad
        # (Wrist_Roll 范围 ±2.79rad, 减去位置消耗后约剩 ±1.5rad)
        deltas=[
            ("自然",   np.array([0.0, 0.0, 0.0])),
            ("roll+1.2", np.array([+1.2, 0.0, 0.0])),
            ("roll-1.2", np.array([-1.2, 0.0, 0.0])),
            ("roll+0.6", np.array([+0.6, 0.0, 0.0])),
            ("回中",   np.array([0.0, 0.0, 0.0])),
        ]

        for name,drpy in deltas:
            if not robot.is_running: break
            target_rpy=natural+drpy
            print(f"[pose] {name}: 目标rpy={target_rpy.round(3)}")
            res=robot.move_ee_to(target_xyz,rpy=target_rpy,
                                 duration=1.5,info=name)
            if res is None: print("  失败或 viewer 关闭"); break
            actual=robot.get_ee_rpy()
            err=float(np.linalg.norm(actual-target_rpy))
            print(f"  actual={actual.round(4)}  Δrpy={err:.3f}rad")
            robot.hold(0.8,info=name)

        robot.hold(2.0,info="DONE")


if __name__ == "__main__":
    demo_pose_control()
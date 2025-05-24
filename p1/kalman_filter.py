import pinocchio as pin
import numpy as np

class LinearKFPositionVelocityEstimator:
    def __init__(self, dt, rotation, lin_acc, ang_vel, qpos, qvel):
        # 初始化变量和状态矩阵
        self.dt = dt  # 采样时间间隔
        # 状态向量和协方差矩阵初始化
        self._ps = np.zeros((6,))  # 位置误差
        self._vs = np.zeros((6,))  # 速度误差
        self._A = np.zeros((12, 12))  # 状态转移矩阵
        self._B = np.zeros((12, 3))  # 控制输入矩阵
        self._C = np.zeros((14, 12))  # 观测矩阵
        self._Q0 = np.eye(12)  # 过程噪声协方差初始矩阵
        self._R0 = np.eye(14)  # 测量噪声协方差初始矩阵
        # 传感器和模型参数
        self.rotation = rotation  # 旋转矩阵，从传感器到世界坐标系
        self.lin_acc = lin_acc  # 线性加速度
        self.ang_vel = ang_vel  # 角速度
        self.qpos = qpos  # 关节位置
        self.qvel = qvel  # 关节速度
        self.phase = np.zeros(2)  # 用于追踪接触相位
        self.setup()  # 初始化矩阵A, B, C


    def initialize(self, urdfmodel, q):
        # 初始化状态和URDF模型数据
        self._P = np.eye(12) * 100  # 状态估计的协方差
        self.urdfmodel = urdfmodel  # URDF模型
        self.urdfdata = self.urdfmodel.createData()  # 创建模型数据
        # 初始化状态估计
        self._xhat = np.zeros((12,))
        self._xhat[:3] = np.array([0, 0, 0.624990685])  # 初始位置估计
        Rbod = self.rotation  # 传感器到世界坐标系的旋转矩阵
        leg_frame_ids = [urdfmodel.getFrameId("foot_L"), urdfmodel.getFrameId("foot_R")]  # 腿部标识
        p_f = np.zeros(6)
        for i in range(2):
            # 计算脚的初始位置
            p_rel = self.compute_foot_position(np.concatenate([np.array([0, 0, 0.624990685]), np.array([1, 0, 0, 0]), q]), leg_frame_ids[i])
            p_f[3*i:3*i+3] = Rbod @ p_rel  # 将脚的位置转换到世界坐标系
        self._xhat[6:] = p_f  # 更新状态向量


    def setup(self):
        # 配置A, B, C矩阵以及噪声矩阵
        dt = self.dt
        self._A[:3, :3] = np.eye(3)  # 位置更新部分
        self._A[:3, 3:6] = dt * np.eye(3)  # 速度对位置的影响
        self._A[3:6, 3:6] = np.eye(3)  # 速度更新部分
        self._A[6:, 6:] = np.eye(6)  # 脚位置更新部分
        self._B[3:6, :3] = dt * np.eye(3)  # 加速度对速度的影响
        # 观测矩阵的设置
        C1 = np.hstack((np.eye(3), np.zeros((3, 3))))
        C2 = np.hstack((np.zeros((3, 3)), np.eye(3)))
        for i in range(2):
            self._C[i * 3:(i + 1) * 3, :6] = C1
            self._C[6 + i * 3:9 + i * 3, :6] = C2
        self._C[:6, 6:] = -np.eye(6)
        self._C[12:, 6:] = np.array([[0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1]])
        # 调整噪声协方差矩阵
        self._Q0[:3, :3] = (dt / 20.0) * np.eye(3)
        self._Q0[3:6, 3:6] = (dt * 9.8 / 20.0) * np.eye(3)
        self._Q0[6:, 6:] = dt * np.eye(6)

    def compute_foot_position(self, q, leg_frame_id):
        pin.forwardKinematics(self.urdfmodel, self.urdfdata, q)  # 计算正向运动学
        pin.updateFramePlacements(self.urdfmodel, self.urdfdata)  # 更新帧的位置
        foot_pos_world = self.urdfdata.oMf[leg_frame_id].translation  # 获取脚部在世界坐标系中的位置

        # 获取body在世界坐标系中的变换矩阵
        body_to_world = self.urdfdata.oMf[self.urdfmodel.getFrameId("base_Link")]

        # 计算脚部相对于body坐标系的位置
        foot_pos_body = body_to_world.inverse().act(foot_pos_world)

        return foot_pos_body

    def compute_jacobian(self, q, leg_frame_id):
        pin.computeJointJacobians(self.urdfmodel, self.urdfdata, q) # 计算关节雅可比矩阵
        pin.framesForwardKinematics(self.urdfmodel, self.urdfdata, q) # 计算帧的正向运动学
        J = pin.getFrameJacobian(self.urdfmodel, self.urdfdata, leg_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED) # 获取帧的雅可比矩阵
        return J[:3, 6:]  # 只取出位置部分

    def run(self, mujoco_data):
        # 执行卡尔曼滤波更新步骤，主要用于估计位置和速度
        # 定义过程噪声和测量噪声的大小
        process_noise_pimu = 0.002
        process_noise_vimu = 0.002
        process_noise_pfoot = 0.00000002
        sensor_noise_pimu_rel_foot = 0.00000002
        sensor_noise_vimu_rel_foot = 0.00000002
        sensor_noise_zfoot = 0.00000002

        # 初始化过程噪声协方差矩阵Q
        Q = np.eye(12)
        Q[:3, :3] = self._Q0[:3, :3] * process_noise_pimu  # 位置的过程噪声
        Q[3:6, 3:6] = self._Q0[3:6, 3:6] * process_noise_vimu  # 速度的过程噪声
        Q[6:, 6:] = self._Q0[6:, 6:] * process_noise_pfoot  # 脚部位置的过程噪声

        # 初始化测量噪声协方差矩阵R
        R = np.eye(14)
        R[:6, :6] = self._R0[:6, :6] * sensor_noise_pimu_rel_foot  # 传感器位置相对于脚部的噪声
        R[6:12, 6:12] = self._R0[6:12, 6:12] * sensor_noise_vimu_rel_foot  # 传感器速度相对于脚部的噪声
        R[12:, 12:] = self._R0[12:, 12:] * sensor_noise_zfoot  # 脚部Z位置的噪声

        # 重力加速度向量
        g = np.array([0, 0, -9.81])
        # 从传感器帧到世界帧的旋转矩阵
        Rbod = self.rotation
        # 根据传感器数据计算世界坐标系下的加速度
        a = Rbod@mujoco_data['sensordata'][:3] + g

        # 初始化观测误差存储数组和信任度数组
        pzs = np.zeros(2)
        trusts = np.zeros(2)

        # 获取当前估计的位置和速度
        p0 = self._xhat[:3]
        v0 = self._xhat[3:6]

        # 获取两脚的URDF模型标识
        leg_frame_ids = [self.urdfmodel.getFrameId("foot_L"), self.urdfmodel.getFrameId("foot_R")]

        # 遍历每个脚部进行状态更新
        for i in range(2):
            q = mujoco_data["qpos"]
            qd = self.qvel  # 关节角速度
            # 计算当前脚的位置和速度的雅可比矩阵
            J = self.compute_jacobian(q, leg_frame_ids[i])
            # 计算当前脚的位置
            p_rel= self.compute_foot_position(q, leg_frame_ids[i])
            # 计算当前脚的速度
            dp_rel = self.compute_foot_velocity(J, qd)
            # 转换到世界坐标系下的脚位置和速度
            p_f = self._xhat[:3]+Rbod @ p_rel
            dp_f = -Rbod @ (np.cross(mujoco_data['sensordata'][3:6], p_rel) + dp_rel)

            # 更新状态向量和观测矩阵的索引
            qindex = 6 + 3 * i
            rindex1 = 3 * i
            rindex2 = 6 + 3 * i
            rindex3 = 12 + i

            # 使用脚触地信息更新相位变量
            contact = mujoco_data['contact_info'][i]
            if contact:
                self.phase[i] += 0.001 # 计算触地持续时间，每次检查间隔为1ms
            else:
                self.phase[i] = 0  # 没有触地时重置相位

            # 计算信任度，用于调整噪声矩阵
            trust_window = 0.4  # 通过改变这个值来调整信任度的计算
            if self.phase[i] / 0.015 > 1:
                self.phase[i] = 0.015
            if self.phase[i] < trust_window * 0.015:
                trust = self.phase[i] / trust_window * 0.015
            elif self.phase[i] > (1.0 - trust_window) * 0.015:
                trust = (1.0 * 0.015 - self.phase[i]) / trust_window * 0.015
            else:
                trust = 1.0

            # 使用信任度调整噪声矩阵
            high_suspect_number = 1000000.0
            Q[qindex:qindex + 3, qindex:qindex + 3] *= (1 + (1 - trust) * high_suspect_number)
            R[rindex1:rindex1 + 3, rindex1:rindex1 + 3] *= (1 + (1 - trust) * high_suspect_number)
            R[rindex2:rindex2 + 3, rindex2:rindex2 + 3] *= (1 + (1 - trust) * high_suspect_number)
            R[rindex3, rindex3] *= (1 + (1 - trust) * high_suspect_number)

            # 更新观测误差和信任度数组
            trusts[i] = trust
            self._ps[rindex1:rindex1 + 3] = -p_rel
            self._vs[rindex1:rindex1 + 3] = (1.0 - trust) * v0 + trust * (dp_f)
            pzs[i] = (1.0 - trust) * (p0[2] + p_f[2])

        # 合并观测误差向量
        y = np.hstack((self._ps, self._vs, pzs))
        # 根据状态转移矩阵和控制输入更新状态估计
        self._xhat = self._A @ self._xhat + self._B @ a
        # 预测协方差
        Pm = self._A @ self._P @ self._A.T + Q
        # 计算预测的观测
        yModel = self._C @ self._xhat
        # 观测残差
        ey = y - yModel
        # 观测协方差
        S = self._C @ Pm @ self._C.T + R
        S = 0.5 * (S + S.T)  # 确保对称性
        # 计算卡尔曼增益
        K = Pm @ self._C.T @ np.linalg.inv(S)
        # 更新状态估计
        self._xhat += K @ ey
        # 更新协方差估计
        self._P = (np.eye(12) - K @ self._C) @ Pm
        self._P = 0.5 * (self._P + self._P.T)  # 确保对称性

        # 防止漂移
        if np.linalg.det(self._P[:2, :2]) > 1e-6:
            self._P[:2, :2] = self._P[:2, :2] / 10
            self._P[:2, 2:] = 0
            self._P[2:, :2] = 0

        return {
            "position": self._xhat[:3], # 位置估计
            "vWorld": self._xhat[3:6], # 世界坐标系下的速度估计
            "vBody": Rbod @ self._xhat[3:6] # 机体坐标系下的速度估计
        }

    def compute_foot_velocity(self, J, qd):
        """Compute foot velocity given Jacobian and joint velocities."""
        base_frame_id = self.urdfmodel.getFrameId('base_Link')
        base_to_world_transform = self.urdfdata.oMf[base_frame_id]
        base_to_world_rotation = base_to_world_transform.rotation
        world_to_base_rotation = base_to_world_rotation.T
        foot_velocity_base = world_to_base_rotation @ J @ qd
        return foot_velocity_base

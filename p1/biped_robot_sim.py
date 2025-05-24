import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import torch
import os
import time
import hydra
from omegaconf import DictConfig
from kalman_filter import LinearKFPositionVelocityEstimator  # 导入卡尔曼滤波器
import pinocchio as pin
import matplotlib.pyplot as plt

# Set the directory for the simulation environment's source code
SOURCE_DIR = (
    "/home/wbb/catkin_ws/src/final_project/project1_state_estimation/"  # TODO: change to your own project dir
)
# Flag to indicate whether to use the simulation environment
USE_SIM = False  # TODO: you should change to False to implement your own state estimator


class MuJoCoSim:
    """Main class for setting up and running the MuJoCo simulation."""

    model: mujoco.MjModel
    data: mujoco.MjData
    policy: torch.nn.Module

    def __init__(self, cfg, policy_plan):
        """Class constructor to set up simulation configuration and control policy."""
        self.cfg = cfg  # Save environment configuration
        self.policy = policy_plan  # Save control policy
        self.init_qpos__ = np.array([0, 0, 9.24990685e-01])
        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(
            os.path.join(SOURCE_DIR, "xiaotian/urdf/xiaotian.xml")
        )
        self.model.opt.timestep = 0.001
        model = pin.buildModelFromUrdf(os.path.join(SOURCE_DIR, "xiaotian/urdf/xiaotian.urdf"))
        self.urdfmodel = model



        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        # 假设 quat 是从 Mujoco 数据中获取的四元数，格式 [x, y, z, w]
        quat = self.data.sensor("imu_quat").data[[1, 2, 3, 0]].astype(np.double)

        # 创建一个 Rotation 对象
        rotation = R.from_quat(quat)

        # 从这个 Rotation 对象获取旋转矩阵
        rotation = rotation.as_matrix()
        self.kf_estimator = LinearKFPositionVelocityEstimator(
            dt=self.model.opt.timestep,
            rotation=rotation,  # Rotation matrix will be set later
            lin_acc=None,  # Linear acceleration will be set later
            ang_vel=None,  # Angular velocity will be set later
            qpos=None,  # Joint positions will be set later
            qvel=None,  # Joint velocities will be set later
        )
        qpos_, qvel_ = self.get_joint_state()
        self.kf_estimator.initialize(self.urdfmodel, qpos_)

        # Start the simulation viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # Initialize the action array
        self.action = np.zeros(cfg.num_actions, dtype=np.double)

        # Control parameters
        self.Kp = np.array([40, 40, 40, 40, 40, 40], dtype=np.double)
        self.Kd = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5], dtype=np.double)
        self.dof_pos_default = np.array([0, 0, 0, 0, 0, 0], dtype=np.double)
        self.tau_limit = 40.0 * np.ones(6, dtype=np.double)
        self.gait_index_ = 0.0

        # Simulation parameters
        self.decimation = 10
        self.action_scale = 0.25
        self.iter_ = 0

        # Set control commands
        self.commands = np.zeros((4,), dtype=np.double)
        self.commands[0] = 1.0  # x velocity
        self.commands[1] = 0.0  # y velocity
        self.commands[2] = 0.0  # yaw angular velocity
        self.commands[3] = 0.625  # base height

        # Initialize the plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.estimate_vels = []
        self.true_vels = []
        self.time_steps = []
        self.commands[3] = 0.625  # base height

    def get_joint_state(self):
        """Retrieve the joint position and velocity states."""
        q_ = self.data.qpos.astype(np.double)
        dq_ = self.data.qvel.astype(np.double)
        return q_[7:], dq_[6:]

    def get_base_state(self):
        """Get the base state including linear velocity, angular velocity, and projected gravity."""
        quat = self.data.sensor("imu_quat").data[[1, 2, 3, 0]].astype(np.double)
        r = R.from_quat(quat)
        base_lin_vel,true_lin_vel = r.apply(self.estimate_base_lin_vel(), inverse=True).astype(
            np.double
        )
        base_ang_vel = self.data.qvel[3:6].astype(np.double)
        projected_gravity = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(
            np.double
        )
        return base_lin_vel, base_ang_vel, projected_gravity

    def get_contact(self):
        """Detect and return contact information."""
        num_contacts = self.data.ncon
        left_contact = 0
        right_contact = 0
        for i in range(num_contacts):
            contact = self.data.contact[i]
            if contact.geom2 == 5:
                left_contact = 1
            if contact.geom2 == 9:
                right_contact = 1
        return np.array([left_contact, right_contact])

    def compute_obs(self):
        """Calculate and return the observed states from the policy input."""
        obs_scales = self.cfg.observation.normalization
        dof_pos, dof_vel = self.get_joint_state()
        base_lin_vel, base_ang_vel, projected_gravity = self.get_base_state()
        CommandScaler = np.array(
            [
                obs_scales.lin_vel,
                obs_scales.lin_vel,
                obs_scales.ang_vel,
                obs_scales.base_height,
            ]
        )

        proprioception_obs = np.zeros([1, self.cfg.num_actor_obs], dtype=np.float32)
        proprioception_obs[0, :3] = projected_gravity * obs_scales.gravity  # 3
        proprioception_obs[0, 3:6] = base_ang_vel * obs_scales.ang_vel  # 3
        proprioception_obs[0, 6:12] = dof_pos * obs_scales.dof_pos  # 6
        proprioception_obs[0, 12:18] = dof_vel * obs_scales.dof_vel  # 6
        proprioception_obs[0, 18:22] = CommandScaler * self.commands  # 4
        proprioception_obs[0, 22:28] = self.action  # 6

        gait = np.array([2.0, 0.5, 0.5, 0.1])  # trot
        self.gait_index_ = self.gait_index_ + 0.02 * gait[0]
        if self.gait_index_ > 1.0:
            self.gait_index_ = 0.0
        gait_clock = np.zeros(2, dtype=np.double)
        gait_clock[0] = np.sin(self.gait_index_ * 2 * np.pi)
        gait_clock[1] = np.cos(self.gait_index_ * 2 * np.pi)

        proprioception_obs[0, 28:30] = gait_clock  # 2
        proprioception_obs[0, 30:34] = gait  # 4

        proprioception_obs[0, 34:37] = base_lin_vel

        return proprioception_obs

    def pd_control(self, target_q):
        """Use PD to find target joint torques"""
        dof_pos, dof_vel = self.get_joint_state()
        return (target_q + self.dof_pos_default - dof_pos) * self.Kp - self.Kd * dof_vel

    def run(self):
        """Main loop of simulation"""
        target_q = np.zeros(self.cfg.num_actions, dtype=np.double)
        while self.data.time < 1000.0 and self.viewer.is_running():
            step_start = time.time()

            if self.iter_ % self.decimation == 0:
                proprioception_obs = self.compute_obs()
                action = (
                    self.policy(torch.tensor(proprioception_obs))[0].detach().numpy()
                )
                self.action[:] = np.clip(
                    action, -self.cfg.clip.clip_actions, self.cfg.clip.clip_actions
                )
                target_q[:] = self.action_scale * self.action

            # Generate PD control
            tau = self.pd_control(target_q)  # Calc torques
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)  # Clip torques
            self.data.ctrl = tau

            mujoco.mj_step(self.model, self.data)

            if self.iter_ % 10 == 0:  # 50Hz
                self.viewer.sync()

            # Update the plot every 100 iterations
            if self.iter_ % 100 == 0:
                estimate_vel,true_vel  = self.estimate_base_lin_vel()
                self.update_plot(self.data.time, true_vel[0]-estimate_vel[0])  # Assuming you want to plot x component

            self.iter_ += 1
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def update_plot(self, t, estimate_error):
        """Update the plot with new data."""
        self.time_steps.append(t)
        self.estimate_vels.append(estimate_error)

        self.ax.clear()
        self.ax.plot(self.time_steps, self.estimate_vels, label='Estimated error')
        self.ax.legend()
        plt.draw()
        plt.pause(0.001)

    def estimate_base_lin_vel(self):
        """Estimate the base linear velocity."""
        contact_info = self.get_contact()  # [left_c, right_c] 1 is contact, 0 is off-ground
        qpos_, qvel_ = self.get_joint_state()  # [left_abad, left_hip, left_knee, right_abad, right_hip, right_knee]
        lin_acc = np.array(self.data.sensor("imu_acc").data, dtype=np.double)
        ang_vel = np.array(self.data.sensor("imu_gyro").data, dtype=np.double)

        # Prepare data for Kalman Filter
        # 假设 quat 是从 Mujoco 数据中获取的四元数，格式 [x, y, z, w]
        quat = self.data.sensor("imu_quat").data[[1, 2, 3, 0]].astype(np.double)

        # 创建一个 Rotation 对象
        rotation = R.from_quat(quat)
        qpos__=self.init_qpos__

        # 从这个 Rotation 对象获取旋转矩阵
        rotation = rotation.as_matrix()
        mujoco_data = {
            'sensordata': np.concatenate([self.data.sensor("imu_acc").data, self.data.sensor("imu_gyro").data]),
            'qpos': np.concatenate([qpos__, self.data.sensor("imu_quat").data[[1, 2, 3, 0]].astype(np.double), qpos_]),
            'qvel': qvel_,
            'contact_info': contact_info
        }
        self.kf_estimator.rotation = rotation
        self.kf_estimator.lin_acc = lin_acc
        self.kf_estimator.ang_vel = ang_vel
        self.kf_estimator.qpos = qpos_
        self.kf_estimator.qvel = qvel_

        if USE_SIM:
            print("self.data.qvel[:3]",self.data.qvel[:3])
            return self.data.qvel[:3]
        else:
            state_estimate = self.kf_estimator.run(mujoco_data) # Run the Kalman Filter
            self.init_qpos__ = state_estimate["position"] # Update the initial position
            print("state_estimate['vWorld']",state_estimate["vWorld"]) # Print the estimated linear velocity
            return state_estimate["vWorld"],self.data.qvel[:3]


@hydra.main(
    version_base=None,
    config_name="xiaotian_config",
    config_path=os.path.join(SOURCE_DIR, "cfg"),
)
def main(cfg: DictConfig) -> None:
    policy_plan_path = os.path.join(SOURCE_DIR, "policy/policy.pt")
    policy_plan = torch.jit.load(policy_plan_path)
    sim = MuJoCoSim(cfg.env, policy_plan)
    sim.run()


if __name__ == "__main__":
    main()

import torch
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml


class DualArmFK:
    def __init__(self, config_path_left=None, config_path_right=None):
        self.tensor_args = TensorDeviceType()
        if config_path_left is None:
            config_path_left = "/data/cuihz/RoboTwin2.0_3D_policy/assets/embodiments/aloha-agilex/curobo_left.yml"
        if config_path_right is None:
            config_path_right = "/data/cuihz/RoboTwin2.0_3D_policy/assets/embodiments/aloha-agilex/curobo_right.yml"
        config_file_left = load_yaml(config_path_left)
        urdf_file_left = config_file_left["robot_cfg"]["kinematics"]["urdf_path"]
        base_link_left = config_file_left["robot_cfg"]["kinematics"]["base_link"]
        ee_link_left = config_file_left["robot_cfg"]["kinematics"]["ee_link"]
        self.left_bias = config_file_left['planner']['frame_bias']
        self.left_bias = torch.tensor(self.left_bias, device='cuda', dtype=torch.float32)
        robot_cfg_left = RobotConfig.from_basic(urdf_file_left, base_link_left, ee_link_left, self.tensor_args)
        self.kin_model_left = CudaRobotModel(robot_cfg_left.kinematics)
        config_file_right = load_yaml(config_path_right)
        urdf_file_right = config_file_right["robot_cfg"]["kinematics"]["urdf_path"]
        base_link_right = config_file_right["robot_cfg"]["kinematics"]["base_link"]
        ee_link_right = config_file_right["robot_cfg"]["kinematics"]["ee_link"]
        self.right_bias = config_file_right['planner']['frame_bias']
        self.right_bias = torch.tensor(self.right_bias, device='cuda', dtype=torch.float32)
        robot_cfg_right = RobotConfig.from_basic(urdf_file_right, base_link_right, ee_link_right, self.tensor_args)
        self.kin_model_right = CudaRobotModel(robot_cfg_right.kinematics)

    def get_fk_left(self, q):
        ee = self.kin_model_left.get_state(q)
        left_position = ee.ee_pose.position - self.left_bias
        left_quaternion = ee.ee_pose.quaternion
        left_pose_7d = torch.cat([left_position, left_quaternion], dim=1)
        left_pose_numpy = left_pose_7d.cpu().numpy()
        return left_pose_numpy

    def get_fk_right(self, q):
        ee = self.kin_model_right.get_state(q)
        right_position = ee.ee_pose.position - self.right_bias
        right_quaternion = ee.ee_pose.quaternion
        right_pose_7d = torch.cat([right_position, right_quaternion], dim=1)
        right_pose_numpy = right_pose_7d.cpu().numpy()
        return right_pose_numpy

    def get_dof_left(self):
        return self.kin_model_left.get_dof()

    def get_dof_right(self):
        return self.kin_model_right.get_dof()
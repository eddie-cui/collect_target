import sapien.core as sapien
import numpy as np
import pdb
import numpy as np
import toppra as ta
import math
import yaml
import os
import transforms3d as t3d
from copy import deepcopy
import sapien.core as sapien
import torch.multiprocessing as mp


class Robot:

    def __init__(self, **kwargs):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging
        self._init_robot_(**kwargs)

    def _init_robot_(self, **kwargs):
        # self.dual_arm = dual_arm_tag
        # self.plan_success = True

        self.left_js = None
        self.right_js = None

        left_embodiment_args = kwargs["left_embodiment_config"]
        right_embodiment_args = kwargs["right_embodiment_config"]
        left_robot_file = kwargs["left_robot_file"]
        right_robot_file = kwargs["right_robot_file"]

        self.left_urdf_path = os.path.join(left_robot_file, left_embodiment_args["urdf_path"])
        self.left_srdf_path = left_embodiment_args.get("srdf_path", None)
        self.left_curobo_yml_path = os.path.join(left_robot_file, "curobo.yml")
        if self.left_srdf_path is not None:
            self.left_srdf_path = os.path.join(left_robot_file, self.left_srdf_path)
        self.left_joint_stiffness = left_embodiment_args.get("joint_stiffness", 1000)
        self.left_joint_damping = left_embodiment_args.get("joint_damping", 200)
        self.left_gripper_stiffness = left_embodiment_args.get("gripper_stiffness", 1000)
        self.left_gripper_damping = left_embodiment_args.get("gripper_damping", 200)
        self.left_planner_type = left_embodiment_args.get("planner", "mplib_RRT")
        self.left_move_group = left_embodiment_args["move_group"][0]
        self.left_ee_name = left_embodiment_args["ee_joints"][0]
        self.left_arm_joints_name = left_embodiment_args["arm_joints_name"][0]
        self.left_gripper_name = left_embodiment_args["gripper_name"][0]
        self.left_gripper_bias = left_embodiment_args["gripper_bias"]
        self.left_gripper_scale = left_embodiment_args["gripper_scale"]
        self.left_homestate = left_embodiment_args.get("homestate", [[0] * len(self.left_arm_joints_name)])[0]
        self.left_fix_gripper_name = left_embodiment_args.get("fix_gripper_name", [])
        self.left_delta_matrix = np.array(left_embodiment_args.get("delta_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.left_inv_delta_matrix = np.linalg.inv(self.left_delta_matrix)
        self.left_global_trans_matrix = np.array(
            left_embodiment_args.get("global_trans_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

        _entity_origion_pose = left_embodiment_args.get("robot_pose", [[0, -0.65, 0, 1, 0, 0, 1]])[0]
        _entity_origion_pose = sapien.Pose(_entity_origion_pose[:3], _entity_origion_pose[-4:])
        self.left_entity_origion_pose = deepcopy(_entity_origion_pose)

        self.right_urdf_path = os.path.join(right_robot_file, right_embodiment_args["urdf_path"])
        self.right_srdf_path = right_embodiment_args.get("srdf_path", None)
        if self.right_srdf_path is not None:
            self.right_srdf_path = os.path.join(right_robot_file, self.right_srdf_path)
        self.right_curobo_yml_path = os.path.join(right_robot_file, "curobo.yml")
        self.right_joint_stiffness = right_embodiment_args.get("joint_stiffness", 1000)
        self.right_joint_damping = right_embodiment_args.get("joint_damping", 200)
        self.right_gripper_stiffness = right_embodiment_args.get("gripper_stiffness", 1000)
        self.right_gripper_damping = right_embodiment_args.get("gripper_damping", 200)
        self.right_planner_type = right_embodiment_args.get("planner", "mplib_RRT")
        self.right_move_group = right_embodiment_args["move_group"][1]
        self.right_ee_name = right_embodiment_args["ee_joints"][1]
        self.right_arm_joints_name = right_embodiment_args["arm_joints_name"][1]
        self.right_gripper_name = right_embodiment_args["gripper_name"][1]
        self.right_gripper_bias = right_embodiment_args["gripper_bias"]
        self.right_gripper_scale = right_embodiment_args["gripper_scale"]
        self.right_homestate = right_embodiment_args.get("homestate", [[1] * len(self.right_arm_joints_name)])[1]
        self.right_fix_gripper_name = right_embodiment_args.get("fix_gripper_name", [])
        self.right_delta_matrix = np.array(right_embodiment_args.get("delta_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.right_inv_delta_matrix = np.linalg.inv(self.right_delta_matrix)
        self.right_global_trans_matrix = np.array(
            right_embodiment_args.get("global_trans_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

        _entity_origion_pose = right_embodiment_args.get("robot_pose", [[0, -0.65, 0, 1, 0, 0, 1]])
        _entity_origion_pose = _entity_origion_pose[0 if len(_entity_origion_pose) == 1 else 1]
        _entity_origion_pose = sapien.Pose(_entity_origion_pose[:3], _entity_origion_pose[-4:])
        self.right_entity_origion_pose = deepcopy(_entity_origion_pose)
        self.is_dual_arm = kwargs["dual_arm_embodied"]

        self.left_rotate_lim = left_embodiment_args.get("rotate_lim", [0, 0])
        self.right_rotate_lim = right_embodiment_args.get("rotate_lim", [0, 0])

        self.left_perfect_direction = left_embodiment_args.get("grasp_perfect_direction",
                                                               ["front_right", "front_left"])[0]
        self.right_perfect_direction = right_embodiment_args.get("grasp_perfect_direction",
                                                                 ["front_right", "front_left"])[1]

    def get_left_world_endpose(self):
        return self.left_ee.global_pose.p.tolist() + self.left_ee.global_pose.q.tolist()
    def get_right_world_endpose(self):
        return self.right_ee.global_pose.p.tolist() + self.right_ee.global_pose.q.tolist()
    def left_base_to_world_pose(self, base_coord):
        base_p = base_coord[:3]
        base_q = base_coord[-4:]
        R_ent = t3d.quaternions.quat2mat(self.left_entity_origion_pose.q)
        R_b   = t3d.quaternions.quat2mat(base_q)
        R_g   = self.left_global_trans_matrix

        p_world = R_ent @ np.asarray(base_p) + np.asarray(self.left_entity_origion_pose.p)
        R_world = R_ent @ R_b @ R_g.T

        q_world = t3d.quaternions.mat2quat(R_world)
        if q_world[3] < 0:
            q_world = -q_world
        return p_world.tolist() + q_world.tolist()

    def right_base_to_world_pose(self, base_coord):
        base_p = base_coord[:3]
        base_q = base_coord[-4:]
        R_ent = t3d.quaternions.quat2mat(self.right_entity_origion_pose.q)
        R_b   = t3d.quaternions.quat2mat(base_q)
        R_g   = self.right_global_trans_matrix

        p_world = R_ent @ np.asarray(base_p) + np.asarray(self.right_entity_origion_pose.p)
        R_world = R_ent @ R_b @ R_g.T

        q_world = t3d.quaternions.mat2quat(R_world)
        if q_world[3] < 0:
            q_world = -q_world
        return p_world.tolist() + q_world.tolist()
    def get_fk_left_endpose(self, endpose_kf):
        return self._trans_kf_endpose(arm_tag="left", is_endpose=False, endpose=endpose_kf)
    def get_fk_right_endpose(self, endpose_kf):
        return self._trans_kf_endpose(arm_tag="right", is_endpose=False, endpose=endpose_kf)
    def _trans_kf_endpose(self, arm_tag=None, is_endpose=False, endpose=None):
            if arm_tag is None:
                print("No arm tag")
                return
            p= endpose[:3]
            q= endpose[-4:]
            gripper_bias = (self.left_gripper_bias if arm_tag == "left" else self.right_gripper_bias)
            global_trans_matrix = (self.left_global_trans_matrix if arm_tag == "left" else self.right_global_trans_matrix)
            delta_matrix = (self.left_delta_matrix if arm_tag == "left" else self.right_delta_matrix)
            endpose_arr = np.eye(4)
            endpose_arr[:3, :3] = (t3d.quaternions.quat2mat(q) @ global_trans_matrix @ delta_matrix)
            dis = gripper_bias
            if is_endpose == False:
                dis -= 0.12
            endpose_arr[:3, 3] = np.array(p) + endpose_arr[:3, :3] @ np.array([dis, 0, 0]).T
            res = (endpose_arr[:3, 3].tolist() + t3d.quaternions.mat2quat(endpose_arr[:3, :3]).tolist())
            return res
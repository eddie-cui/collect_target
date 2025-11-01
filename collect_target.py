from fk import DualArmFK
import numpy as np
import transforms3d as t3d
from utils import Robot
import pdb
import yaml
import importlib
import json
import h5py
from tqdm import tqdm
import traceback
import os
import torch
import time
from argparse import ArgumentParser
def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join('/data/sea_disk0/cuihz/RoboTwin/assets/embodiments/aloha-agilex/', "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

def collect_target(folder_path, task_config='demo_clean'):
    config_path = f"/data/sea_disk0/cuihz/RoboTwin/task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)


    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join('/data/sea_disk0/cuihz/RoboTwin/task_config/', "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])
    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    robot = Robot(**args)
    fk = DualArmFK()
    file_name_list = os.listdir(folder_path)
    for file in tqdm(file_name_list, desc="Processing files"):
        print(f"Processing file: {file}")
        if not file.endswith('.hdf5'):
            continue
        file_path = os.path.join(folder_path, file)
        with h5py.File(file_path, 'r') as hf:
            target_joint_left = hf['joint_action/left_arm'][:]
            target_joint_right = hf['joint_action/right_arm'][:]
        base_fk_left = fk.get_fk_left(torch.tensor(target_joint_left,device='cuda',dtype=torch.float32))
        base_fk_right = fk.get_fk_right(torch.tensor(target_joint_right,device='cuda',dtype=torch.float32))
        target_left_endpose = []
        target_right_endpose = []
        for i in range(base_fk_left.shape[0]):
            world_fk_left = robot.left_base_to_world_pose(base_fk_left[i])
            world_fk_right = robot.right_base_to_world_pose(base_fk_right[i])
            target_left_endpose.append(robot.get_fk_left_endpose(world_fk_left))
            target_right_endpose.append(robot.get_fk_right_endpose(world_fk_right))
        with h5py.File(file_path, 'r+') as hf:
            grp = hf.require_group('target_endpose')
            if 'left_arm' in grp:
                del grp['left_arm']
            if 'right_arm' in grp:
                del grp['right_arm']

            grp.create_dataset('left_arm',  data=target_left_endpose)
            grp.create_dataset('right_arm', data=target_right_endpose)
        
if __name__ == "__main__":
    collect_target_folder = '/data/sea_disk0/cuihz/collect_target/move_pillbottle_pad/demo_clean/data/'
    collect_target(collect_target_folder, "demo_clean")
    # collect_target_folder = '/data/sea_disk0/wushr/3D-Policy/RoboTwin_modified/data/move_pillbottle_pad/demo_clean/data'
    # collect_target(collect_target_folder, "demo_clean")
    # collect_target_folder = '/data/sea_disk0/wushr/3D-Policy/RoboTwin_modified/data/place_bread_basket/demo_clean/data'
    # collect_target(collect_target_folder, "demo_clean")
    # collect_target_folder = '/data/sea_disk0/wushr/3D-Policy/RoboTwin_modified/data/place_bread_skillet/demo_clean/data'
    # collect_target(collect_target_folder, "demo_clean")
    # collect_target_folder = '/data/sea_disk0/wushr/3D-Policy/RoboTwin_modified/data/place_fan/demo_clean/data'
    # collect_target(collect_target_folder, "demo_clean")
    # collect_target_folder = '/data/sea_disk0/wushr/3D-Policy/RoboTwin_modified/data/place_shoe/demo_clean/data'
    # collect_target(collect_target_folder, "demo_clean")
    # collect_target_folder = '/data/sea_disk0/wushr/3D-Policy/RoboTwin_modified/data/rotate_qrcode/demo_clean/data'
    # collect_target(collect_target_folder, "demo_clean")
    # collect_target_folder = '/data/sea_disk0/wushr/3D-Policy/RoboTwin_modified/data/stamp_seal/demo_clean/data'
    # collect_target(collect_target_folder, "demo_clean")
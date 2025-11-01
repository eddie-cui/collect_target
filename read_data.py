import h5py
import numpy as np

def print_hdf5_structure(name, obj):
    """递归打印HDF5文件结构"""
    indent = "  " * name.count('/')
    if isinstance(obj, h5py.Group):
        print(f"{indent}{name}/ (Group)")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")

def main():
    # data_path = '/Users/eddie/code/research/trajectory.pointcloud.pd_joint_pos.physx_cuda.h5'
    data_path='/data/sea_disk0/wushr/3D-Policy/RoboTwin_modified/data/place_shoe/demo_clean/data/episode0.hdf5'
    print("=== HDF5 File Structure ===")
    with h5py.File(data_path, 'r') as f:
        f.visititems(print_hdf5_structure)
        # for i in range(107):
        #     print('QPOS',f['traj_0/obs/agent/qpos'][i+1])
        #     print("Action",f['traj_0/actions'][i],"\n")
    print("\n=== Detailed endpose structure ===")
    with h5py.File(data_path, 'r') as f:
        if 'endpose' in f:
            endpose_group = f['endpose']
            print("Keys in endpose:", list(endpose_group.keys()))
            # qpos = f['observation/qpos']
            
            for key in endpose_group.keys():
                data = endpose_group[key]
                print(f"{key}:")
                print(f"  Shape: {data.shape}")
                print(f"  Dtype: {data.dtype}")
                print(f"  First few values: {data[:3] if len(data) > 0 else 'No data'}")
                print()
        endpose_left = f['endpose/left_endpose'][:]
        endpose_right = f['endpose/right_endpose'][:]
        target_endpose_left = f['target_endpose/left_arm'][:]
        target_endpose_right = f['target_endpose/right_arm'][:]

        np.set_printoptions(precision=4, suppress=True, linewidth=200)
        for i in range(len(endpose_left)):
            print(f"Endpose Left {i}: {endpose_left[i]}")
            print(f"Target Endpose Left {i}: {target_endpose_left[i]}")
            # print(f"GT Endpose Left {i}: {gt_endpose_left[i]}")
            # print(f"Check Endpose Left {i}: {check_endpose_left[i]}")
            print(f"Delta Left {i}: {endpose_left[i]-target_endpose_left[i]}")
            print()
            print(f"Endpose Right {i}: {endpose_right[i]}")
            print(f"Target Endpose Right {i}: {target_endpose_right[i]}")
            print(f"Delta Right {i}: {endpose_right[i]-target_endpose_right[i]}")
            # print(f"GT Endpose Right {i}: {gt_endpose_right[i]}")
            # print(f"Check Endpose Right {i}: {check_endpose_right[i]}")
            print()

if __name__ == "__main__":
    main()
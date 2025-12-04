import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import SkeletonRobotViewer as RobotMotionViewer
import general_motion_retargeting.torch_utils as torch_utils
from rich import print
from tqdm import tqdm
import os
import numpy as np
import torch
import pickle
import mujoco
from scipy.spatial.transform import Rotation as R

def exp_map_to_quat_numpy(exp_map):
    r = R.from_rotvec(exp_map)
    quat_xyzw = r.as_quat()
    if len(exp_map.shape) == 1:
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    else:
        quat_wxyz = np.zeros_like(quat_xyzw)
        quat_wxyz[:, 0] = quat_xyzw[:, 3]
        quat_wxyz[:, 1] = quat_xyzw[:, 0]
        quat_wxyz[:, 2] = quat_xyzw[:, 1]
        quat_wxyz[:, 3] = quat_xyzw[:, 2]
    return quat_wxyz

def load_humanoid_data(pkl_file):
    """
    Load Humanoid data from pkl file.
    """
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    frames_data = data['frames'] # Shape: (N, 34)
    num_frames = frames_data.shape[0]
    
    # Load humanoid model using Mujoco for consistent FK
    assets_dir = pathlib.Path(__file__).parent.parent / "assets"
    humanoid_model_path = str(assets_dir / "humanoid" / "humanoid.xml")
    
    try:
        humanoid_model = mujoco.MjModel.from_xml_path(humanoid_model_path)
        humanoid_data = mujoco.MjData(humanoid_model)
    except Exception as e:
        print(f"Error loading humanoid model: {e}")
        raise e
        
    frame_rate = data.get('fps', 30)
    
    # Parse data
    root_pos = frames_data[:, 0:3]
    root_rot_exp = frames_data[:, 3:6]
    joint_dof = frames_data[:, 6:]
    
    # Convert root rot to wxyz
    root_rot_quat = exp_map_to_quat_numpy(root_rot_exp)
    
    # Structural info
    body_names = [mujoco.mj_id2name(humanoid_model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(humanoid_model.nbody)]
    parents = np.array(humanoid_model.body_parentid)
    
    frames = []
    
    # Compute FK using Mujoco
    # This ensures consistency with visualization scripts and avoids coordinate system issues
    # with the custom KinematicsModel
    for i in range(num_frames):
        # Set root
        humanoid_data.qpos[0:3] = root_pos[i]
        humanoid_data.qpos[3:7] = root_rot_quat[i]
        
        # Set joints (map 28 dofs to qpos starting at 7)
        # Verify dimensions
        num_dof_data = joint_dof.shape[1]
        # humanoid_model.nq is usually 7 + num_joints (if all hinges)
        # check actuators or joints? 
        # 28 dofs matches the number of actuators/joints in humanoid.xml
        
        # Safety check for size
        target_dof_size = min(num_dof_data, humanoid_model.nq - 7)
        humanoid_data.qpos[7:7+target_dof_size] = joint_dof[i, :target_dof_size]
        
        mujoco.mj_forward(humanoid_model, humanoid_data)
        
        frame_dict = {}
        for body_id in range(humanoid_model.nbody):
            body_name = body_names[body_id]
            if body_name is None or body_name == "world":
                continue
            
            # Mujoco returns pos and quat (wxyz) in global frame
            pos = humanoid_data.xpos[body_id].copy()
            quat = humanoid_data.xquat[body_id].copy()
            frame_dict[body_name] = (pos, quat)
            
        frames.append(frame_dict)

    # Estimate human height (Head z - Min Foot z)
    if "head" in frames[0]:
        head_pos = frames[0]["head"][0]
        # Simple estimation
        human_height = head_pos[2] + 0.1
    else:
        human_height = 1.6
        
    if human_height < 1.0: 
        human_height = 1.6
        
    return frames, human_height, frame_rate, parents, body_names

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        help="Humanoid data pkl file to load.",
        # required=True,
        type=str,
        default="/home/phi/CLX/projects/misc/GMR/third_party/dataset/parc/platform/beyond_platform_001_flipped.pkl",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "limx_oli"],
        default="unitree_g1",
    )
        
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/humanoid_retarget.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    args = parser.parse_args()
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    # Load Humanoid Data
    print(f"Loading data from {args.data_file}...")
    humanoid_frames, actual_human_height, motion_fps, human_parents, human_body_names = load_humanoid_data(args.data_file)
    print(f"Loaded {len(humanoid_frames)} frames. FPS: {motion_fps}. Est Height: {actual_human_height:.2f}m")
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human="humanoid", # Corresponds to humanoid_to_g1.json
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=motion_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=args.video_path,
    )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    pbar = tqdm(total=len(humanoid_frames), desc="Retargeting")
    
    i = 0
    while i < len(humanoid_frames):
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        pbar.update(1)

        # Update task targets
        human_data = humanoid_frames[i]

        # Retarget
        qpos = retargeter.retarget(human_data)

        # Visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            human_parents=human_parents,
            human_body_names=human_body_names,
            rate_limit=args.rate_limit,
        )

        i += 1
        time.sleep(0.05)

        if args.save_path is not None:
            qpos_list.append(qpos)
    
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw (mujoco convention to some other convention if needed, but usually kept consistent)
        # GMR returns wxyz, if downstream needs xyzw:
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list]) 
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    pbar.close()
    robot_motion_viewer.close()


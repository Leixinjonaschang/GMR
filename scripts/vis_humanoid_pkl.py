import argparse
import pickle
import time
import numpy as np
import mujoco
import mujoco.viewer
import pathlib
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting.kinematics_model import KinematicsModel
import general_motion_retargeting.torch_utils as torch_utils
import torch

# Create a simple MuJoCo scene (skybox + floor + lights)
EMPTY_SCENE_XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <statistic center="0 0 1" extent="2"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3.0" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""

def draw_skeleton_frame(viewer, frame_data, parents, body_names, color_joint=[0.2, 0.6, 1.0, 1.0], color_bone=[0.7, 0.7, 0.7, 1.0], show_axes=True):
    """
    Draw one frame of skeleton data in viewer (joints + bone connectors)
    """
    SPHERE_RADIUS = 0.03
    BONE_WIDTH = 0.015
    AXIS_RADIUS = 0.005
    AXIS_LENGTH = 0.1
    
    # 1. Draw bones (connect parent and child joints)
    if parents is not None:
        for i, body_name in enumerate(body_names):
            parent_idx = parents[i]
            # Skip root node (no parent or parent is -1)
            if parent_idx == -1:
                continue
            
            parent_name = body_names[parent_idx]
            
            # Make sure both joints exist in current frame data
            if body_name in frame_data and parent_name in frame_data:
                p_start = frame_data[parent_name][0]
                p_end = frame_data[body_name][0]
                
                viewer.user_scn.ngeom += 1
                geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
                
                # Initialize geometry
                mujoco.mjv_initGeom(
                    geom,
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=color_bone
                )
                
                # Use connector to connect two points automatically
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    BONE_WIDTH,
                    p_start,
                    p_end
                )

    # 2. Draw joints
    for body_name, (pos, quat) in frame_data.items():
        # Draw joint sphere
        viewer.user_scn.ngeom += 1
        geom_id = viewer.user_scn.ngeom - 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_id],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([SPHERE_RADIUS, 0, 0]),
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=color_joint
        )

        # 3. Draw axes
        if show_axes:
            # Convert quaternion (wxyz) to rotation matrix
            rot_matrix = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
            
            axis_colors = [
                [1, 0, 0, 1],  # X: red
                [0, 1, 0, 1],  # Y: green
                [0, 0, 1, 1],  # Z: blue
            ]
            
            for axis_idx in range(3):
                viewer.user_scn.ngeom += 1
                geom_id = viewer.user_scn.ngeom - 1
                
                # Compute cylinder rotation
                if axis_idx == 0:   # X axis
                    cyl_rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                elif axis_idx == 1: # Y axis
                    cyl_rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                else:               # Z axis
                    cyl_rot = np.eye(3)
                
                final_rot = rot_matrix @ cyl_rot
                
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[geom_id],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    size=np.array([AXIS_RADIUS, AXIS_RADIUS, AXIS_LENGTH]),
                    pos=pos,
                    mat=final_rot.flatten(),
                    rgba=axis_colors[axis_idx]
                )

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

def main():
    parser = argparse.ArgumentParser(description="Visualize Humanoid PKL motion file")
    parser.add_argument("--pkl_file", type=str, required=True, help="Path to the humanoid data pkl file")
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS (overrides file FPS if set)")
    parser.add_argument("--loop", action="store_true", default=True, help="Loop playback")
    parser.add_argument("--no_axes", action="store_true", help="Do not show axes")
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading data from {args.pkl_file}...")
    try:
        with open(args.pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if 'frames' not in data:
        print("Error: 'frames' key not found in pickle file.")
        return

    frames = data['frames'] # (N, 34)
    file_fps = data.get('fps', 30)
    play_fps = args.fps if args.fps is not None else file_fps
    
    num_frames = frames.shape[0]
    print(f"Loaded {num_frames} frames.")
    print(f"File FPS: {file_fps}, Playback FPS: {play_fps}")

    # 2. Load Humanoid Model for Kinematics Calculation
    current_dir = pathlib.Path(__file__).parent.resolve()
    project_root = current_dir.parent
    model_path = project_root / "assets" / "humanoid.xml"
    
    print(f"Loading kinematics model from {model_path}...")
    try:
        # Use KinematicsModel to do FK, avoiding direct mujoco rendering of this xml
        kinematics_model = KinematicsModel(str(model_path), device="cpu")
    except Exception as e:
        print(f"Error loading KinematicsModel: {e}")
        return

    # 3. Pre-process Data (FK) to get per-frame body positions/rotations
    print("Computing Forward Kinematics...")
    
    frames_tensor = torch.tensor(frames, dtype=torch.float32)
    root_pos = frames_tensor[:, 0:3]
    root_rot_exp = frames_tensor[:, 3:6]
    joint_dof = frames_tensor[:, 6:]
    
    root_rot_quat = torch_utils.exp_map_to_quat(root_rot_exp) # (N, 4) wxyz
    
    # Check DOF dimension
    if joint_dof.shape[1] != kinematics_model.num_dof:
        print(f"Warning: Data DOF dim ({joint_dof.shape[1]}) does not match Model DOF dim ({kinematics_model.num_dof}).")
        
    # Compute FK
    body_pos_batch, body_rot_batch = kinematics_model.forward_kinematics(root_pos, root_rot_quat, joint_dof)
    
    # Extract structural info
    body_names = kinematics_model.body_names
    parents = kinematics_model.parent_indices.numpy()
    
    # Convert to list of dicts for visualization
    processed_frames = []
    for i in range(num_frames):
        frame_dict = {}
        for j, body_name in enumerate(body_names):
            pos = body_pos_batch[i, j].numpy()
            rot = body_rot_batch[i, j].numpy() # (w, x, y, z)
            frame_dict[body_name] = (pos, rot)
        processed_frames.append(frame_dict)

    # 4. Load Visualization Environment (Empty Scene)
    print("Loading visualization environment...")
    model = mujoco.MjModel.from_xml_string(EMPTY_SCENE_XML)
    data = mujoco.MjData(model)

    # 5. Visualization Loop
    print("\nStarting visualization...")
    print("Controls:")
    print("  - Space: Pause/Resume (Not implemented yet, running loop)")
    print("  - Left mouse drag: rotate view")
    print("  - Right mouse drag: pan")
    print("  - Mouse wheel: zoom")
    print("  - ESC: Exit")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set initial camera
        viewer.cam.lookat[:] = [0, 0, 1.0]
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 45
        
        frame_idx = 0
        last_render_time = time.time()
        frame_duration = 1.0 / play_fps
        is_paused = False
        
        while viewer.is_running():
            current_time = time.time()
            
            if not is_paused and (current_time - last_render_time >= frame_duration):
                # Clear previous frame
                viewer.user_scn.ngeom = 0
                
                # Draw Skeleton
                current_frame_data = processed_frames[frame_idx]
                draw_skeleton_frame(
                    viewer, 
                    current_frame_data, 
                    parents, 
                    body_names,
                    show_axes=not args.no_axes
                )
                
                # Update viewer
                viewer.sync()
                
                # Advance frame
                frame_idx += 1
                if frame_idx >= num_frames:
                    if args.loop:
                        frame_idx = 0
                    else:
                        frame_idx = num_frames - 1
                        is_paused = True
                
                last_render_time = current_time
            
            # Sleep to save CPU
            time.sleep(0.001)

if __name__ == "__main__":
    main()

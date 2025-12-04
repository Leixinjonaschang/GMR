"""
Quick viewer for humanoid motion stored in PKL files.

Features:
- Loads a MuJoCo humanoid model from assets/humanoid/humanoid.xml.
- Reads PKL motion data with format [root_pos(3), root_rot_exp(3), joint_dof(28)].
- Converts exp-map root rotation to MuJoCo quaternion (wxyz).
- Drives the humanoid model with this motion and visualizes it in a MuJoCo viewer.
- Can render either:
  - only the mesh model,
  - only a skeleton (joints + bones + optional axes), or
  - both mesh and skeleton overlaid (controlled by --mode).
"""

import argparse
import pickle
import time
import numpy as np
import mujoco
import mujoco.viewer
import pathlib
from scipy.spatial.transform import Rotation as R

EMPTY_SCENE_XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <statistic center="0 0 1" extent="2"/>
  <visual>
    <!-- soft neutral lighting -->
    <headlight diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
    <!-- light grey haze -->
    <rgba haze="0.9 0.9 0.9 1"/>
    <!-- neutral camera setup -->
    <global azimuth="120" elevation="-20"/>
  </visual>
  <asset>
    <!-- very light grey skybox -->
    <texture type="skybox" builtin="gradient" rgb1="0.95 0.95 0.95" rgb2="0.85 0.85 0.85" width="512" height="3072"/>
    <!-- light grey checker ground -->
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.9 0.9 0.9" rgb2="0.8 0.8 0.8" markrgb="0.7 0.7 0.7"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.05"/>
  </asset>
  <worldbody>
    <!-- soft overhead light -->
    <light pos="0 0 4.0" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""

def exp_map_to_quat_numpy(exp_map):
    """
    Convert exponential map to quaternion (w, x, y, z) using scipy.
    """
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

def draw_skeleton_frame(
    viewer,
    frame_data,
    parents,
    body_names,
    # make joints and bones blue-ish by default
    color_joint=[0.2, 0.6, 1.0, 1.0],
    color_bone=[0.1, 0.3, 1.0, 0.8],
    show_axes=True,
):
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

def main():
    parser = argparse.ArgumentParser(description="Visualize Humanoid PKL motion file (model / skeleton / both)")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the humanoid data pkl file")
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS (overrides file FPS if set)")
    parser.add_argument("--loop", action="store_true", default=True, help="Loop playback")
    parser.add_argument("--mode", type=str, choices=["model", "skeleton", "both"], default="both", help="Visualization mode")
    parser.add_argument("--no_axes", action="store_true", help="Do not show axes for skeleton")
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading data from {args.data_file}...")
    try:
        with open(args.data_file, 'rb') as f:
            pkl_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if 'frames' not in pkl_data:
        print("Error: 'frames' key not found in pickle file.")
        return

    frames = pkl_data['frames'] # (N, 34)
    file_fps = pkl_data.get('fps', 30)
    play_fps = args.fps if args.fps is not None else file_fps
    
    num_frames = frames.shape[0]
    print(f"Loaded {num_frames} frames.")
    print(f"File FPS: {file_fps}, Playback FPS: {play_fps}")

    # 2. Load Humanoid Model (similar camera/lighting style as RobotMotionViewer)
    current_dir = pathlib.Path(__file__).parent.resolve()
    project_root = current_dir.parent
    model_path = project_root / "assets" / "humanoid" / "humanoid.xml"
    
    print(f"Loading Mujoco model from {model_path}...")
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading Mujoco model: {e}")
        return

    # 3. Pre-process Data to match qpos format
    # PKL Format: [root_pos(3), root_rot_exp(3), joints(28)]
    # Mujoco qpos (with freejoint): [root_pos(3), root_quat(4, wxyz), joints(N)]
    
    print("Processing motion data...")
    
    # Separate components
    root_pos = frames[:, 0:3]
    root_rot_exp = frames[:, 3:6]
    joint_data = frames[:, 6:]
    
    # Convert exp map to quaternion (wxyz)
    root_rot_quat = exp_map_to_quat_numpy(root_rot_exp)
    
    # Prepare skeleton structure info
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    parents = np.array(model.body_parentid)

    # Control model appearance:
    # - skeleton mode: hide model completely
    # - model / both: make model semi-transparent for better comparison
    if args.mode == "skeleton":
        for i in range(model.ngeom):
            model.geom_rgba[i][3] = 0.0
    else:
        for i in range(model.ngeom):
            # reduce alpha but keep other channels
            model.geom_rgba[i][3] *= 0.3
    
    # 4. Visualization Loop
    print("\nStarting visualization...")
    print("Controls:")
    print("  - Space: Pause/Resume (Not implemented yet)")
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
                
                # Update humanoid state (model)
                data.qpos[0:3] = root_pos[frame_idx]
                data.qpos[3:7] = root_rot_quat[frame_idx]
                num_joints_to_set = min(joint_data.shape[1], model.nq - 7)
                data.qpos[7:7+num_joints_to_set] = joint_data[frame_idx, :num_joints_to_set]
                
                mujoco.mj_forward(model, data)

                # Clear previous skeleton geoms
                viewer.user_scn.ngeom = 0

                if args.mode in ["skeleton", "both"]:
                    # Build frame_data from current humanoid state
                    frame_data = {}
                    for body_id in range(model.nbody):
                        body_name = body_names[body_id]
                        if body_name is None or body_name == "world":
                            continue
                        pos = data.xpos[body_id].copy()
                        quat = data.xquat[body_id].copy()  # wxyz
                        frame_data[body_name] = (pos, quat)

                    draw_skeleton_frame(
                        viewer,
                        frame_data,
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

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting.utils.lafan1 import load_lafan1_file
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

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

def draw_skeleton_frame(viewer, frame_data, parents, bones, color_joint=[0.2, 0.6, 1.0, 1.0], color_bone=[0.7, 0.7, 0.7, 1.0], show_axes=True):
    """
    Draw one frame of skeleton data in viewer (joints + bone connectors)
    """
    SPHERE_RADIUS = 0.03
    BONE_WIDTH = 0.015
    AXIS_RADIUS = 0.005
    AXIS_LENGTH = 0.1
    
    # 1. Draw bones (connect parent and child joints)
    if parents is not None and bones is not None:
        for i, bone_name in enumerate(bones):
            parent_idx = parents[i]
            # Skip root node (no parent)
            if parent_idx == -1:
                continue
            
            parent_name = bones[parent_idx]
            
            # Make sure both joints exist in current frame data
            if bone_name in frame_data and parent_name in frame_data:
                p_start = frame_data[parent_name][0]
                p_end = frame_data[bone_name][0]
                
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
        # Optionally: show joint name label
        # viewer.user_scn.geoms[geom_id].label = body_name

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
    parser = argparse.ArgumentParser(description="BVH Visualization Tool (with skeleton)")
    parser.add_argument("--bvh_file", type=str, required=True, help="Path to BVH file")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback framerate (default: 30)")
    parser.add_argument("--loop", action="store_true", default=True, help="Loop playback")
    parser.add_argument("--no_axes", action="store_true", help="Do not show axes")
    args = parser.parse_args()

    print(f"Loading BVH file: {args.bvh_file}")
    try:
        # 1. Load motion data (per-frame joint positions and orientations)
        frames, human_height = load_lafan1_file(args.bvh_file)
        
        # 2. Load skeleton topology (parent-child relationships)
        # load_lafan1_file internally uses read_bvh, but doesn't return the structure; load explicitly here
        bvh_struct = read_bvh(args.bvh_file)
        parents = bvh_struct.parents
        bones = bvh_struct.bones
        
        print(f"Loaded successfully!")
        print(f"- Total frames: {len(frames)}")
        print(f"- Number of joints: {len(bones)}")
        print(f"- Estimated human height: {human_height:.2f}m")
    except Exception as e:
        print(f"Error: Failed to load BVH file - {e}")
        return

    # Load empty MuJoCo model as the environment
    model = mujoco.MjModel.from_xml_string(EMPTY_SCENE_XML)
    data = mujoco.MjData(model)

    print("\nLaunching visualization window...")
    print("Controls:")
    print("  - Left mouse drag: rotate view")
    print("  - Right mouse drag: pan")
    print("  - Mouse wheel: zoom")
    print("  - ESC: exit")

    # Launch Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set initial camera view
        viewer.cam.lookat[:] = [0, 0, 1.0]
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 45

        frame_idx = 0
        last_render_time = time.time()
        frame_duration = 1.0 / args.fps
        
        is_paused = False

        while viewer.is_running():
            current_time = time.time()
            
            # Simple framerate control
            if not is_paused and (current_time - last_render_time >= frame_duration):
                # Clear previous frame geometries
                viewer.user_scn.ngeom = 0
                
                # Get current frame data
                current_frame_data = frames[frame_idx]
                
                # Draw skeleton
                draw_skeleton_frame(
                    viewer, 
                    current_frame_data, 
                    parents,
                    bones,
                    show_axes=not args.no_axes
                )
                
                # Advance frame index
                frame_idx += 1
                if frame_idx >= len(frames):
                    if args.loop:
                        frame_idx = 0
                    else:
                        frame_idx = len(frames) - 1
                        is_paused = True # Pause at end of playback
                
                last_render_time = current_time
                
                # Update viewer
                viewer.sync()

            # Reduce CPU usage
            time.sleep(0.001)

if __name__ == "__main__":
    main()

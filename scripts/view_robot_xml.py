import argparse
import sys
import types
import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# Mock 'util' module just in case imports need it
if 'util' not in sys.modules:
    sys.modules['util'] = types.ModuleType('util')

from general_motion_retargeting.params import ROBOT_XML_DICT, VIEWER_CAM_DISTANCE_DICT

def draw_coordinate_frame(viewer, pos, quat_wxyz, size=0.5, axis_radius=0.005, label=None):
    """
    Draw a coordinate frame (Red=X, Green=Y, Blue=Z)
    """
    # Convert wxyz to xyzw for scipy
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    rot_matrix = R.from_quat(quat_xyzw).as_matrix()
    
    axis_colors = [
        [1.0, 0.0, 0.0, 1.0],  # X: Red
        [0.0, 1.0, 0.0, 1.0],  # Y: Green
        [0.0, 0.0, 1.0, 1.0],  # Z: Blue
    ]
    
    for axis_idx in range(3):
        viewer.user_scn.ngeom += 1
        geom_id = viewer.user_scn.ngeom - 1
        
        # Rotate cylinder to point to X, Y, Z
        if axis_idx == 0:  # X
            cyl_rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]) # +Z -> +X
        elif axis_idx == 1:  # Y
            cyl_rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]) # +Z -> +Y
        else:  # Z
            cyl_rot = np.eye(3)
        
        final_rot = rot_matrix @ cyl_rot
        
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_id],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.array([axis_radius, axis_radius, size]),
            pos=pos,
            mat=final_rot.flatten(),
            rgba=axis_colors[axis_idx]
        )
    
    # Optional Label
    if label:
        viewer.user_scn.ngeom += 1
        label_id = viewer.user_scn.ngeom - 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[label_id],
            type=mujoco.mjtGeom.mjGEOM_LABEL,
            size=np.array([1, 1, 1]), # font scale?
            pos=pos + np.array([0, 0, size + 0.02]),
            mat=np.eye(3).flatten(),
            rgba=[1, 1, 1, 1]
        )
        viewer.user_scn.geoms[label_id].label = label

def main():
    parser = argparse.ArgumentParser(description="View Robot XML with Coordinate Frames")
    parser.add_argument("--robot", type=str, default="unitree_g1", choices=ROBOT_XML_DICT.keys(), help="Robot name to visualize")
    parser.add_argument("--show_labels", action="store_true", help="Show body name labels")
    
    args = parser.parse_args()

    robot_xml = str(ROBOT_XML_DICT[args.robot])
    print(f"Loading Robot: {args.robot} from {robot_xml}")
    
    try:
        model = mujoco.MjModel.from_xml_path(robot_xml)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading XML: {e}")
        return

    # Ensure robot is in zero pose (qpos=0)
    mujoco.mj_resetData(model, data)
    
    # Lift robot pelvis if needed (find freejoint and set Z)
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            qadr = model.jnt_qposadr[i]
            # If Z is 0, lift it up a bit so feet are on ground roughly
            # Or just leave it at 0 if model default is good.
            # Usually ~1.0m is safe for humanoid
            if data.qpos[qadr+2] < 0.1:
                data.qpos[qadr+2] = 1.0 
            break
            
    mujoco.mj_forward(model, data)

    print("\nStarting Viewer...")
    print("Red=X, Green=Y, Blue=Z")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera
        viewer.cam.lookat[:] = [0, 0, 0.8]
        viewer.cam.distance = VIEWER_CAM_DISTANCE_DICT.get(args.robot, 3.0)
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -15
        
        # Make robot transparent if requested
        # if args.transparent:
        for i in range(model.ngeom):
            model.geom_rgba[i][3] = 0.3

        while viewer.is_running():
            
            viewer.user_scn.ngeom = 0
            
            # Draw Coordinate Frames for all bodies
            for i in range(model.nbody):
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name == "world": continue
                
                pos = data.xpos[i]
                quat_wxyz = data.xquat[i] # MuJoCo format: wxyz
                
                draw_coordinate_frame(
                    viewer, 
                    pos, 
                    quat_wxyz, 
                    size=0.1, 
                    label=body_name if args.show_labels else None
                )

            viewer.sync()
            time.sleep(0.02)

if __name__ == "__main__":
    main()


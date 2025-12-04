import mujoco as mj
import numpy as np
from .robot_motion_viewer import RobotMotionViewer, draw_frame
from scipy.spatial.transform import Rotation as R


def draw_skeleton(
    human_motion_data,
    parents,
    body_names,
    viewer,
    pos_offset=np.array([0, 0, 0]),
    color_bone=[0.7, 0.7, 0.7, 1.0]
):
    """
    Draw simple lines/capsules between joints
    """
    BONE_WIDTH = 0.015
    
    if parents is not None:
        for i, body_name in enumerate(body_names):
            parent_idx = parents[i]
            if parent_idx == -1:
                continue
            
            parent_name = body_names[parent_idx]
            
            if body_name in human_motion_data and parent_name in human_motion_data:
                p_start = human_motion_data[parent_name][0] + pos_offset
                p_end = human_motion_data[body_name][0] + pos_offset
                
                viewer.user_scn.ngeom += 1
                geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
                
                mj.mjv_initGeom(
                    geom,
                    type=mj.mjtGeom.mjGEOM_CAPSULE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=color_bone
                )
                
                mj.mjv_connector(
                    geom,
                    mj.mjtGeom.mjGEOM_CAPSULE,
                    BONE_WIDTH,
                    p_start,
                    p_end
                )


class SkeletonRobotViewer(RobotMotionViewer):
    def step(self, 
            # robot data
            root_pos, root_rot, dof_pos, 
            # human data
            human_motion_data=None, 
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization    
            human_pos_offset=np.array([0.0, 0.0, 0]),
            # skeleton info
            human_parents=None,
            human_body_names=None,
            # rate limit
            rate_limit=True, 
            follow_camera=True,
            ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.
        
        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """
        
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot # quat need to be scalar first! for mujoco
        self.data.qpos[7:] = dof_pos
        
        mj.mj_forward(self.model, self.data)
        
        if follow_camera:
            self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id]
            self.viewer.cam.distance = self.viewer_cam_distance
            self.viewer.cam.elevation = -10  # 正面视角，轻微向下看
            # self.viewer.cam.azimuth = 180    # 正面朝向机器人
        
        if human_motion_data is not None:
            # Clean custom geometry
            self.viewer.user_scn.ngeom = 0
            
            # Draw Skeleton if structure provided
            if human_parents is not None and human_body_names is not None:
                draw_skeleton(
                    human_motion_data, 
                    human_parents, 
                    human_body_names, 
                    self.viewer, 
                    pos_offset=human_pos_offset
                )

            # Draw the task targets (coordinate frames)
            for human_body_name, (pos, rot) in human_motion_data.items():
                draw_frame(
                    pos,
                    R.from_quat(rot, scalar_first=True).as_matrix(),
                    self.viewer,
                    human_point_scale,
                    pos_offset=human_pos_offset,
                    joint_name=human_body_name if show_human_body_name else None
                    )

        self.viewer.sync()
        if rate_limit is True:
            self.rate_limiter.sleep()

        if self.record_video:
            # Use renderer for proper offscreen rendering
            self.renderer.update_scene(self.data, camera=self.viewer.cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)
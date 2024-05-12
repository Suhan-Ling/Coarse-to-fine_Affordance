"""
    Environment with one object at center
        external: one robot, one camera
"""

from __future__ import division
import sapien.core as sapien
from sapien.core import Pose, SceneConfig

import numpy as np


class ContactError(Exception):
    pass


class SVDError(Exception):
    pass


def process_angle_limit(x):
    if np.isneginf(x):
        x = -10
    if np.isinf(x):
        x = 10
    return x


class Env(object):

    def __init__(self, flog=None, show_gui=True, render_rate=20, timestep=1 / 500,
                 object_position_offset=0.0, succ_ratio=0.1, use_kuafu=True):
        self.current_step = 0

        self.flog = flog
        self.show_gui = show_gui
        self.render_rate = render_rate
        self.timestep = timestep
        self.succ_ratio = succ_ratio
        self.object_position_offset = object_position_offset
        self.use_kuafu = use_kuafu

        # engine and renderer
        self.engine = sapien.Engine(0, 0.001, 0.005)
        self.renderer = sapien.VulkanRenderer()
        if use_kuafu:
            renderer_config = sapien.KuafuConfig()
            renderer_config.use_viewer = False
            renderer_config.spp = 64
            renderer_config.max_bounces = 8
            renderer_config.use_denoiser = True
            self.renderer = sapien.KuafuRenderer(renderer_config)
        self.engine.set_renderer(self.renderer)

        scene_config = SceneConfig()
        scene_config.gravity = [0, 0, -9.81]
        scene_config.solver_iterations = 20
        scene_config.enable_pcm = False
        scene_config.sleep_threshold = 0.0
        self.scene = self.engine.create_scene(config=scene_config)
        self.scene.set_timestep(timestep)

        # add lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1 + object_position_offset, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1 + object_position_offset, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1 + object_position_offset, 0, 1], [1, 1, 1], shadow=True)

        # default Nones
        self.object = None
        self.object_target_joint = None

        # check contact
        self.check_contact = False
        self.contact_error = False
        self.non_target_object_part_actor_id = list()

    def rebuild(self):

        timestep = self.timestep
        object_position_offset = self.object_position_offset

        scene_config = SceneConfig()
        scene_config.gravity = [0, 0, -9.81]
        scene_config.solver_iterations = 20
        scene_config.enable_pcm = False
        scene_config.sleep_threshold = 0.0

        self.scene = self.engine.create_scene(config=scene_config)

        self.scene.set_timestep(timestep)

        # add lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1 + object_position_offset, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1 + object_position_offset, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1 + object_position_offset, 0, 1], [1, 1, 1], shadow=True)

    def set_controller_camera_pose(self, x, y, z, yaw, pitch):
        self.renderer_controller.set_camera_position(x, y, z)
        self.renderer_controller.set_camera_rotation(yaw, pitch)
        self.renderer_controller.render()

    def load_object(self, urdf, material, state='closed', target_part_id=-1):
        loader = self.scene.create_urdf_loader()

        self.object = loader.load(urdf, {"material": material})
        # self.object = loader.load(urdf, material)
        pose = Pose([self.object_position_offset, 0, 0], [1, 0, 0, 0])
        self.object.set_root_pose(pose)

        # compute link actor information
        self.all_link_ids = [l.get_id() for l in self.object.get_links()]
        self.movable_link_ids = []
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                self.movable_link_ids.append(j.get_child_link().get_id())
        if self.flog is not None:
            self.flog.write('All Actor Link IDs: %s\n' % str(self.all_link_ids))
            self.flog.write('All Movable Actor Link IDs: %s\n' % str(self.movable_link_ids))

        # set joint property
        for joint in self.object.get_joints():
            joint.set_drive_property(stiffness=0, damping=10)

        # set initial qpos
        joint_angles = []
        self.joint_angles_lower = []
        self.joint_angles_upper = []
        target_part_joint_idx = -1
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    target_part_joint_idx = len(joint_angles)
                l = process_angle_limit(j.get_limits()[0, 0])
                self.joint_angles_lower.append(float(l))
                r = process_angle_limit(j.get_limits()[0, 1])
                self.joint_angles_upper.append(float(r))
                if state == 'closed':
                    joint_angles.append(float(l))
                elif state == 'open':
                    joint_angles.append(float(r))
                elif state == 'middle':
                    joint_angles.append(float(l + (r - l) / 4))
                else:
                    raise ValueError('ERROR: object init state %s unknown!' % state)

        self.object.set_qpos(joint_angles)
        if target_part_id >= 0:
            return joint_angles, target_part_joint_idx

        return joint_angles

    def load_real_object(self, urdf, material, joint_angles=None):
        loader = self.scene.create_urdf_loader()
        self.object = loader.load(urdf, {"material": material})
        pose = Pose([self.object_position_offset, 0, 0], [1, 0, 0, 0])
        self.object.set_root_pose(pose)

        # compute link actor information
        self.all_link_ids = [l.get_id() for l in self.object.get_links()]
        self.movable_link_ids = []
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                self.movable_link_ids.append(j.get_child_link().get_id())
        if self.flog is not None:
            self.flog.write('All Actor Link IDs: %s\n' % str(self.all_link_ids))
            self.flog.write('All Movable Actor Link IDs: %s\n' % str(self.movable_link_ids))

        # set joint property
        for joint in self.object.get_joints():
            joint.set_drive_property(stiffness=0, damping=10)

        if joint_angles is not None:
            self.object.set_qpos(joint_angles)

        return None

    def update_and_set_joint_angles_all(self, state='closed'):
        joint_angles = []
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                l = process_angle_limit(j.get_limits()[0, 0])
                self.joint_angles_lower.append(float(l))
                r = process_angle_limit(j.get_limits()[0, 1])
                self.joint_angles_upper.append(float(r))
                if state == 'closed':
                    joint_angles.append(float(l))
                elif state == 'open':
                    joint_angles.append(float(r))
                else:
                    raise ValueError('ERROR: object init state %s unknown!' % state)
        self.object.set_qpos(joint_angles)
        return joint_angles

    def get_target_part_axes(self, target_part_id):
        joint_axes = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    pos = j.get_global_pose()
                    mat = pos.to_transformation_matrix()
                    joint_axes = [float(-mat[1, 0]), float(mat[2, 0]), float(-mat[0, 0])]
        if joint_axes is None:
            raise ValueError('joint axes error!')
        return joint_axes

    def get_target_part_axes_new(self, target_part_id):
        joint_axes = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    pos = j.get_global_pose()
                    mat = pos.to_transformation_matrix()
                    joint_axes = [float(-mat[0, 0]), float(-mat[1, 0]), float(mat[2, 0])]
        if joint_axes is None:
            raise ValueError('joint axes error!')

        return joint_axes

    def get_target_part_axes_dir(self, target_part_id):
        joint_axes = self.get_target_part_axes(target_part_id=target_part_id)
        axes_dir = -1
        for idx_axes_dim in range(3):
            if abs(joint_axes[idx_axes_dim]) > 0.5:
                axes_dir = idx_axes_dim
        return axes_dir

    def get_target_part_axes_dir_new(self, target_part_id):
        joint_axes = self.get_target_part_axes_new(target_part_id=target_part_id)
        axes_dir = -1
        for idx_axes_dim in range(3):
            if abs(joint_axes[idx_axes_dim]) > 0.1:
                axes_dir = idx_axes_dim
        return axes_dir

    def get_target_part_origins_new(self, target_part_id):
        joint_origins = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    pos = j.get_global_pose()
                    joint_origins = pos.p.tolist()
        if joint_origins is None:
            raise ValueError('joint origins error!')

        return joint_origins

    def get_target_part_origins(self, target_part_id):
        print("attention!!! origin")
        joint_origins = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    pos = j.get_global_pose()
                    mat = pos.to_transformation_matrix()
                    joint_origins = [float(-mat[1, 3]), float(mat[2, 3]), float(-mat[0, 3])]
        if joint_origins is None:
            raise ValueError('joint origins error!')

        return joint_origins

    def update_joint_angle(self, joint_angles, target_part_joint_idx, state, task_lower, push=True, pull=False,
                           drawer=False):
        if push:
            if drawer:
                l = max(self.joint_angles_lower[target_part_joint_idx],
                        self.joint_angles_lower[target_part_joint_idx] + task_lower)
                r = self.joint_angles_upper[target_part_joint_idx]
            else:
                l = max(self.joint_angles_lower[target_part_joint_idx],
                        self.joint_angles_lower[target_part_joint_idx] + task_lower * np.pi / 180)
                r = self.joint_angles_upper[target_part_joint_idx]
        if pull:
            if drawer:
                l = self.joint_angles_lower[target_part_joint_idx]
                r = self.joint_angles_upper[target_part_joint_idx] - task_lower
            else:
                l = self.joint_angles_lower[target_part_joint_idx]
                r = self.joint_angles_upper[target_part_joint_idx] - task_lower * np.pi / 180
        if state == 'closed':
            joint_angles[target_part_joint_idx] = (float(l))
        elif state == 'open':
            joint_angles[target_part_joint_idx] = float(r)
        else:
            raise ValueError('ERROR: object init state %s unknown!' % state)
        return joint_angles

    def set_object_joint_angles(self, joint_angles):
        self.object.set_qpos(joint_angles)

    def set_target_object_part_actor_id(self, actor_id):
        if self.flog is not None:
            self.flog.write('Set Target Object Part Actor ID: %d\n' % actor_id)
        self.target_object_part_actor_id = actor_id
        self.non_target_object_part_actor_id = list(set(self.all_link_ids) - {actor_id})

        # get the link handler
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == actor_id:
                    self.target_object_part_actor_link = j.get_child_link()

        # moniter the target joint
        idx = 0
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == actor_id:
                    self.target_object_part_joint_id = idx
                    self.target_object_part_joint_type = j.type
                idx += 1

    def get_object_qpos(self):
        return self.object.get_qpos()

    def get_target_part_qpos(self):
        qpos = self.object.get_qpos()
        return float(qpos[self.target_object_part_joint_id])

    def get_target_part_pose(self):
        return self.target_object_part_actor_link.get_pose()

    def start_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, strict):
        self.check_contact = True
        self.check_contact_strict = strict
        self.first_timestep_check_contact = True
        self.robot_hand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids
        self.contact_error = False

    def end_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, strict):
        self.check_contact = False
        self.check_contact_strict = strict
        self.first_timestep_check_contact = False
        self.robot_hand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids

    def get_material(self, static_friction, dynamic_friction, restitution):
        return self.engine.create_physical_material(static_friction, dynamic_friction, restitution)

    def render(self):
        if self.show_gui and (not self.window):
            self.window = True
            self.renderer_controller.show_window()
        self.scene.update_render()
        if self.show_gui and (self.current_step % self.render_rate == 0):
            self.renderer_controller.render()

    def step(self):
        self.current_step += 1
        self.scene.step()
        if self.check_contact:
            if not self.check_contact_is_valid():
                raise ContactError()

    def close_render(self):
        if self.window:
            self.renderer_controller.hide_window()
        self.window = False

    def wait_to_start(self):
        print('press q to start\n')
        while not self.renderer_controller.should_quit:
            self.scene.update_render()
            if self.show_gui:
                self.renderer_controller.render()

    def close(self):
        if self.show_gui:
            self.renderer_controller.set_current_scene(None)
        self.scene = None

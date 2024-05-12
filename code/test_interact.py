"""
    For panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
        50% all parts closed, 50% middle (for each part, 50% prob. closed, 50% prob. middle)
        Simulate until static before starting
"""

import os
import sys
import utils
import shutil
import numpy as np
from utils import get_global_position_from_camera, save_h5
import json
from argparse import ArgumentParser
from sapien.core import Pose
from env_old import Env, ContactError
from camera import Camera
from robots.panda_robot import Robot
import random
from PIL import Image
import open3d as o3d
import imageio
from data_interact import SAPIENVisionDataset

parser = ArgumentParser()
parser.add_argument('--cnt_id', type=int, default=0)
parser.add_argument('--shape_id', type=str, default='40147')
parser.add_argument('--category', type=str, default='StorageFurniture')
parser.add_argument('--state', type=str, default='closed')
parser.add_argument('--primact_type', type=str, default='pulling')
parser.add_argument('--data_dir', type=str, default='../data')
parser.add_argument('--trial_id', type=int, default=0, help='trial id')
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--rand', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--num_point_per_shape', type=int, default=10000)
parser.add_argument('--sapien_dir', type=str, default='../data')
parser.add_argument('--now_dist', type=float, default=2.5)
parser.add_argument('--nxt_dist', type=float, default=0.6)
parser.add_argument('--show_img', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--height', type=int, default=1080)
parser.add_argument('--width', type=int, default=1920)
args = parser.parse_args()

device = args.device

data_features = ['shape_id', 'zoom_in_view', 'zoom_in_point', 'manipulate_point', 'traj_save', 'save_number']
train_dataset = SAPIENVisionDataset([args.primact_type], data_features)
train_dataset.load_data(args.data_dir)

def run_collect(cnt_id=args.cnt_id, trial_id=args.trial_id, shape_id=args.shape_id,
                primact_type=args.primact_type, out_dir=args.data_dir):
    flog = open(os.path.join(out_dir, 'log.txt'), 'w')
    env = Env(flog=flog, show_gui=(not args.no_gui))

    test_log = open(os.path.join(out_dir, 'test_log.txt'), 'w')
    utils.printout(test_log, ' '.join(sys.argv) + '\n')

    cam = Camera(env, theta=np.pi, phi=0.7826405702413783, dist=args.now_dist)
    cam1 = Camera(env, theta=np.pi, phi=0, dist=args.now_dist)

    mu1 = 4
    mu2 = 4
    density = 1
    target_part_id = 0
    robot_urdf_fn = './robots/panda_gripper.urdf'
    robot_material = env.get_material(4, 4, 0.01)
    object_material = env.get_material(mu1, mu2, 0.01)

    succ_cnt = 0
    tot_cnt = 0
    failure_rate = 0
    grasp_succ = 0
    succ_rate = 0
    grasp_succ_rate = 0
    ct_error = 0

    for ww in range(len(train_dataset.data_buffer)):
        out_info = {}
        up, forward, manipulate_point, shape_id, _, _ = train_dataset.data_buffer[ww]
        if up is None:
            continue
        up = np.array(up)
        forward = np.array(forward)
        object_urdf_fn = '%s/where2act_original_sapien_dataset/%s/mobility_vhacd.urdf' % (args.sapien_dir, shape_id)
        flog.write('object_urdf_fn: %s\n' % object_urdf_fn)
        state = args.state
        env.load_object(object_urdf_fn, object_material, state=state, density=density)
        out_info['joint_angles_lower'] = env.joint_angles_lower
        out_info['joint_angles_upper'] = env.joint_angles_upper

        cur_qpos = env.get_object_qpos()

        out_info['cur_qpos'] = cur_qpos.tolist()

        env.step()
        env.render()

        left = np.cross(up, forward)
        left /= np.linalg.norm(left)
        forward = np.cross(left, up)
        forward /= np.linalg.norm(forward)

        out_info['gripper_up'] = up.tolist()
        out_info['gripper_left'] = left.tolist()
        out_info['gripper_forward'] = forward.tolist()
        out_info['shape_id'] = shape_id
        out_info['manipulation_point'] = manipulate_point

        robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))

        rotmat = np.eye(4).astype(np.float32)
        rotmat[:3, 0] = forward
        rotmat[:3, 1] = left
        rotmat[:3, 2] = up

        position_world = np.array(manipulate_point[2])

        final_dist = 0.05
        out_info['dist'] = final_dist

        final_rotmat = np.array(rotmat, dtype=np.float32)
        final_rotmat[:3, 3] = position_world - up * 0.06 - cam1.mat44[:3, 0] * 0.2
        final_pose = Pose().from_transformation_matrix(final_rotmat)
        out_info['target_rotmat_world'] = final_rotmat.tolist()

        start_rotmat = np.array(rotmat, dtype=np.float32)
        start_rotmat[:3, 3] = position_world - up * 0.2
        start_pose = Pose().from_transformation_matrix(start_rotmat)
        out_info['start_rotmat_world'] = start_rotmat.tolist()

        end_rotmat = np.array(rotmat, dtype=np.float32)
        end_rotmat[:3, 3] = position_world - up * 0.06
        out_info['end_rotmat_world'] = end_rotmat.tolist()

        robot.robot.set_root_pose(start_pose)

        ### main steps
        out_info['start_part_qpos'] = env.get_object_qpos().tolist()

        grasp_fail = False
        success = True
        succ_images = []
        try:
            init_success = True
            success_grasp = False
            try:
                robot.open_gripper()
                if args.show_img:
                    imgs = robot.move_to_target_pose(end_rotmat, 1000, vis_gif=True, vis_gif_interval=200, cam=cam)
                    succ_images.extend(imgs)
                    imgs = robot.wait_n_steps(1000, vis_gif=True, vis_gif_interval=200, cam=cam)
                    succ_images.extend(imgs)
                    robot.close_gripper()
                    imgs = robot.wait_n_steps(1000, vis_gif=True, vis_gif_interval=200, cam=cam)
                    succ_images.extend(imgs)
                else:
                    robot.move_to_target_pose(end_rotmat, 1000)
                    robot.wait_n_steps(1000)
                    robot.close_gripper()
                    robot.wait_n_steps(1000)
                now_qpos = robot.robot.get_qpos().tolist()
                finger1_qpos = now_qpos[-1]
                finger2_qpos = now_qpos[-2]
                if finger1_qpos + finger2_qpos > 0.01:
                    success_grasp = True
            except ContactError:
                init_success = False
            if not (success_grasp and init_success):
                success = False
                grasp_fail = True
            else:
                try:
                    if args.show_img:
                        imgs = robot.move_to_target_pose(final_rotmat, 1000, vis_gif=True, vis_gif_interval=200, cam=cam)
                        succ_images.extend(imgs)
                        imgs = robot.wait_n_steps(1000, vis_gif=True, vis_gif_interval=200, cam=cam)
                        succ_images.extend(imgs)
                    else:
                        robot.move_to_target_pose(final_rotmat, 1000)
                        robot.wait_n_steps(1000)
                except Exception:
                    ct_error = ct_error + 1
                    success = False
        except ContactError:
            success = False

        env.scene.remove_articulation(robot.robot)

        succ_dir = os.path.join(out_dir, 'succ_gif')
        if not os.path.exists(succ_dir):
            os.mkdir(succ_dir)
        grasp_dir = os.path.join(out_dir, 'grasp_gif')
        if not os.path.exists(grasp_dir):
            os.mkdir(grasp_dir)
        fail_dir = os.path.join(out_dir, 'fail_gif')
        if not os.path.exists(fail_dir):
            os.mkdir(fail_dir)

        if success:
            grasp_succ += 1
            out_info['final_part_qpos'] = env.get_object_qpos().tolist()
            out_info['gt_labels'] = 0
            gt_motion = 0
            for ct in range(len(out_info['final_part_qpos'])):
                gt_motion = max(gt_motion,
                                abs(out_info['final_part_qpos'][ct] -
                                    out_info['start_part_qpos'][ct]))
            if gt_motion > 0.01:
                succ_cnt += 1
                succ_rate = succ_cnt / (ww + 1)
                grasp_rate = grasp_succ / (ww + 1)
                utils.printout(test_log,
                               f'success: {shape_id}   {ww}   {succ_rate}   {grasp_rate}')
                if args.show_img and len(succ_images) > 0:
                    imageio.mimsave(os.path.join(succ_dir, 'pic_%s.gif' % (ww)), succ_images)
            else:
                succ_rate = succ_cnt / (ww + 1)
                grasp_rate = grasp_succ / (ww + 1)
                utils.printout(test_log, f'grasp:   {shape_id}   {ww}   {succ_rate}   {grasp_rate}')
                if args.show_img and len(succ_images) > 0:
                    imageio.mimsave(os.path.join(grasp_dir, 'pic_%s.gif' % (ww)), succ_images)
        else:
            succ_rate = succ_cnt / (ww + 1)
            grasp_rate = grasp_succ / (ww + 1)
            utils.printout(test_log,
                           f'fail:    {shape_id}   {ww}   {succ_rate}   {grasp_rate}')
            if args.show_img and len(succ_images) > 0:
                imageio.mimsave(os.path.join(fail_dir, 'pic_%s.gif' % (ww)), succ_images)

        env.scene.remove_articulation(env.object)
        succ_rate = succ_cnt / (ww + 1)
        grasp_succ_rate = grasp_succ / (ww + 1)

    utils.printout(test_log,
                   f'succ_rate: {succ_rate} grasp_succ_rate: {grasp_succ_rate}')
    flog.close()
    env.close()

for idx_process in range(1):
    run_collect(idx_process, args.trial_id, args.shape_id, args.primact_type, args.data_dir)

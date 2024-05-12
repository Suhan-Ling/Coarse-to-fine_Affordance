"""
    For panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
        50% all parts closed, 50% middle (for each part, 50% prob. closed, 50% prob. middle)
        Simulate until static before starting
"""

import os
import sys
import numpy as np
import json
import utils
import imageio

from argparse import ArgumentParser
from sapien.core import Pose
from env_old import Env, ContactError
from camera import Camera
from robots.panda_robot import Robot
import torch
from sensor import Sensor
import random
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from data import SAPIENVisionDataset
from models import model_all
from plyfile import PlyData, PlyElement
import time

parser = ArgumentParser()
parser.add_argument('--cnt_id', type=int, default=0)
parser.add_argument('--shape_id', type=str, default='40147')
parser.add_argument('--category', type=str, default='StorageFurniture')
parser.add_argument('--primact_type', type=str, default='pulling')
parser.add_argument('--out_dir', type=str, default='../data')
parser.add_argument('--trial_id', type=int, default=0, help='trial id')
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--now_dist', type=float, default=2.5)
parser.add_argument('--nxt_dist', type=float, default=0.6)
parser.add_argument('--show_img', action='store_true', default=False)
parser.add_argument('--far_samples', type=int, default=10000)
parser.add_argument('--near_samples', type=int, default=10000)
parser.add_argument('--sample_type', type=str, default='fps')
parser.add_argument('--sapien_dir', type=str, default='../data')
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--load_dir', type=str, default=None)
parser.add_argument('--load_epoch', type=int, default=20)
parser.add_argument('--actor_dir', type=str, default=None)
parser.add_argument('--actor_epoch', type=int, default=20)
parser.add_argument('--data_per_shape', type=int, default=10)
parser.add_argument('--feat_dim', type=int, default=128)
parser.add_argument('--rv_dim', type=int, default=10)
parser.add_argument('--rv_cnt', type=int, default=100)
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--date', type=str, default='0905')
parser.add_argument('--collect_num', type=int, default=1)
parser.add_argument('--num_processes', type=int, default=20)
parser.add_argument('--single_succ', type=int, default=2)
args = parser.parse_args()

device = args.device

critic = model_all.Critic(args.feat_dim).to(args.device)
actor = model_all.Actor(feat_dim=args.feat_dim, rv_dim=args.rv_dim, rv_cnt=args.rv_cnt).to(args.device)
data_to_restore = torch.load(os.path.join("../logs", args.load_dir, 'ckpts', f"{args.load_epoch}-network.pth"))
critic.load_state_dict(data_to_restore)
data_to_restore = torch.load(os.path.join("../logs", args.actor_dir, 'ckpts', f"{args.actor_epoch}-actor.pth"))
actor.load_state_dict(data_to_restore)
critic.eval()
actor.eval()

data_features = ['gt_labels', 'up', 'forward', 'pc_near']
train_dataset = SAPIENVisionDataset([args.primact_type], data_features)
train_dataset.load_data(args.data_dir, real=True)


def run_collect(cnt_id=args.cnt_id, trial_id=args.trial_id, shape_id=args.shape_id, primact_type=args.primact_type,
                out_dir=args.out_dir):

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, 'process_%d' % (cnt_id))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    succ_dir = os.path.join(out_dir, 'succ')
    grasp_dir = os.path.join(out_dir, 'grasp')
    fail_dir = os.path.join(out_dir, 'fail')
    if args.show_img:
        if not os.path.exists(succ_dir):
            os.mkdir(succ_dir)
        if not os.path.exists(grasp_dir):
            os.mkdir(grasp_dir)
        if not os.path.exists(fail_dir):
            os.mkdir(fail_dir)

    np_random_seed = random.randint(1, 1000) + cnt_id
    np.random.seed(np_random_seed)
    random_seed = random.randint(1, 1000) + cnt_id
    random.seed(random_seed)

    flog = open(os.path.join(out_dir, 'log.txt'), 'w')
    env = Env(flog=flog, show_gui=(not args.no_gui))
    cam = Camera(env, theta=np.pi, phi=0.7826405702413783, dist=args.now_dist)
    cam1 = Camera(env, theta=np.pi, phi=0, dist=args.now_dist)

    my_log = open(os.path.join(out_dir, 'my_log.txt'), 'w')
    my_log.write(' '.join(sys.argv))
    my_log.write('np random seed: %d\n' % np_random_seed)
    my_log.write('   random seed: %d\n' % random_seed)

    mu1 = 4
    mu2 = 4
    density = 1
    target_part_id = 0
    succ_cnt = 0
    grasp_cnt = 0
    fail_cnt = 0
    tot_cnt = 0
    failure_rate = 0
    robot_urdf_fn = './robots/panda_gripper.urdf'
    robot_material = env.get_material(4, 4, 0.01)
    object_material = env.get_material(mu1, mu2, 0.01)
    grasp_succ = 0
    tot_trial = 0
    data_buffer_len = len(train_dataset.data_buffer)
    now_l = data_buffer_len // 3
    for ww in range(now_l * cnt_id, now_l * (cnt_id + 1)):
        out_info = {}
        gt_labels, up, left, forward, zoom_in_point, manipulate_point, shape_id, zoom_in_view, traj_far, traj_near, traj_save, j, _ = train_dataset.data_buffer[ww]

        utils.printout(my_log, f'{ww} / {now_l * 3} : {shape_id}')
        if utils.is_ignored(shape_id):
            continue

        plydata_far = PlyData.read(traj_far)
        pc_far_numpy = plydata_far['vertex'].data
        pc_far_numpy = np.array([[x, y, z] for x, y, z in pc_far_numpy], dtype=np.float32)

        while 0 < len(pc_far_numpy) < args.far_samples:
            pc_far_numpy = np.concatenate((pc_far_numpy, pc_far_numpy), axis=0)
        sample = np.random.permutation(len(pc_far_numpy))[:args.far_samples]
        pc_far_numpy = pc_far_numpy[sample, :3]
        pc_far = torch.FloatTensor(pc_far_numpy).to(args.device).reshape(1, args.far_samples, 3)

        plydata_near = PlyData.read(traj_near)
        pc_near_numpy = plydata_near['vertex'].data
        pc_near_numpy = np.array([[x, y, z] for x, y, z in pc_near_numpy], dtype=np.float32)

        while 0 < len(pc_near_numpy) < args.near_samples:
            pc_near_numpy = np.concatenate((pc_near_numpy, pc_near_numpy), axis=0)
        sample = np.random.permutation(len(pc_near_numpy))[:args.near_samples]
        pc_near_numpy = pc_near_numpy[sample, :3]
        pc_near = torch.FloatTensor(pc_near_numpy).to(args.device).reshape(1, args.near_samples, 3)

        object_urdf_fn = '%s/where2act_original_sapien_dataset/%s/mobility_vhacd.urdf' % (args.sapien_dir, shape_id)
        flog.write('object_urdf_fn: %s\n' % object_urdf_fn)
        state = 'closed'
        env.load_object(object_urdf_fn, object_material, state=state, density=density)

        cur_qpos = env.get_object_qpos()
        n = 20
        with torch.no_grad():
            near_feats = critic.get_near_feats(pc_near, pc_far)[0]
            actions = actor.inference_nactor_whole_pc(near_feats, n=n)
            pred_Rs = actor.bgs(actions.reshape(-1, 3, 2))
            up = pred_Rs[:, :, 0]
            forward = pred_Rs[:, :, 1]
            critic_result = critic.inference_critic_score_diff_naction(up, forward, near_feats, n)
            critic_result = critic_result.view(10000, n, 1).topk(k=3, dim=1)[0].mean(dim=1).view(-1)
            result = critic_result.detach().cpu().numpy()

        result = result.reshape(10000)
        accu = 0.95
        xs = np.where(result > accu)[0]
        while len(xs) < 1000:
            accu -= 0.03
            xs = np.where(result > accu)[0]

        trial_num = 1000
        if gt_labels == 0:
            trial_num = 20
        single_succ = 0
        start_time = time.time()

        for jj in range(trial_num):
            tot_trial += 1
            env.object.set_qpos(cur_qpos)

            env.step()
            env.render()

            p_id = xs[random.randint(0, len(xs) - 1)]
            if random.random() < 0.2:
                p_id = random.randint(0, 9999)
            position_world = pc_near_numpy[p_id]
            near_feat = near_feats[p_id]

            with torch.no_grad():
                pred_6d = actor.inference_actor(near_feat)[0]  # RV_CNT x 6
                pred_Rs = actor.bgs(pred_6d.reshape(-1, 3, 2))
                up = pred_Rs[:, :, 0]
                forward = pred_Rs[:, :, 1]
                id = random.randint(1, 99)
                up = up[id].detach().cpu().numpy()
                forward = forward[id].detach().cpu().numpy()

            left = np.cross(up, forward)
            left /= np.linalg.norm(left)
            forward = np.cross(left, up)
            forward /= np.linalg.norm(forward)

            out_info['gripper_up'] = up.tolist()
            out_info['gripper_left'] = left.tolist()
            out_info['gripper_forward'] = forward.tolist()
            out_info['shape_id'] = shape_id
            out_info['manipulation_point'] = [None, None, position_world.tolist()]
            out_info['zoom_in_point'] = zoom_in_point
            out_info['traj_far'] = traj_far
            out_info['traj_near'] = traj_near
            out_info['p_id'] = int(p_id)

            robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))

            rotmat = np.eye(4).astype(np.float32)
            rotmat[:3, 0] = forward
            rotmat[:3, 1] = left
            rotmat[:3, 2] = up

            final_dist = 0.15

            final_rotmat = np.array(rotmat, dtype=np.float32)
            final_rotmat[:3, 3] = position_world - up * 0.1 - cam1.mat44[:3, 0] * 0.15
            final_pose = Pose().from_transformation_matrix(final_rotmat)

            start_rotmat = np.array(rotmat, dtype=np.float32)
            start_rotmat[:3, 3] = position_world - up * 0.15
            start_pose = Pose().from_transformation_matrix(start_rotmat)

            end_rotmat = np.array(rotmat, dtype=np.float32)
            end_rotmat[:3, 3] = position_world - up * 0.1

            robot.robot.set_root_pose(start_pose)

            env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)

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
                        img = robot.move_to_target_pose(end_rotmat, 1000, vis_gif=True, vis_gif_interval=200, cam=cam)
                        succ_images.extend(img)
                        img = robot.wait_n_steps(1000, vis_gif=True, vis_gif_interval=200, cam=cam)
                        succ_images.extend(img)
                        robot.close_gripper()
                        img = robot.wait_n_steps(1000, vis_gif=True, vis_gif_interval=200, cam=cam)
                        succ_images.extend(img)
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
                            imgs = robot.move_to_target_pose(
                                final_rotmat, 1000, vis_gif=True,
                                vis_gif_interval=200, cam=cam)
                            succ_images.extend(imgs)
                            imgs = robot.wait_n_steps(
                                1000, vis_gif=True,
                                vis_gif_interval=200, cam=cam)
                            succ_images.extend(imgs)
                        else:
                            robot.move_to_target_pose(final_rotmat, 1000)
                            robot.wait_n_steps(1000)
                    except Exception:
                        success = False
            except ContactError:
                success = False

            env.scene.remove_articulation(robot.robot)
            out_info['gt_labels'] = 0

            if success:
                print(f'{ww} {jj}: grasp succ')
                grasp_succ += 1
                out_info['final_part_qpos'] = env.get_object_qpos().tolist()

                gt_motion = 0
                for ct in range(len(out_info['final_part_qpos'])):
                    gt_motion = max(gt_motion,
                                    out_info['final_part_qpos'][ct] - \
                                    out_info['start_part_qpos'][ct])

                if gt_motion > 0.01:
                    single_succ += 1
                    succ_cnt += 1
                    tot_cnt += 1
                    print(f'succ!!!!!!!!!!!   {succ_cnt}')
                    out_info['gt_labels'] = 1
                    with open(os.path.join(out_dir, 'result_more_%d.json' % (tot_cnt)), 'w') as fout:
                        json.dump(out_info, fout)
                    my_log.write('      %d: success | shape_id: %s | ww: %d | jj: %d\n' % (tot_cnt, shape_id, ww, jj))
                    if args.show_img and len(succ_images) > 0:
                        imageio.mimsave(os.path.join(succ_dir, 'pic_%s.gif' % (tot_cnt)), succ_images)
                else:
                    if grasp_cnt <= (succ_cnt // 2):
                        grasp_cnt += 1
                        tot_cnt += 1
                        out_info['gt_labels'] = 0
                        with open(os.path.join(out_dir, 'result_more_%d.json' % (tot_cnt)), 'w') as fout:
                            json.dump(out_info, fout)
                        my_log.write('    %d: grasp   | shape_id: %s | ww: %d | jj: %d\n' % (tot_cnt, shape_id, ww, jj))
                        if args.show_img and len(succ_images) > 0:
                            imageio.mimsave(os.path.join(grasp_dir, 'pic_%s.gif' % (ww)), succ_images)
            else:
                if fail_cnt <= (succ_cnt // 2):
                    fail_cnt += 1
                    tot_cnt += 1
                    out_info['gt_labels'] = 0
                    with open(os.path.join(out_dir, 'result_more_%d.json' % (tot_cnt)), 'w') as fout:
                        json.dump(out_info, fout)
                    my_log.write('  %d: fail    | shape_id: %s | ww: %d | jj: %d\n' % (tot_cnt, shape_id, ww, jj))
                    if args.show_img and len(succ_images) > 0:
                        imageio.mimsave(os.path.join(fail_dir, 'pic_%s.gif' % (ww)), succ_images)

            if single_succ > args.single_succ:
                print(f"single succ rate: {single_succ} / {jj + 1}")
                break

            end_time = time.time()
            if end_time - start_time > 100:
                break

        env.scene.remove_articulation(env.object)
        succ_rate = succ_cnt / tot_trial
        grasp_succ_rate = grasp_succ / tot_trial
        str = f'succ_rate: {succ_rate} grasp_rate: {grasp_succ_rate}'
        utils.printout(my_log, str)

    flog.close()
    env.close()


for idx_process in range(args.num_processes):
    print("start collect!")
    run_collect(idx_process, args.trial_id, args.shape_id, args.primact_type, args.out_dir)

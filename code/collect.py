"""
    For panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
        50% all parts closed, 50% middle (for each part, 50% prob. closed, 50% prob. middle)
        Simulate until static before starting
"""

import os
import sys
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
import warnings
import torch
import utils
from models.model_3d_w2a import Network

parser = ArgumentParser()
parser.add_argument('--cnt_id', type=int, default=0)
parser.add_argument('--shape_id', type=str, default='40147')
parser.add_argument('--category', type=str, default='StorageFurniture')
parser.add_argument('--primact_type', type=str, default='pulling')
parser.add_argument('--out_dir', type=str, default='../data')
parser.add_argument('--trial_id', type=int, default=0, help='trial id')
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--rand', default=False)
parser.add_argument('--num_processes', type=int, default=20)
parser.add_argument('--ww_range', type=int, default=20000)
parser.add_argument('--date', type=str, default='0905')
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--final_dist', type=float, default=0.2)
parser.add_argument('--min_mass', type=float, default=0.01)
parser.add_argument('--max_mass', type=float, default=0.05)
parser.add_argument('--true_thres', type=float, default=0.5)
parser.add_argument('--collect_num', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0',
                    help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--cuda', type=str, default='0, 1, 2, 3')
parser.add_argument('--state', type=str, default='closed')
parser.add_argument('--num_point_per_shape', type=int, default=10000)
parser.add_argument('--sample_type', type=str, default='fps')
parser.add_argument('--sapien_dir', type=str, default='../data')
parser.add_argument('--now_dist', type=float, default=2.5)
parser.add_argument('--nxt_dist', type=float, default=0.6)
parser.add_argument('--show_img', action='store_true', default=False)
parser.add_argument('--rand_view', action='store_true', default=False)
parser.add_argument('--rand_dir', action='store_true', default=False)
parser.add_argument('--far_samples', type=int, default=10000)
parser.add_argument('--near_samples', type=int, default=10000)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--w2a_dir', type=str, default='../logs/where2actPP')
parser.add_argument('--height', type=int, default=1080)
parser.add_argument('--width', type=int, default=1920)
parser.add_argument('--middle_free', action='store_true', default=False)
args = parser.parse_args()

device = args.device

import torch

train_conf = torch.load(os.path.join(args.w2a_dir, "logs/final_logs/finalexp-model_all_final-pulling-None-train_all_v1", 'conf.pth'))
network_pull = Network(feat_dim = train_conf.feat_dim, rv_dim = train_conf.rv_dim, rv_cnt = train_conf.rv_cnt)
data_to_restore = torch.load(os.path.join(args.w2a_dir, "logs/final_logs/finalexp-model_all_final-pulling-None-train_all_v1", 'ckpts', '81-network.pth'))
network_pull.load_state_dict(data_to_restore, strict=False)
network_pull.to(device)
network_pull.eval()


def run_collect(cnt_id=args.cnt_id, trial_id=args.trial_id, primact_type=args.primact_type,
                out_dir=args.out_dir):
    if args.test:
        out_dir = os.path.join(out_dir, '%s_pull_%s_val_%s' % (args.date, args.category, args.collect_num))
    else:
        out_dir = os.path.join(out_dir, '%s_pull_%s_%s' % (args.date, args.category, args.collect_num))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, 'process_%d' % (cnt_id))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_info = dict()
    train_file_dir = "../stats/train_data_list.txt"
    if args.test:
        train_file_dir = "../stats/test_data_list.txt"

    all_shape_list = []
    train_shape_list = []
    all_cat_list = [args.category]
    len_shape = {}
    len_train_shape = {}
    shape_cat_dict = {}
    cat_shape_id_list = {}
    train_cat_shape_id_list = {}
    for cat in all_cat_list:
        len_shape[cat] = 0
        len_train_shape[cat] = 0
        cat_shape_id_list[cat] = []
        train_cat_shape_id_list[cat] = []

    with open(train_file_dir, 'r') as fin:
        for l in fin.readlines():
            shape_id, cat = l.rstrip().split()
            if utils.is_ignored(shape_id):
                continue
            if cat not in all_cat_list:
                continue
            train_shape_list.append(shape_id)
            all_shape_list.append(shape_id)
            shape_cat_dict[shape_id] = cat
            len_shape[cat] += 1
            len_train_shape[cat] += 1
            cat_shape_id_list[cat].append(shape_id)
            train_cat_shape_id_list[cat].append(shape_id)

    np_random_seed = random.randint(1, 1000) + cnt_id
    np_random_seed = 1
    np.random.seed(np_random_seed)
    random_seed = random.randint(1, 1000) + cnt_id
    random_seed = 1
    random.seed(random_seed)

    flog = open(os.path.join(out_dir, 'log.txt'), 'w')
    env = Env(flog=flog, show_gui=(not args.no_gui))
    my_log = open(os.path.join(out_dir, 'my_log.txt'), 'w')
    my_log.write(' '.join(sys.argv))
    my_log.write('np random seed: %d\n' % np_random_seed)
    my_log.write('   random seed: %d\n' % random_seed)

    if args.rand_view:
        cam = Camera(env, random_position=True, restrict_dir=True, dist=args.now_dist)
    else:
        cam = Camera(env, theta=np.pi, phi=0.7826405702413783, dist=args.now_dist)
    cam1 = Camera(env, theta=np.pi, phi=0)
    out_info['root_cam_info'] = cam.get_metadata_json()

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

    for ww in range(args.ww_range):
        print(f'{cnt_id} {ww}')
        while True:
            selected_cat = all_cat_list[random.randint(0, len(all_cat_list) - 1)]
            shape_id = train_cat_shape_id_list[selected_cat][random.randint(0, len_train_shape[selected_cat] - 1)]
            out_info['shape_id'] = shape_id
            object_urdf_fn = '%s/where2act_original_sapien_dataset/%s/mobility_vhacd.urdf' % (args.sapien_dir, shape_id)
            flog.write('object_urdf_fn: %s\n' % object_urdf_fn)
            state = 'closed'
            load_object = env.load_object(object_urdf_fn, object_material, state=state, density=density)
            out_info['joint_angles_lower'] = env.joint_angles_lower
            out_info['joint_angles_upper'] = env.joint_angles_upper

            cur_qpos = env.get_object_qpos()
            # simulate some steps for the object to stay rest
            still_timesteps = 0
            wait_timesteps = 0
            while still_timesteps < 150 and wait_timesteps < 200:
                env.step()
                env.render()
                cur_new_qpos = env.get_object_qpos()
                invalid_contact = False
                for c in env.scene.get_contacts():
                    for p in c.points:
                        if abs(p.impulse @ p.impulse) > 1e-4:
                            invalid_contact = True
                            break
                    if invalid_contact:
                        break
                if np.max(np.abs(cur_new_qpos - cur_qpos)) < 1e-6 and (not invalid_contact):
                    still_timesteps += 1
                else:
                    still_timesteps = 0
                cur_qpos = cur_new_qpos
                wait_timesteps += 1
            if still_timesteps < 150:
                env.scene.remove_articulation(env.object)
                print("not still")
                continue
            out_info['cur_qpos'] = cur_qpos.tolist()
            cam.get_observation()
            break

        # simulate some steps for the object to stay rest
        env.step()
        env.render()

        rgb, pc_far, positions = cam.get_observation()

        mask = positions[..., -1]
        sample = np.random.permutation(len(pc_far))[:args.near_samples]
        pc_far = pc_far[sample, :3]

        x, y, position_world = select_point(mask, positions, cam, args.height, args.width,
                                            network=network_pull, random_rate=0.3,
                                            middle_select=False,
                                            pc=pc_far, num_point=args.far_samples,
                                            device=args.device, sample=sample)

        pc_far[0] = position_world
        pc2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_far))
        o3d.io.write_point_cloud(os.path.join(out_dir, f'pc_far_{tot_cnt + 1}.ply'), pc2)

        out_info['zoom_in_point'] = [int(x), int(y), position_world.tolist()]
        ################################ normal direction ##################################
        nxt_dist = args.nxt_dist
        out_info['nxt_dist'] = nxt_dist

        mat44 = cam1.mat44
        mat44[:3, 3] = position_world - mat44[:3, 0] * args.nxt_dist
        cam1.change_pose_by_mat(mat44)
        out_info['nxt_cam_info'] = cam1.get_metadata_json()

        env.render()

        view_dir = os.path.join(out_dir, 'view')
        succ_dir = os.path.join(out_dir, 'succ')
        grasp_dir = os.path.join(out_dir, 'grasp')
        fail_dir = os.path.join(out_dir, 'fail')
        if args.show_img:
            if not os.path.exists(view_dir):
                os.mkdir(view_dir)
            if not os.path.exists(succ_dir):
                os.mkdir(succ_dir)
            if not os.path.exists(grasp_dir):
                os.mkdir(grasp_dir)
            if not os.path.exists(fail_dir):
                os.mkdir(fail_dir)

        rgb_save, pc, positions = cam1.get_observation()

        mask = positions[..., -1]
        sample = np.random.permutation(len(pc))[:args.near_samples]
        pc = pc[sample, :3]
        x, y, position_world = select_point(mask, positions, cam1, args.height, args.width,
                                            network=network_pull, random_rate=0.3,
                                            middle_select=True, pc=pc, sample=sample,
                                            device=args.device)

        pc[0] = position_world
        pc2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
        o3d.io.write_point_cloud(os.path.join(out_dir, f'pc_near_{tot_cnt + 1}.ply'), pc2)

        out_info['manipulate_point'] = [int(x), int(y), position_world.tolist()]
        gt_nor = cam1.get_normal_map()
        pc_far[0] = position_world
        up, left, forward = get_direction(gt_nor, cam1, x, y,
                                          random_prob=0.2, pc_far=pc_far,
                                          network=network_pull, cam_far=cam)

        out_info['gripper_up'] = up.tolist()
        out_info['gripper_left'] = left.tolist()
        out_info['gripper_forward'] = forward.tolist()

        robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=(args.primact_type in primact_type))

        rotmat = np.eye(4).astype(np.float32)
        rotmat[:3, 0] = forward
        rotmat[:3, 1] = left
        rotmat[:3, 2] = up

        final_dist = 0.15
        out_info['dist'] = final_dist

        final_rotmat = np.array(rotmat, dtype=np.float32)
        final_rotmat[:3, 3] = position_world - up * 0.1 - cam1.mat44[:3, 0] * 0.15
        final_pose = Pose().from_transformation_matrix(final_rotmat)
        out_info['target_rotmat_world'] = final_rotmat.tolist()

        start_rotmat = np.array(rotmat, dtype=np.float32)
        start_rotmat[:3, 3] = position_world - up * 0.15
        start_pose = Pose().from_transformation_matrix(start_rotmat)
        out_info['start_rotmat_world'] = start_rotmat.tolist()

        end_rotmat = np.array(rotmat, dtype=np.float32)
        end_rotmat[:3, 3] = position_world - up * 0.1
        out_info['end_rotmat_world'] = end_rotmat.tolist()

        robot.robot.set_root_pose(start_pose)

        env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)

        ### main steps
        out_info['start_part_qpos'] = env.get_object_qpos().tolist()

        env.render()
        rgb_save1 = cam.get_rgb()

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
                    success = False
        except ContactError:
            success = False

        env.scene.remove_articulation(robot.robot)

        if success:
            out_info['final_part_qpos'] = env.get_object_qpos().tolist()
            out_info['gt_labels'] = 0
            print("success???????????????????????????")
            gt_motion = 0
            for ct in range(len(out_info['final_part_qpos'])):
                gt_motion = max(gt_motion,
                                out_info['final_part_qpos'][ct] - \
                                out_info['start_part_qpos'][ct])

            if gt_motion > 0.01:
                succ_cnt += 1
                tot_cnt += 1
                print("success!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                out_info['gt_labels'] = 1
                with open(os.path.join(out_dir, 'result_%d.json' % (tot_cnt)), 'w') as fout:
                    json.dump(out_info, fout)
                my_log.write('%d: success | shape_id: %s | ww: %d\n'%(tot_cnt, shape_id, ww))
                if args.show_img and len(succ_images) > 0:
                    imageio.mimsave(os.path.join(succ_dir, 'pic_%s.gif' % (ww)), succ_images)
            else:
                if grasp_cnt <= (succ_cnt // 2) or args.debug:
                    grasp_cnt += 1
                    tot_cnt += 1
                    out_info['gt_labels'] = 0
                    with open(os.path.join(out_dir, 'result_%d.json' % (tot_cnt)), 'w') as fout:
                        json.dump(out_info, fout)
                    my_log.write('%d: grasp   | shape_id: %s | ww: %d\n'%(tot_cnt, shape_id, ww))
                    if args.show_img and len(succ_images) > 0:
                        imageio.mimsave(os.path.join(grasp_dir, 'pic_%s.gif' % (ww)), succ_images)
        else:
            if fail_cnt <= (succ_cnt // 2) or args.debug:
                fail_cnt += 1
                tot_cnt += 1
                out_info['gt_labels'] = 0
                with open(os.path.join(out_dir, 'result_%d.json' % (tot_cnt)), 'w') as fout:
                    json.dump(out_info, fout)
                my_log.write('%d: fail    | shape_id: %s | ww: %d\n'%(tot_cnt, shape_id, ww))
                if args.show_img and len(succ_images) > 0:
                    imageio.mimsave(os.path.join(fail_dir, 'pic_%s.gif' % (ww)), succ_images)
        env.scene.remove_articulation(env.object)
        failure_rate = succ_cnt / (ww + 1)

    print(succ_cnt)
    my_log.close()
    flog.close()
    env.close()


def select_point(mask, cam_XYZA, cam, height, width, network,
                 random_rate=0.3, middle_select=False, pc=None,
                 num_point=10000, device='cuda:0', sample=None):
    if random.random() < random_rate or (middle_select and (not args.middle_free)):
        ############################ random select ctpt ############################################
        if middle_select:
            x = int((0.4 + random.random() * 0.2) * height)
            y = int((0.4 + random.random() * 0.2) * width)
            cnt = 0
            while mask[x, y] and cnt < 5:
                x = int((0.4 + random.random() * 0.2) * height)
                y = int((0.4 + random.random() * 0.2) * width)
                cnt += 1
            if cnt > 5:
                x = int(0.5 * height)
                y = int(0.5 * width)
        else:
            xs, ys = np.where(mask)
            idx = np.random.randint(len(xs))
            x, y = xs[idx], ys[idx]
        position_cam = cam_XYZA[x, y, :3]
        model_matrix = cam.camera.get_model_matrix()
        position_world = position_cam @ model_matrix[:3, :3].T + model_matrix[:3, 3]
        ###########################################################################################
    else:
        mat = np.linalg.inv(cam.mat44[:3, :3])
        pc_cam = (mat @ pc.T).T
        input_pcs = torch.tensor(pc_cam, dtype=torch.float32).reshape(1, num_point, 3).to(device)
        ################################### w2a select ctpt #######################################
        with torch.no_grad():
            pred_action_score_map = network.inference_action_score(input_pcs)[0]
            pred_action_score_map = pred_action_score_map.cpu().numpy()

        accu = 0.95
        idxx = np.where(pred_action_score_map > accu)[0]
        while len(idxx) < 100:
            accu = accu - 0.05
            idxx = np.where(pred_action_score_map > accu)[0]
        p_id = random.randint(0, len(idxx) - 1)
        idx = idxx[p_id]
        xs, ys = np.where(mask)
        x = xs[sample[idx]]
        y = ys[sample[idx]]
        position_world = pc[idx]
        ################################### finish w2a select ctpt ################################

    return x, y, position_world


def get_direction(gt_nor, cam, x, y, random_prob=0.2, pc_far=None,
                  network=None, cam_far=None, num_point=10000):

    direction_cam = -gt_nor[x, y, :3]
    direction_cam /= np.linalg.norm(direction_cam)
    if random.random() < random_prob:
        action_direction_cam = np.random.randn(3).astype(np.float32)
        action_direction_cam /= np.linalg.norm(action_direction_cam)
        while action_direction_cam @ direction_cam < 0.7:
            action_direction_cam = np.random.randn(3).astype(np.float32)
            action_direction_cam /= np.linalg.norm(action_direction_cam)
        direction_world = cam.mat44[:3, :3] @ action_direction_cam
        up = np.array(direction_world, dtype=np.float32)
        up /= np.linalg.norm(up)
        forward = np.random.randn(3).astype(np.float32)
        while abs(up @ forward) > 0.99:
            forward = np.random.randn(3).astype(np.float32)
    else:
        mat = np.linalg.inv(cam_far.mat44[:3, :3])
        pc_cam = (mat @ pc_far.T).T
        input_pcs = torch.tensor(pc_cam, dtype=torch.float32).reshape(1, num_point, 3).to(device)
        with torch.no_grad():
            pred_6d = network_pull.inference_actor(input_pcs)[0]  # RV_CNT x 6
            pred_Rs = network_pull.actor.bgs(pred_6d.reshape(-1, 3, 2)).detach().cpu().numpy()

        a_id = random.randint(0, len(pred_Rs) - 1)
        gripper_direction_camera = pred_Rs[a_id:a_id + 1, :, 0][0]
        gripper_forward_direction_camera = pred_Rs[a_id:a_id + 1, :, 1][0]

        up = gripper_direction_camera
        forward = gripper_forward_direction_camera
        up = cam_far.mat44[:3, :3] @ up
        forward = cam_far.mat44[:3, :3] @ forward
        up = np.array(up, dtype=np.float32)
        forward = np.array(forward, dtype=np.float32)

    ################################################################################
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward)

    return up, left, forward


for idx_process in range(0, args.num_processes):
    print(f'Start collect! Process: {idx_process}')
    run_collect(idx_process, args.trial_id, args.primact_type, args.out_dir)

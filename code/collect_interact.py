"""
    For panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
        50% all parts closed, 50% middle (for each part, 50% prob. closed, 50% prob. middle)
        Simulate until static before starting
"""

import os
import sys
import numpy as np
from argparse import ArgumentParser
import json

from camera import Camera
from env import Env, ContactError
import torch
from sensor import Sensor
import random
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from data import SAPIENVisionDataset
import utils

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
parser.add_argument('--far_samples', type=int, default=10000)
parser.add_argument('--near_samples', type=int, default=10000)
parser.add_argument('--sample_type', type=str, default='fps')
parser.add_argument('--sapien_dir', type=str, default='../data')
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--no_fine_part', action='store_true', default=False)
parser.add_argument('--random_zoom_in', action='store_true', default=False)
parser.add_argument('--random_fine_part', action='store_true', default=False)
parser.add_argument('--fixed', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--load_dir', type=str, default=None)
parser.add_argument('--load_epoch', type=int, default=20)
parser.add_argument('--actor_dir', type=str, default=None)
parser.add_argument('--actor_epoch', type=int, default=20)
parser.add_argument('--data_per_shape', type=int, default=10)
parser.add_argument('--feat_dim', type=int, default=128)
parser.add_argument('--rv_dim', type=int, default=10)
parser.add_argument('--rv_cnt', type=int, default=100)
parser.add_argument('--model_version', type=str, default='model_all')
args = parser.parse_args()

seed = random.randint(1, 1000)
np.random.seed(seed)
random.seed(seed)

train_file_dir = "../stats/train_data_list.txt"
if args.test:
    train_file_dir = "../stats/test_data_list.txt"
train_file_dir = '../stats/all.txt'
all_shape_list = []
train_shape_list = []
all_cat_list = [args.category]
tot_cat = len(all_cat_list)
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
        if cat not in all_cat_list:
            continue
        train_shape_list.append(shape_id)
        all_shape_list.append(shape_id)
        shape_cat_dict[shape_id] = cat
        len_shape[cat] += 1
        len_train_shape[cat] += 1
        cat_shape_id_list[cat].append(shape_id)
        train_cat_shape_id_list[cat].append(shape_id)

model_def = utils.get_model_module(args.model_version)
critic = model_def.Critic(args.feat_dim).to(args.device)
actor = model_def.Actor(feat_dim=args.feat_dim, rv_dim=args.rv_dim, rv_cnt=args.rv_cnt).to(args.device)
data_to_restore = torch.load(os.path.join("../logs", args.load_dir, 'ckpts', f"{args.load_epoch}-network.pth"))
critic.load_state_dict(data_to_restore)
data_to_restore = torch.load(os.path.join("../logs", args.actor_dir, 'ckpts', f"{args.actor_epoch}-actor.pth"))
actor.load_state_dict(data_to_restore)
critic.eval()
actor.eval()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)
traj_save = os.path.join(args.out_dir, 'process_0')
if not os.path.exists(traj_save):
    os.mkdir(traj_save)
pic_dir = os.path.join(args.out_dir, 'pics')
if not os.path.exists(pic_dir):
    os.mkdir(pic_dir)

my_log = open(os.path.join(args.out_dir, f'my_log.txt'), 'w')
utils.printout(my_log, ' '.join(sys.argv))
utils.printout(my_log, f'seed: {seed}')

flog = open(os.path.join(args.out_dir, f'log.txt'), 'w')
env = Env(flog=flog, show_gui=(not args.no_gui), use_kuafu=True)
object_material = env.get_material(4, 4, 0.01)
env.close()
tot = 0

shape_id_list = train_cat_shape_id_list[args.category]
idx = np.random.permutation(len(shape_id_list))
shape_id_list = np.array(shape_id_list)[idx].tolist()
shape_num = len(shape_id_list)
utils.printout(my_log, 'shape_id_list: ' + ' '.join(shape_id_list))

for i in range(shape_num):
    shape_id = shape_id_list[i]

    utils.printout(my_log, f'{shape_id}: {i + 1} / {shape_num}')

    if utils.is_ignored(shape_id):
        continue
    if (not args.test) and tot > 200:
        break

    env.rebuild()
    state = 'closed'
    sense = Sensor(env, theta=np.pi, phi=0.7826405702413783, dist=args.now_dist)
    sense1 = Sensor(env, theta=np.pi, phi=0, dist=args.now_dist)
    if args.no_fine_part:
        sense1 = Sensor(env, theta=np.pi, phi=0.7826405702413783, dist=args.now_dist)
    object_urdf_fn = f'{args.sapien_dir}/where2act_original_sapien_dataset/{shape_id}/mobility_vhacd.urdf'
    env.load_object(object_urdf_fn, object_material, state=state)\

    for ww in range(args.data_per_shape):
        sense.sensor.clear_cache()

        env.scene.step()
        env.scene.update_render()

        sense.sensor.take_picture()
        depth_far = sense.sensor.get_depth()
        pc_far_numpy = sense.depth_to_pc(depth_far)

        while 0 < len(pc_far_numpy) < args.far_samples:
            pc_far_numpy = np.concatenate((pc_far_numpy, pc_far_numpy), axis=0)

        sample = np.random.permutation(len(pc_far_numpy))[:args.far_samples]
        pc_far_numpy = pc_far_numpy[sample, :3]
        pc_far = torch.FloatTensor(pc_far_numpy).to(args.device).reshape(1, args.far_samples, 3)
        with torch.no_grad():
            far_scores = critic.get_aff_far(pc_far).reshape(-1).detach().cpu().numpy()

        if args.fixed:
            p_id = np.argmax(far_scores)
        else:
            score_bar = 0.97
            idxx = np.where(far_scores > score_bar)[0]
            while len(idxx) < 100:
                score_bar = score_bar - 0.02
                idxx = np.where(far_scores > score_bar)[0]
            p_id = idxx[random.randint(0, len(idxx) - 1)]

        if args.random_zoom_in:
            p_id = np.random.randint(10000)

        zoom_in_point = [None, None, pc_far_numpy[p_id].tolist()]
        pc_far_numpy[p_id] = pc_far_numpy[0]
        pc_far_numpy[0] = np.array(zoom_in_point[2])

        pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_far_numpy))
        o3d.io.write_point_cloud(os.path.join(traj_save, f'pc_far_real_{tot}.ply'), pc1)

        mat44 = sense1.mat44
        mat44[:3, 3] = pc_far_numpy[0] - mat44[:3, 0] * args.nxt_dist
        if not args.no_fine_part:
            sense1.change_pose_by_mat(mat44)

        env.scene.step()
        env.scene.update_render()
        sense1.sensor.take_picture()
        depth_near = sense1.sensor.get_depth()
        pc_near_numpy = sense1.depth_to_pc(depth_near)

        while 0 < len(pc_near_numpy) < args.near_samples:
            pc_near_numpy = np.concatenate((pc_near_numpy, pc_near_numpy), axis=0)

        sample = np.random.permutation(len(pc_near_numpy))[:args.near_samples]
        pc_near_numpy = pc_near_numpy[sample, :3]
        if pc_near_numpy.shape[0] == 0:
            action_dict = {}
            action_dict['gripper_up'] = None
            action_dict['gripper_forward'] = None
            action_dict['manipulate_point'] = None
            action_dict['shape_id'] = shape_id
            action_dict['zoom_in_point'] = zoom_in_point

            with open(os.path.join(traj_save, 'action_%d.json' % (tot)), 'w') as fout:
                json.dump(action_dict, fout)
            utils.printout(my_log, f"done shape id {shape_id} : {ww} / {args.data_per_shape}")
            continue

        pc_near = torch.FloatTensor(pc_near_numpy).to(args.device).reshape(1, args.near_samples, 3)

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
        if args.fixed:
            p_id = np.argmax(result)
        else:
            accu = 0.97
            xs = np.where(result > accu)[0]
            while len(xs) < 10:
                accu = accu - 0.02
                xs = np.where(result > accu)[0]
            p_id = xs[random.randint(0, len(xs) - 1)]
        near_feat = near_feats[p_id]

        if args.random_fine_part:
            p_id = np.random.randint(10000)

        with torch.no_grad():
            pred_6d = actor.inference_actor(near_feat)[0]  # RV_CNT x 6
            pred_Rs = actor.bgs(pred_6d.reshape(-1, 3, 2))
            up = pred_Rs[:, :, 0]
            forward = pred_Rs[:, :, 1]
            critic_result = critic.inference_critic_score_naction(up, forward, near_feat).reshape(-1)
            result = torch.max(critic_result, dim=0, keepdim=False)
            id = result.indices.item()
            up = up[id].detach().cpu().numpy()
            forward = forward[id].detach().cpu().numpy()

        manipulate_point = [None, None, pc_near_numpy[p_id].tolist()]
        pc_near_numpy[p_id] = pc_near_numpy[0]
        pc_near_numpy[0] = np.array(manipulate_point[2])

        pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_near_numpy))
        o3d.io.write_point_cloud(os.path.join(traj_save, f'pc_near_real_{tot}.ply'), pc1)

        action_dict = {}
        action_dict['gripper_up'] = up.tolist()
        action_dict['gripper_forward'] = forward.tolist()
        action_dict['manipulate_point'] = manipulate_point
        action_dict['shape_id'] = shape_id
        action_dict['zoom_in_point'] = zoom_in_point

        with open(os.path.join(traj_save, 'action_%d.json' % (tot)), 'w') as fout:
            json.dump(action_dict, fout)

        utils.printout(my_log, f"done shape id {shape_id} : {ww} / {args.data_per_shape}")
        tot += 1

    env.close()

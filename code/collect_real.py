"""
    For panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
        50% all parts closed, 50% middle (for each part, 50% prob. closed, 50% prob. middle)
        Simulate until static before starting
"""

import os
import numpy as np
from argparse import ArgumentParser

from env import Env, ContactError

from sensor import Sensor
import random
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from data import SAPIENVisionDataset

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
parser.add_argument('--state', type=str, default='closed')
parser.add_argument('--data_dir1', type=str, default=None)
parser.add_argument('--data_dir2', type=str, default=None)
parser.add_argument('--data_dir3', type=str, default=None)
parser.add_argument('--data_dir4', type=str, default=None)
parser.add_argument('--data_dir5', type=str, default=None)
parser.add_argument('--data_dir6', type=str, default=None)
parser.add_argument('--data_dir7', type=str, default=None)
parser.add_argument('--data_dir8', type=str, default=None)
args = parser.parse_args()

data_features = ['shape_id', 'zoom_in_view', 'zoom_in_point', 'manipulate_point', 'traj_save', 'save_number']
train_dataset = SAPIENVisionDataset([args.primact_type], data_features)
if args.data_dir1 is not None:
    train_dataset.load_data(args.data_dir1, real_collect=True)
if args.data_dir2 is not None:
    train_dataset.load_data(args.data_dir2, real_collect=True)
if args.data_dir3 is not None:
    train_dataset.load_data(args.data_dir3, real_collect=True)
if args.data_dir4 is not None:
    train_dataset.load_data(args.data_dir4, real_collect=True)
if args.data_dir5 is not None:
    train_dataset.load_data(args.data_dir5, real_collect=True)
if args.data_dir6 is not None:
    train_dataset.load_data(args.data_dir6, real_collect=True)
if args.data_dir7 is not None:
    train_dataset.load_data(args.data_dir7, real_collect=True)
if args.data_dir8 is not None:
    train_dataset.load_data(args.data_dir8, real_collect=True)

dict = {}
for ww in range(len(train_dataset.data_buffer)):
    gt_labels, up, left, forward, zoom_in_point, manipulate_point, \
        shape_id, zoom_in_view, traj_far, traj_near, traj_save, j, _ = \
        train_dataset.data_buffer[ww]
    if shape_id not in dict:
        dict[shape_id] = []
    dict[shape_id].append(ww)

flog = open(os.path.join(args.data_dir1, f'log.txt'), 'w')
env = Env(flog=flog, show_gui=(not args.no_gui))
object_material = env.get_material(4, 4, 0.01)
env.close()

now_data_cnt = 0
tot_data_cnt = len(train_dataset.data_buffer)

for shape_id in dict.keys():
    env.rebuild()
    state = 'closed'
    sense = Sensor(env, theta=np.pi, phi=0.7826405702413783, dist=args.now_dist)
    sense1 = Sensor(env, theta=np.pi, phi=0.7826405702413783, dist=args.now_dist)
    object_urdf_fn = f'{args.sapien_dir}/where2act_original_sapien_dataset/{shape_id}/mobility_vhacd.urdf'
    env.load_object(object_urdf_fn, object_material, state=state)

    cnt = 0
    tot = len(dict[shape_id])
    for ww in dict[shape_id]:
        cnt += 1
        now_data_cnt += 1
        sense1.sensor.clear_cache()

        gt_labels, up, left, forward, zoom_in_point, manipulate_point, \
            shape_id, zoom_in_view, traj_far, traj_near, traj_save, j, _ = \
            train_dataset.data_buffer[ww]

        sense1.change_pose_by_mat(np.array(zoom_in_view['mat44']))
        env.scene.step()
        env.scene.update_render()
        sense.sensor.take_picture()
        sense1.sensor.take_picture()

        depth_far = sense.sensor.get_depth()
        depth_near = sense1.sensor.get_depth()

        ctpt_far = sense.depth_point_to_pc(depth_far, zoom_in_point[0], zoom_in_point[1], zoom_in_point[2])
        pc_far = sense.depth_to_pc(depth_far)
        if len(pc_far) == 0:
            print("no pc far, can't tell why!!!")
            continue
        sample = np.random.permutation(len(pc_far))[:args.far_samples]
        pc_far = pc_far[sample, :3]
        pc_far[0] = ctpt_far
        pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_far))
        o3d.io.write_point_cloud(os.path.join(traj_save, f'pc_far_real_{j}.ply'), pc1)

        ctpt_near = sense1.depth_point_to_pc(depth_near, manipulate_point[0], manipulate_point[1], manipulate_point[2])
        pc_near = sense1.depth_to_pc(depth_near)
        if len(pc_near) == 0:
            print("no pc near, can't tell why!!!")
            continue
        sample = np.random.permutation(len(pc_near))[:args.near_samples]
        pc_near = pc_near[sample, :3]
        pc_near[0] = ctpt_near
        pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_near))
        o3d.io.write_point_cloud(os.path.join(traj_save, f'pc_near_real_{j}.ply'), pc1)

        print(os.path.join(traj_save, f'pc_near_real_{j}.ply'))
        print(f"done shape id {shape_id} : {cnt} / {tot}")
        print(f"done tot friction : {now_data_cnt} / {tot_data_cnt}")

    env.close()

"""
    SAPIENVisionDataset
        Joint data loader for six primacts
        for panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
"""

import os
import h5py
import torch
import torch.utils.data as data
import numpy as np
import json
from plyfile import PlyData, PlyElement
from camera import Camera
import math

class SAPIENVisionDataset(data.Dataset):

    def __init__(self, primact_types, data_features):
        self.primact_types = primact_types

        # data features
        self.data_features = data_features

        self.data_buffer = []  # (gripper_direction_world, gripper_action_dis, gt_motion)
        self.seq = []

    def load_data(self, dir, real=False, real_collect=False, succ_only=False):

        for i in range(3):
            print(f"process{i} : \n")
            cur_dir = os.path.join(dir, f'process_{i}')
            if not os.path.exists(cur_dir):
                continue
            for j in range(1, 3000):
                if not os.path.exists(os.path.join(cur_dir, f'result_{j}.json')):
                    continue
                if os.path.exists(os.path.join(cur_dir, f'pc_far_real_{j}.ply')) and real_collect:
                    continue
                if real and (not os.path.exists(os.path.join(cur_dir, f'pc_far_real_{j}.ply'))):
                    continue
                if real and (not os.path.exists(os.path.join(cur_dir, f'pc_near_real_{j}.ply'))):
                    continue
                with open(os.path.join(cur_dir, f'result_{j}.json'), 'r') as fin:
                    result_data = json.load(fin)
                    ignore_list = ['7265',
                                   '47466', '46172', '47585', '45516',
                                   '45600', '48721', '45087', '45642',
                                   '11211']
                    if result_data['shape_id'] in ignore_list:
                        continue
                    gt_labels = result_data['gt_labels']
                    up = result_data['gripper_up']
                    left = result_data['gripper_left']
                    forward = result_data['gripper_forward']
                    zoom_in_point = result_data['zoom_in_point']
                    manipulate_point = result_data['manipulate_point']
                    shape_id = result_data['shape_id']
                    zoom_in_view = result_data['nxt_cam_info']
                    if real:
                        traj_far = os.path.join(cur_dir, f'pc_far_real_{j}.ply')
                        traj_near = os.path.join(cur_dir, f'pc_near_real_{j}.ply')
                    else:
                        traj_far = os.path.join(cur_dir, f'pc_far_{j}.ply')
                        traj_near = os.path.join(cur_dir, f'pc_near_{j}.ply')

                    if (not succ_only) or gt_labels > 0.5:
                        self.data_buffer.append((gt_labels, up, left, forward, zoom_in_point, manipulate_point, shape_id, zoom_in_view, traj_far, traj_near, cur_dir, j, None))

    def load_data_more(self, dir, succ_only=False):

        for i in range(3):
            print(f"process{i} : \n")
            cur_dir = os.path.join(dir, f'process_{i}')
            if not os.path.exists(cur_dir):
                continue
            for j in range(1, 4000):
                if not os.path.exists(os.path.join(cur_dir, f'result_more_{j}.json')):
                    continue

                with open(os.path.join(cur_dir, f'result_more_{j}.json'), 'r') as fin:
                    result_data = json.load(fin)
                    gt_labels = result_data['gt_labels']
                    up = result_data['gripper_up']
                    left = result_data['gripper_left']
                    forward = result_data['gripper_forward']
                    zoom_in_point = result_data['zoom_in_point']
                    manipulate_point = result_data['manipulation_point']
                    shape_id = result_data['shape_id']
                    p_id = result_data['p_id']
                    zoom_in_view = None

                    traj_far = result_data['traj_far']
                    traj_near = result_data['traj_near']

                    if (not succ_only) or gt_labels > 0.5:
                        self.data_buffer.append((
                                                gt_labels, up, left, forward, zoom_in_point, manipulate_point, shape_id,
                                                zoom_in_view, traj_far, traj_near, cur_dir, j, p_id))


    def __str__(self):
        return "PhysicsDataLoader"

    def __len__(self):
        return len(self.data_buffer)

    def __getitem__(self, index):
        gt_labels, up, left, forward, zoom_in_point, manipulate_point, shape_id, zoom_in_view, traj_far, traj_near, traj_save, j, p_id = self.data_buffer[index]
        data_feats = ()
        plydata_far = PlyData.read(traj_far)
        pc_far = plydata_far['vertex'].data
        pc_far = np.array([[x, y, z] for x, y, z in pc_far], dtype=np.float32)
        plydata_near = PlyData.read(traj_near)
        pc_near = plydata_near['vertex'].data
        pc_near = np.array([[x, y, z] for x, y, z in pc_near], dtype=np.float32)
        while 0 < len(pc_near) < 10000:
            pc_near = np.concatenate((pc_near, pc_near), axis=0)
        if p_id is not None:
            ctpt = pc_near[p_id]
            pc_near[p_id] = pc_near[0]
            pc_near[0] = ctpt

        def dist(p1, p2):
            return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[2] - p2[2]) * (
                        p1[2] - p2[2]))

        for feat in self.data_features:
            if feat == 'gt_labels':
                if dist(pc_near[0], manipulate_point[2]) > 0.02:
                    gt_labels = 0
                out = gt_labels
                data_feats = data_feats + (out,)
            elif feat == "up":
                out = up
                data_feats = data_feats + (out,)
            elif feat == "left":
                out = left
                data_feats = data_feats + (out,)
            elif feat == "forward":
                out = forward
                data_feats = data_feats + (out,)
            elif feat == "zoom_in_point":
                out = zoom_in_point
                data_feats = data_feats + (out,)
            elif feat == "manipulate_point":
                out = manipulate_point
                data_feats = data_feats + (out,)
            elif feat == "shape_id":
                out = shape_id
                data_feats = data_feats + (out,)
            elif feat == "zoom_in_view":
                out = zoom_in_view
                data_feats = data_feats + (out,)
            elif feat == "pc_far":
                while pc_far.shape[0] < 10000:
                    pc_far = np.concatenate((pc_far, pc_far))
                pc_far = pc_far[:10000]
                out = torch.from_numpy(pc_far).unsqueeze(0)
                data_feats = data_feats + (out,)
            elif feat == "pc_near":
                while pc_near.shape[0] < 10000:
                    pc_near = np.concatenate((pc_near, pc_near))
                pc_near = pc_near[:10000]
                out = torch.from_numpy(pc_near).unsqueeze(0)
                data_feats = data_feats + (out,)
            elif feat == "traj_save":
                out = traj_save
                data_feats = data_feats + (out,)
            elif feat == "save_number":
                out = j
                data_feats = data_feats + (out,)
            elif feat == 'far_valid':
                if dist(pc_near[0], pc_far[0]) < 0.17 and (p_id is None):
                    out = True
                else:
                    out = False
                data_feats = data_feats + (out,)
            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats

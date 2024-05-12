"""
    Kuafu sensor
"""
import numpy as np
from sapien.core import Pose
from sapien.sensor import ActiveLightSensor


class Sensor(object):

    def __init__(self, env, dist=5.0, theta=np.pi, phi=np.pi/10):
        self.env = env
        # set sensor intrinsics
        self.sensor = ActiveLightSensor('sensor', env.renderer, env.scene, sensor_type='fakesense_j415')

        pos = np.array([dist*np.cos(phi)*np.cos(theta),
                dist*np.cos(phi)*np.sin(theta),
                dist*np.sin(phi)])
        forward = -pos / np.linalg.norm(pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.vstack([forward, left, up]).T
        mat44[:3, 3] = pos      # mat44 is cam2world
        mat44[0, 3] += env.object_position_offset
        self.mat44 = mat44

        self.dist = dist
        self.theta = theta
        self.phi = phi
        self.pos = pos

        self.sensor.set_pose(Pose.from_transformation_matrix(mat44))

    def change_pose_by_mat(self, mat44):
        self.mat44 = mat44
        self.sensor.set_pose(Pose.from_transformation_matrix(mat44))
        self.pos = mat44[:3, 3]
        self.dist = None
        self.theta = None
        self.phi = None

    def get_observation(self):
        rgb = self.sensor.get_rgb()
        depth = self.sensor.get_depth()
        pc = self.depth_to_pc(depth)
        return rgb, depth, pc

    def depth_to_pc(self, depth_map):
        cam_extrinsic = self.sensor._pose2cv2ex(self.sensor.pose)
        feature_grid = ActiveLightSensor._get_pixel_grids_np(
            depth_map.shape[0], depth_map.shape[1])

        uv = np.matmul(np.linalg.inv(self.sensor.rgb_intrinsic), feature_grid)
        cam_points = np.reshape(depth_map, (1, -1))
        _, mask = np.where(cam_points > 0.2)
        cam_points = uv * cam_points
        cam_points = cam_points[:, mask]

        r = cam_extrinsic[:3, :3]
        t = cam_extrinsic[:3, 3:4]
        r_inv = np.linalg.inv(r)

        world_points = np.matmul(r_inv, cam_points - t).transpose()
        return world_points

    def depth_point_to_pc(self, depth_map, x, y, pos_world, k=100):
        cam_extrinsic = self.sensor._pose2cv2ex(self.sensor.pose)
        feature_grid = ActiveLightSensor._get_pixel_grids_np(
            depth_map.shape[0], depth_map.shape[1])

        uv = np.matmul(np.linalg.inv(self.sensor.rgb_intrinsic), feature_grid)
        uv = uv.reshape(3, 1080, 1920)
        xl = max(0, x - k)
        xr = min(1080, x + k)
        yl = max(0, y - k)
        yr = min(1920, y + k)
        uv = uv[:, xl:xr, yl:yr].reshape(3, -1)
        cam_points = depth_map[xl:xr, yl:yr].reshape(1, -1)
        cam_points = uv * cam_points

        r = cam_extrinsic[:3, :3]
        t = cam_extrinsic[:3, 3:4]
        r_inv = np.linalg.inv(r)

        points = np.matmul(r_inv, cam_points - t).transpose()

        dist = np.linalg.norm(points - pos_world, ord=2, axis=1)
        index = np.argmin(dist)
        return points[index]

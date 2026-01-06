'''The goal of this module is to provide a dataset class for the CARDSet dataset on the RoadBEV model.'''
import os
import open3d as o3d
from typing import Any, Callable, Optional, Dict
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
import copy
import json, re, cv2
import PIL
from torch.utils.data import random_split
from torch.utils.data import Subset



def pose_to_matrix(x, y, z, qw, qx, qy, qz):
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4, dtype=np.float32); T[:3,:3] = R; T[:3,3] = [x,y,z]
    return T

def _slerp_scalar_first(q0, q1, t):
    q0_xyzw = np.array([q0[1], q0[2], q0[3], q0[0]])
    q1_xyzw = np.array([q1[1], q1[2], q1[3], q1[0]])
    Rends = Rotation.from_quat(np.stack([q0_xyzw, q1_xyzw], 0))
    s = Slerp([0.0, 1.0], Rends)
    q_xyzw = s([t]).as_quat()[0]
    return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]

def interpolate_pose(traj: dict, ts_us: int):
    poses = traj["trajectory_poses"]  # [[ts, [x,y,z,qw,qx,qy,qz]], ...]
    i = 0
    while i < len(poses) and poses[i][0] <= ts_us:
        i += 1
    if i == 0: return poses[0][1]
    if i >= len(poses): return poses[-1][1]
    ts0, p0 = poses[i-1]; ts1, p1 = poses[i]
    t = float(np.clip((ts_us - ts0) / max(1, (ts1 - ts0)), 0.0, 1.0))
    trans = (1.0 - t)*np.array(p0[:3]) + t*np.array(p1[:3])
    quat = _slerp_scalar_first(p0[3:], p1[3:], t)
    return [*trans.astype(float).tolist(), *quat]


class CARDSetDataset(Dataset):

    def __init__(self, 
                 root_dir: str,
                 mode: str = 'train',
                 down_scale = 4):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        
        #work with the model with nardo_sp2_bumps, nardo_sp1, nardo_sp3_bumps
        scenes = os.listdir(root_dir)
        scenes_sb = []
        for scene in scenes:
            if 'bump' in str(scene):
                scenes_sb.append(os.path.join(root_dir, scene))
        
        self.scenes_sb = scenes_sb[:6] #take only the first four this should be enough

        
        self.make_pairs()
        length = len(self.pairs)
        train_len = length * 70 // 100
        val_len = length - train_len
        self.train_pairs, self.test_pairs = random_split(
            self.pairs,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42)
        )
        #self.train_pairs = self.pairs[:length*70//100]
        #select everey 5th pair fo rtraining
        new_indices = self.train_pairs.indices[::5]
        self.train_pairs = Subset(self.train_pairs.dataset, new_indices)
        #self.test_pairs = self.pairs[length*70//100:]
        new_indices = self.test_pairs.indices[::50]
        self.test_pairs = Subset(self.train_pairs.dataset, new_indices)
        #self.test_pairs = self.test_pairs[::15]
        self.down_scale = down_scale

        self.base_height = 1.857  # in meter, the reference height of the camera w.r.t. road surface
        self.y_range = 0.2  # in meter, the range of interest above and below the base height， i.e., [-20cm, 20cm]
        self.roi_x = torch.tensor([-1, 0.92])    # in meter, the lateral range of interest (in the horizontal coordinate of camera)
        self.roi_z = torch.tensor([2.16, 7.08])    # in meter, the longitudinal range of interest

        self.grid_res = torch.tensor([0.03, 0.01, 0.03])  # in [x, y(vertical), z] order. The range of interest above should be integer times of resolution here
        

        self.num_grids_x = int((self.roi_x[1] - self.roi_x[0]) / self.grid_res[0])
        self.num_grids_z = int((self.roi_z[1] - self.roi_z[0]) / self.grid_res[2])
        self.num_grids_y = int(self.y_range*2 / self.grid_res[1])

        # generate the centers of every horizontal grid
        hori_centers = torch.zeros((self.num_grids_z, self.num_grids_x, 2), dtype=torch.float32)
        hori_centers[:, :, 0] = (torch.arange(self.num_grids_x) * self.grid_res[0] + self.roi_x[0] + self.grid_res[0]/2).unsqueeze(0).repeat([self.num_grids_z, 1])
        hori_centers[:, :, 1] = (-torch.arange(self.num_grids_z) * self.grid_res[2] + self.roi_z[1] - self.grid_res[2]/2).unsqueeze(1).repeat([1, self.num_grids_x])
        self.map_centers = hori_centers.reshape(-1, 2)
        self.num_center = self.map_centers.shape[0]

        # generate the centers of every 3D voxel
        voxel_centers = torch.zeros((self.num_grids_z, self.num_grids_x, self.num_grids_y, 3), dtype=torch.float32)
        voxel_centers[:, :, :, [0, 2]] = hori_centers.unsqueeze(2).repeat([1, 1, self.num_grids_y, 1])
        voxel_centers[:, :, :, 1] = (torch.arange(self.num_grids_y) * self.grid_res[1] + self.base_height - self.y_range + self.grid_res[1]/2).unsqueeze(0).unsqueeze(0).repeat([self.num_grids_z, self.num_grids_x, 1])
        self.voxel_centers = voxel_centers.reshape(-1, 3).transpose(1, 0)
        
        self.transform_jpg = transforms.Compose([
            transforms.ToTensor(),  # image --> [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0, 1] --> [-1, 1]
        ])
    
    # def __init__(self, 
    #              root_dir: str,
    #              mode: str = 'train',
    #              down_scale = 4):
    #     super().__init__()
    #     self.root_dir = root_dir
    #     self.mode = mode
        

    #     self.img_dir = os.path.join(root_dir, 'img', 'cam_1')
    #     self.depth_dir = os.path.join(root_dir, 'agg_depth', 'cam_1')
    #     self.image_files = sorted(os.listdir(self.img_dir), key=lambda x: (len(x), x))
    #     self.depth_files = sorted(os.listdir(self.depth_dir), key=lambda x: (len(x), x))

    #     self.down_scale = down_scale

    #     self.base_height = 1.857  # in meter, the reference height of the camera w.r.t. road surface
    #     self.y_range = 0.2  # in meter, the range of interest above and below the base height， i.e., [-20cm, 20cm]
    #     self.roi_x = torch.tensor([-1, 0.92])    # in meter, the lateral range of interest (in the horizontal coordinate of camera)
    #     self.roi_z = torch.tensor([5.16, 10.08])    # in meter, the longitudinal range of interest

    #     self.grid_res = torch.tensor([0.03, 0.01, 0.03])  # in [x, y(vertical), z] order. The range of interest above should be integer times of resolution here
        

    #     self.num_grids_x = int((self.roi_x[1] - self.roi_x[0]) / self.grid_res[0])
    #     self.num_grids_z = int((self.roi_z[1] - self.roi_z[0]) / self.grid_res[2])
    #     self.num_grids_y = int(self.y_range*2 / self.grid_res[1])

    #     len_images = len(self.image_files)
    #     len_depths = len(self.depth_files)
    #     assert len_images == len_depths, f"The number of images and depth files should be the same.{len_images} != {len_depths}"
    #     if(mode not in ['train', 'test']):
    #         raise ValueError("mode should be either 'train' or 'test'")
        
    #     elif mode == 'train':
            
    #         self.image_files = self.image_files[:len_images*7//10]
    #         self.depth_files = self.depth_files[:len_images*7//10]
    #     else:
    #         self.image_files = self.image_files[len_images*7//10:]
    #         self.depth_files = self.depth_files[len_images*7//10:]

    #     # generate the centers of every horizontal grid
    #     hori_centers = torch.zeros((self.num_grids_z, self.num_grids_x, 2), dtype=torch.float32)
    #     hori_centers[:, :, 0] = (torch.arange(self.num_grids_x) * self.grid_res[0] + self.roi_x[0] + self.grid_res[0]/2).unsqueeze(0).repeat([self.num_grids_z, 1])
    #     hori_centers[:, :, 1] = (-torch.arange(self.num_grids_z) * self.grid_res[2] + self.roi_z[1] - self.grid_res[2]/2).unsqueeze(1).repeat([1, self.num_grids_x])
    #     self.map_centers = hori_centers.reshape(-1, 2)
    #     self.num_center = self.map_centers.shape[0]

    #     # generate the centers of every 3D voxel
    #     voxel_centers = torch.zeros((self.num_grids_z, self.num_grids_x, self.num_grids_y, 3), dtype=torch.float32)
    #     voxel_centers[:, :, :, [0, 2]] = hori_centers.unsqueeze(2).repeat([1, 1, self.num_grids_y, 1])
    #     voxel_centers[:, :, :, 1] = (torch.arange(self.num_grids_y) * self.grid_res[1] + self.base_height - self.y_range + self.grid_res[1]/2).unsqueeze(0).unsqueeze(0).repeat([self.num_grids_z, self.num_grids_x, 1])
    #     self.voxel_centers = voxel_centers.reshape(-1, 3).transpose(1, 0)
    #     print(fr"voxel_centers shape{voxel_centers.shape}")
        
    #     self.transform_jpg = transforms.Compose([
    #         transforms.ToTensor(),  # image --> [0, 1]
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0, 1] --> [-1, 1]
    #     ])
    
    def make_pairs(self):
        """ self.pairs= []
        for scene in self.scenes_sb:
            img_dir = os.path.join(scene, "img", "cam_1")
            agg_dir = os.path.join(scene, "agg_depth", "cam_1")
            for img in sorted(os.listdir(img_dir), key= lambda x: (len(x), x)):
                name= img.split('.')[0]
                for agg in os.listdir(agg_dir):                   
                    if name in agg:
                        self.pairs.append((img, agg))
 """
        self.pairs= []
        for scene in self.scenes_sb:
            img_dir = Path(scene) / "img" / "cam_1"
            agg_dir = Path(scene) / "agg_depth" / "cam_1"

            if not (img_dir.is_dir() and agg_dir.is_dir()):
                continue

            # Index agg by stem
            agg_index = {p.stem: p for p in agg_dir.iterdir() if p.is_file()}

            # Sort images as you specified
            img_files = sorted(
                (p for p in img_dir.iterdir() if p.is_file()),
                key=lambda p: (len(p.name), p.name)
            )

            for img_path in img_files:
                agg_path = agg_index.get(img_path.stem)
                if agg_path:
                    self.pairs.append((img_path, agg_path))

                    

    def __len__(self):
            if self.mode == "train":
                return len(self.train_pairs)
            else:
                return len(self.test_pairs)
            
    
    
    def matrix2euler(self, m):
        # order='XYZ'
        d = np.clip
        m = m.reshape(-1)
        a, f, g, k, l, n, e = m[0], m[1], m[2], m[4], m[5], m[7], m[8]
        y = np.arcsin(d(g, -1, 1))
        if 0.99999 > np.abs(g):
            x = np.arctan2(- l, e)
            z = np.arctan2(- f, a)
        else:
            x = np.arctan2(n, k)
            z = 0
        return np.array([x, y, z], dtype=np.float32)

    def crop_image (self, K, image):
        W, H = image.size
        w_c = 960
        h_c = 528
        u_0 = (W - w_c) / 2
        v_0 = (H - h_c) / 2
        u_1 = (W + w_c) / 2
        v_1 = (H + h_c) / 2
        #imgs_left = image.crop((u_0, v_0, u_1, v_1))
        imgs_left = image.resize((w_c, h_c))
        scale_x = W/w_c
        scale_y = H/h_c
        intrinsic = K
        intrinsic[0:2] = intrinsic[0:2] / scale_x  #cx
        intrinsic[1:2] = intrinsic[1:2] / scale_y  #cy
        intrinsic[0:0] = intrinsic[0:0] / scale_x  #fx
        intrinsic[1:1] = intrinsic[1:1] / scale_y  #fy
       

        return imgs_left, intrinsic 
    
    def __getitem__(self, idx):
        img_path = self.train_pairs[idx][0] if self.mode == 'train' else self.test_pairs[idx][0]
        depth_path =self.train_pairs[idx][1] if self.mode == 'train' else self.test_pairs[idx][1]
        imgs_left = Image.open(img_path)
        print(img_path)
        print(depth_path)

        #get the camera parameter of the camera
        extrinsic, intrinsic, _, ground_info = self.get_cam_payload(str(img_path))
        extrinsic_inv = np.linalg.inv(extrinsic)
        
        imgs_left, intrinsic = self.crop_image(intrinsic, imgs_left)

        if idx == 0:
            imgs_left.save("mon_image.jpg")
        imgs_left = self.transform_jpg(imgs_left)
                
        #print("mbouuuu", imgs_left.shape)


        self.ground_normal_vector = ground_info['plane_normal_world']
        self.camera_height = ground_info['camera_height_above_ground']

        #print(depth_path)
        data = np.load(depth_path)
        points = data['pts_cam'] # N*3 array
        print(f"Height before values (y): min={points[:, 1].min()}, max={points[:, 1].max()}")
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        R = coord_frame.get_rotation_matrix_from_xyz((0.0, 0.0, np.pi))  # (Rx, Ry, Rz) in radians
        coord_frame = coord_frame.rotate(R, center=(0, 0, 0))


        #Transform the pts in plane coordinate system: Hg = Hc*N + delta(camera height - ground height)
        #if idx == 0:
        #draw_voxel_bounding_boxes(img_path, self.voxel_centers, intrinsic, extrinsic, down_scale=self.down_scale)


        ####
        [pitch_cam, roll_cam, yaw_cam] = self.matrix2euler((extrinsic[:3, :3].numpy()).T)  #transpose to get the inverse of the rotation
        #print(pitch_cam, roll_cam, yaw_cam)
        pitch_cam -= 1.5708  # pi/2
        R_X = np.array(
            [[1, 0, 0], [0, np.cos(pitch_cam), np.sin(pitch_cam)], [0, -np.sin(pitch_cam), np.cos(pitch_cam)]], dtype=np.float32)
        R_Z = np.array(
            [[np.cos(roll_cam), np.sin(roll_cam), 0], [-np.sin(roll_cam), np.cos(roll_cam), 0], [0, 0, 1]], dtype=np.float32)
        R_cam2vert = R_X @ R_Z  # the rotation matrix from the current camera coord to the vertical status
        R_vert2cam = torch.from_numpy(np.linalg.inv(R_cam2vert))
        ####
        
        R_cam2_vert = R_cam2vert
        #points[:, 1] -= self.camera_height
        R_vert2_cam = np.linalg.inv(R_cam2vert)

        camera_frame = copy.deepcopy(coord_frame)

        camera_frame.rotate(R_cam2vert, center=(0, 0, 0))

        # Ensure voxel_centers is [N, 3]
        voxel_centers = self.voxel_centers  # if currently [3, N]


        voxel_cam_left = torch.tensor(R_vert2_cam) @ voxel_centers

        # Downscale intrinsics
        intrinsic_downscaled = (intrinsic / self.down_scale).to(torch.float32)
        intrinsic_downscaled[2, 2] = 1
        uvz_left =  intrinsic_downscaled @ voxel_cam_left
        voxel_uv_left = torch.floor(uvz_left[:2, :] / uvz_left[2:, :]).type(torch.long)

        #print(f"voxel_uv_left{voxel_uv_left.shape}")
        #print(voxel_uv_left[:, -1])
        #print(voxel_uv_left[:, 16000])

        pcd_cam2vert = o3d.geometry.PointCloud()
        pcd_cam2vert.points = o3d.utility.Vector3dVector(points)
        pcd_cam2vert.rotate(R_cam2_vert, center=(0, 0, 0))
        #self.save_gt_as_image(pcd_cam2vert, img_path, intrinsic)

        #print("fjklnnini", ((np.asarray(pcd_cam2vert.points))[:, 1]).max())
        crop_bounding = np.array([[self.roi_x[0], 0, self.roi_z[0]],
                                   [self.roi_x[0], 0, self.roi_z[1]],
                                   [self.roi_x[1], 0, self.roi_z[1]],
                                   [self.roi_x[1], 0, self.roi_z[0]]]).astype("float64")
        points = np.asarray(pcd_cam2vert.points)
        #print(f"Height after transformation values (y): min={points[:, 1].min()}, max={points[:, 1].max()}")
        vol = o3d.visualization.SelectionPolygonVolume()
        vol.orthogonal_axis = "Y"
        vol.axis_max = 5
        vol.axis_min = 0
        vol.bounding_polygon = o3d.utility.Vector3dVector(crop_bounding)
        #self.save_gt_points(pcd_cam2vert, "before_crop" + str(img_path.stem) + ".png")
        pcd_cam2vert = vol.crop_point_cloud(pcd_cam2vert)
        points = np.asarray(pcd_cam2vert.points)
        #self.save_gt_points(pcd_cam2vert, str(img_path.stem) + ".png")
        #o3d.visualization.draw_geometries([camera_frame, coord_frame, pcd_cam2vert])
        print(f"Height after crop values (y): min={points[:, 1].min()}, max={points[:, 1].max()}")

        app = o3d.visualization.gui.Application.instance
        app.initialize()
        #viz = o3d.visualization.O3DVisualizer("Scene", 1280, 720)

        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        enu = copy.deepcopy(camera)
        enu.rotate(extrinsic[:3, :3], center=(0,0,0))
        
        road = copy.deepcopy(camera)
        road.rotate(R_cam2_vert, center=(0,0,0))
        #viz.add_geometry("camera", camera)
        #viz.add_geometry("enu", enu)
        #viz.add_geometry("road", road)
        #viz.show_settings = True
        #app.add_window(viz)
        #app.run()
        # Project to height map
        height_map, mask = self.get_gt_elevation(pcd_cam2vert)
        #print(f"croppoint count{np.asarray(pcd_cam2vert.points)[:, 1].max()}")
        #if idx % 10 == 0:
            #self.visualize_height_map_and_mask(height_map, mask, colormap='viridis', save_path='Heightmap/height_map_visualization' + str(img_path.stem) + 'png')
        #print(height_map.shape, height_map[:, 1])
        
        # Convert to tensors
        ele_gt = height_map # (H, W)
     
        mask = mask.type(torch.float32)
        # if idx == 0:
        #     print(f"elevation mask size{mask.shape}")

        
        #     print("imgs_left:", type(imgs_left), getattr(imgs_left, 'shape', 'No shape'))
        #     print("ele_gt:", type(ele_gt), getattr(ele_gt, 'shape', 'No shape'))
        #     print("mask:", type(mask), getattr(mask, 'shape', 'No shape'))
        #     print("voxel_uv_left:", type(voxel_uv_left), getattr(voxel_uv_left, 'shape', 'No shape'))
        #     print("timestamp_us:", ground_info.get('timestamp_us', 'Not found'))
        #     print(f"voxel_uv_left shape: {voxel_uv_left.shape}")
        #     print(f"voxel_uv_left min: {voxel_uv_left.min(dim=0).values}, max: {voxel_uv_left.max(dim=0).values}")


            
        return imgs_left, ele_gt, mask, voxel_uv_left, ground_info['timestamp_us']
        print(img_path)

        return None
   # def __getitem__(self, idx):
        # Load image from cam_1
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        print(img_path)
        imgs_left = Image.open(img_path)


        #get the camera parameter of the camera
        extrinsic, intrinsic, _, ground_info = self.get_cam_payload(str(img_path))
        extrinsic_inv = np.linalg.inv(extrinsic)
        
        imgs_left, intrinsic = self.crop_image(intrinsic, imgs_left)

        if idx == 0:
            imgs_left.save("mon_image.jpg")
        imgs_left = self.transform_jpg(imgs_left)
                
        print("mbouuuu", imgs_left.shape)


        self.ground_normal_vector = ground_info['plane_normal_world']
        self.camera_height = ground_info['camera_height_above_ground']

        print(extrinsic)
        # Load point cloud from npz and crop it
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        print(depth_path)
        data = np.load(depth_path)
        points = data['pts_cam'] # N*3 array
        print(f"Height before values (y): min={points[:, 1].min()}, max={points[:, 1].max()}")
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        R = coord_frame.get_rotation_matrix_from_xyz((0.0, 0.0, np.pi))  # (Rx, Ry, Rz) in radians
        coord_frame = coord_frame.rotate(R, center=(0, 0, 0))


        #Transform the pts in plane coordinate system: Hg = Hc*N + delta(camera height - ground height)
        #if idx == 0:
        #draw_voxel_bounding_boxes(img_path, self.voxel_centers, intrinsic, extrinsic, down_scale=self.down_scale)


        ####
        [pitch_cam, roll_cam, yaw_cam] = self.matrix2euler((extrinsic[:3, :3].numpy()).T)  #transpose to get the inverse of the rotation
        print(pitch_cam, roll_cam, yaw_cam)
        pitch_cam -= 1.5708  # pi/2
        R_X = np.array(
            [[1, 0, 0], [0, np.cos(pitch_cam), np.sin(pitch_cam)], [0, -np.sin(pitch_cam), np.cos(pitch_cam)]], dtype=np.float32)
        R_Z = np.array(
            [[np.cos(roll_cam), np.sin(roll_cam), 0], [-np.sin(roll_cam), np.cos(roll_cam), 0], [0, 0, 1]], dtype=np.float32)
        R_cam2vert = R_X @ R_Z  # the rotation matrix from the current camera coord to the vertical status
        R_vert2cam = torch.from_numpy(np.linalg.inv(R_cam2vert))
        ####
        
        R_cam2_vert = R_cam2vert
        #points[:, 1] -= self.camera_height
        R_vert2_cam = np.linalg.inv(R_cam2vert)

        camera_frame = copy.deepcopy(coord_frame)

        camera_frame.rotate(R_cam2vert, center=(0, 0, 0))

        # Ensure voxel_centers is [N, 3]
        voxel_centers = self.voxel_centers  # if currently [3, N]


        voxel_cam_left = torch.tensor(R_vert2_cam) @ voxel_centers

        # Downscale intrinsics
        intrinsic_downscaled = torch.tensor(intrinsic / self.down_scale, dtype=torch.float32)
        intrinsic_downscaled[2, 2] = 1
        uvz_left =  intrinsic_downscaled @ voxel_cam_left
        voxel_uv_left = torch.floor(uvz_left[:2, :] / uvz_left[2:, :]).type(torch.long)

        print(f"voxel_uv_left{voxel_uv_left.shape}")
        print(voxel_uv_left[:, -1])
        print(voxel_uv_left[:, 16000])

        pcd_cam2vert = o3d.geometry.PointCloud()
        pcd_cam2vert.points = o3d.utility.Vector3dVector(points)
        pcd_cam2vert.rotate(R_cam2_vert, center=(0, 0, 0))
        print("fjklnnini", ((np.asarray(pcd_cam2vert.points))[:, 1]).max())
        crop_bounding = np.array([[self.roi_x[0], 0, self.roi_z[0]],
                                   [self.roi_x[0], 0, self.roi_z[1]],
                                   [self.roi_x[1], 0, self.roi_z[1]],
                                   [self.roi_x[1], 0, self.roi_z[0]]]).astype("float64")
        points = np.asarray(pcd_cam2vert.points)
        print(f"Height after transformation values (y): min={points[:, 1].min()}, max={points[:, 1].max()}")
        vol = o3d.visualization.SelectionPolygonVolume()
        vol.orthogonal_axis = "Y"
        vol.axis_max = 5
        vol.axis_min = 0
        vol.bounding_polygon = o3d.utility.Vector3dVector(crop_bounding)
        #self.save_gt_points(pcd_cam2vert, "before_crop" + str(self.image_files[idx]))
        pcd_cam2vert = vol.crop_point_cloud(pcd_cam2vert)
        #self.save_gt_as_image(pcd_cam2vert, img_path, intrinsic)
        points = np.asarray(pcd_cam2vert.points)
        #self.save_gt_points(pcd_cam2vert, str(self.image_files[idx]))
        #o3d.visualization.draw_geometries([camera_frame, coord_frame, pcd_cam2vert])
        print(f"Height after crop values (y): min={points[:, 1].min()}, max={points[:, 1].max()}")

        app = o3d.visualization.gui.Application.instance
        app.initialize()
        #viz = o3d.visualization.O3DVisualizer("Scene", 1280, 720)

        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        enu = copy.deepcopy(camera)
        enu.rotate(extrinsic[:3, :3], center=(0,0,0))
        
        road = copy.deepcopy(camera)
        road.rotate(R_cam2_vert, center=(0,0,0))
        #viz.add_geometry("camera", camera)
        #viz.add_geometry("enu", enu)
        #viz.add_geometry("road", road)
        #viz.show_settings = True
        #app.add_window(viz)
        #app.run()
        # Project to height map
        height_map, mask = self.get_gt_elevation(pcd_cam2vert)
        print(f"croppoint count{np.asarray(pcd_cam2vert.points)[:, 1].max()}")
        #self.visualize_height_map_and_mask(height_map, mask, colormap='viridis', save_path='Heightmap/height_map_visualization.png' + str(self.image_files[idx]))
        print(height_map.shape, height_map[:, 1])
        
        # Convert to tensors
        ele_gt = height_map # (H, W)
        if (idx == 0):
            print(f"elevation size{ele_gt.shape}")
        mask = mask.type(torch.float32)
        if idx == 0:
            print(f"elevation mask size{mask.shape}")

        
            print("imgs_left:", type(imgs_left), getattr(imgs_left, 'shape', 'No shape'))
            print("ele_gt:", type(ele_gt), getattr(ele_gt, 'shape', 'No shape'))
            print("mask:", type(mask), getattr(mask, 'shape', 'No shape'))
            print("voxel_uv_left:", type(voxel_uv_left), getattr(voxel_uv_left, 'shape', 'No shape'))
            print("timestamp_us:", ground_info.get('timestamp_us', 'Not found'))
            print(f"voxel_uv_left shape: {voxel_uv_left.shape}")
            print(f"voxel_uv_left min: {voxel_uv_left.min(dim=0).values}, max: {voxel_uv_left.max(dim=0).values}")


            
        return imgs_left, ele_gt, mask, voxel_uv_left, ground_info['timestamp_us']


    def visualize_height_map_and_mask(self, height_map, mask, colormap='plasma', save_path='Heightmap/height_map_visualization.png'):
        """
        Visualize the height map and mask. Cells without points are black, and cells with height GT are mapped with a colormap.

        Parameters:
            height_map (torch.Tensor): The height map tensor of shape (H, W).
            mask (torch.Tensor): The mask tensor of shape (H, W), where 1 indicates valid points and 0 indicates no points.
            colormap (str): The colormap to use for valid height values (default: 'plasma').
            save_path (str): Path to save the visualization image.
        """
       # Ensure height_map and mask are numpy arrays
        height_map = height_map.cpu().numpy() if isinstance(height_map, torch.Tensor) else height_map
        mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

        # Create an RGBA image where invalid cells are black
        height_map_normalized = (height_map - np.min(height_map[mask > 0])) / (np.max(height_map[mask > 0]) - np.min(height_map[mask > 0]))
        height_map_normalized = np.clip(height_map_normalized, 0, 1)  # Normalize to [0, 1]
        colormap_func = plt.cm.get_cmap(colormap)
        height_map_colored = colormap_func(height_map_normalized)  # Apply colormap
        height_map_colored = (height_map_colored[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB

        # Set invalid cells (mask == 0) to black
        height_map_colored[mask == 0] = [0, 0, 0]

        # Display the visualization
        plt.figure(figsize=(10, 6))
        plt.imshow(height_map_colored)
        plt.axis()
        plt.title('Height Map Visualization')
        plt.tight_layout()
        cbar = plt.colorbar()
        cbar.set_label('cm')
        max = height_map[mask > 0].max()
        min = height_map[mask > 0].min()

        def to_cm(tick_val, pos=None):
            # tick_val is in [0, 255]; convert to cm proportionally to the original range
            return (tick_val / 255.0) * (max - min) + min
        

        from matplotlib.ticker import FuncFormatter

        cbar.formatter = FuncFormatter(lambda v, pos: f"{to_cm(v):.2f}")
        cbar.update_ticks()


        # Save the visualization
        plt.savefig(save_path, dpi=300)
        #plt.show()
        ##print(f"Height map visualization saved to {save_path}")


    def get_cam2vert(self, cam_up, normal_y_up):
        v = np.cross(cam_up, normal_y_up)
        print(f"vsddddd {v}")

        s = np.linalg.norm(v)  # sin(theta)
        c = np.dot(cam_up, normal_y_up)  # cos(theta)

        if s == 0:
            if c > 0:  # vectors are the same
                return np.eye(3)
            else:  # vectors are opposite
                # Pick the x axis for rotation
                perp_axis = np.array([1, 0, 0])
                vx = np.array([[0, -perp_axis[2], perp_axis[1]],
                            [perp_axis[2], 0, -perp_axis[0]],
                            [-perp_axis[1], perp_axis[0], 0]])
                R = np.eye(3) + vx + vx @ vx * ((1 - (-1)) / (1**2))  # 180° rotation
                return R

        vx = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
        return R


    def save_gt_as_image(self, pcd, img_path, intrinsic):
        
        # Assuming you already have your point cloud object `pcd`
        # Extract points as NumPy array
        points = np.asarray(pcd.points)
        img = cv2.imread(img_path)

    # Project points to image plane
        uvz = (intrinsic @ points.T).T  # shape: (N, 3)
        uv = uvz[:, :2] / uvz[:, 2].reshape(-1, 1)

        # Draw points on image
        for (u, v) in uv:
            u_int, v_int = int(u), int(v)
            if 0 <= u_int < img.shape[1] and 0 <= v_int < img.shape[0]:
                cv2.circle(img, (u_int, v_int), 2, (0, 255, 0), -1)  # green dots

        # Save the overlay image
        output_path = "Pcd_on_image/visualization pcd" + str(img_path.stem) + ".jpg"
        cv2.imwrite(output_path, img)
        print(f"Overlay image saved to {output_path}")
    
    def save_gt_points(self, pcd, str = " "):
        pts = np.asarray(pcd.points)
       # print('Erwan,', pts.shape)
        x = pts[:, 0]  # longitudinal
        y = pts[:, 1]  # height
        z = pts[:, 2]  # lateral

        # Plot f(x, z) = y using color encoding for y
        plt.figure(figsize=(8, 4))
        scatter = plt.scatter(x, z, c=y, cmap='plasma', s=2)
        plt.colorbar(scatter, label='Height (Y)')
        plt.xlabel('X (Lateral)')
        plt.ylabel('Z (Longitidunal)')
        plt.title('f(x, z) = y encoded with plasma colormap')
        plt.tight_layout()

        # Save the plot as a PNG file
        #plt.savefig('pointcloud_projection' + str + '.png', dpi=300)
        plt.show()

    def get_gt_elevation(self, xyz):
        #transform in world coordinate
        xyz = np.asarray(xyz.points)


        points_y = xyz[:, 1]*100  # points, m --> cm
        points_xz = xyz[:, [0, 2]]
        grids_y = torch.zeros((self.num_grids_z, self.num_grids_x), dtype=torch.float32)
        grids_count = torch.zeros((self.num_grids_z, self.num_grids_x), dtype=torch.int8)

        for xz, y in zip(points_xz, points_y):
            idx_x = torch.clip(((xz[0] - self.roi_x[0]) / self.grid_res[0]).int(), max=self.num_grids_x-1)
            idx_z = torch.clip(self.num_grids_z - 1 - ((xz[1] - self.roi_z[0]) / self.grid_res[2]).int(), min=0)
            grids_y[idx_z, idx_x] += y
            grids_count[idx_z, idx_x] += 1
        mask = grids_count > 0

        grids_y[mask] = self.camera_height*100 - grids_y[mask] / grids_count[mask]

        return grids_y, mask
    
    def get_transformation_matrices(self, extrinsic, intrinsic):

        """
        Compute the camera projection matrix given intrinsics and extrinsics.
        
        Parameters:
            intrinsic_matrix (np.ndarray): 3x3 camera intrinsic matrix
            extrinsic_matrix (np.ndarray): 4x4 camera extrinsic matrix (pose)
        
        Returns:
            np.ndarray: 3x4 projection matrix
        """
        # Validate shapes
        if intrinsic.shape != (3, 3):
            raise ValueError("Intrinsic matrix must be 3x3.")
        if extrinsic.shape != (4, 4):
            raise ValueError("Extrinsic matrix must be 4x4.")
        
        # Extract rotation and translation from extrinsic (first 3 rows)
        Rt = extrinsic[:3, :]  # 3x4
        
        # Compute projection matrix
        projection_matrix = intrinsic @ Rt  # 3x4
        
        return projection_matrix



    def get_cam_payload(self, img_file):

        path_of_the_sequence, _ = self._seq_root_from_rel_CARD_Reconstruction(img_file)
        traj = self._get_traj(path_of_the_sequence)
        _, relative_path = self._seq_root_from_rel_CARD_Reconstruction(img_file)
        img = self.get_img(Path(img_file))
        img = torch.from_numpy(img).permute(2,0,1).float()  # Normalize to [0, 1]
        ts = self._ts_from_rel_CARD_Reconstruction(str(relative_path))	
        cam_name, _ = self._parse_cam_ts(img_file)
        intrinsic, dist0 = self._K_dist_from_traj(traj, cam_name)
        extrinsic = self._T_cam_world(traj, cam_name, ts)
        norm, pt, wh = self._compute_road_plane_from_wheels(traj, ts)
        h_cam = self._compute_camera_height_from_ground_plane(traj, cam_name, ts, norm, pt)
        gp_info = {
            "plane_normal_world": torch.from_numpy(norm), "plane_point_world": torch.from_numpy(pt),
            "wheel_points_world": torch.from_numpy(wh), "camera_height_above_ground": h_cam, "timestamp_us": ts
        }
        return torch.Tensor(extrinsic), torch.Tensor(intrinsic), torch.Tensor(img), gp_info


    def _seq_root_from_rel_CARD_Reconstruction(self, rel: str) :
        # split at "/img/" and take the left part
        parts = rel.replace("\\","/").split("/img/")
        rel_path = Path(parts[1]) if len(parts) > 1 else None
        return Path(parts[0]), rel_path
    
    def _get_traj(self, seq_root: Path):
        key = str(seq_root)
        traj_path = seq_root / "export" / "output.laz.trajectory.json"
        print("Loading trajectory from:", traj_path)
        with open(traj_path, "r") as f:
            d = json.load(f)
        #self._traj_cache[key] = d

        return d
    
    @staticmethod
    def _parse_cam_ts(rel: str) -> tuple[Optional[str], Optional[int]]:
        name = Path(rel).name  # e.g., cam_1_101801867.jpg
        m = re.match(r"(cam_\d+)_(\d+)\.(?:jpg|jpeg|png)$", name)
        if not m: return None, None
        return m.group(1), int(m.group(2))
    
    @staticmethod
    def _K_dist_from_traj(traj: dict, cam: str):
        ci = traj["camera_infos"][cam]
        fx,fy,cx,cy = ci["fx"],ci["fy"],ci["cx"],ci["cy"]
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float32)
        dist = np.array(ci.get("dist_coeffs", [0,0,0,0,0]), np.float32)
        return K, dist
    

    def _ts_from_rel_CARD_Reconstruction(self, rel: str) -> Optional[int]:
        _, ts = self._parse_cam_ts(rel)
        return ts
    
    @staticmethod
    def get_img(path: Path) -> np.ndarray:
        im = Image.open(str(path)).convert("RGB")
        return np.array(im)


    
    def _T_cam_world(self, traj: dict, cam: str, ts_img_us: int):
        ci = traj["camera_infos"][cam]
        ts = ts_img_us + ci.get("timestamp_offset", 0)
        x,y,z, qw,qx,qy,qz = interpolate_pose(traj, ts)
        T_rig_world = pose_to_matrix(x,y,z, qw,qx,qy,qz)
        T_cam_rig = self._T_cam_to_rig_from_traj(traj, cam)
        return T_rig_world @ T_cam_rig

    @staticmethod
    def _T_cam_to_rig_from_traj(traj: dict, cam: str):
        x,y,z, qw,qx,qy,qz = traj["sensor_to_trajectory_poses"][cam]
        return pose_to_matrix(x,y,z, qw,qx,qy,qz)
    

    def get_normalized_intrinsics(K, image):
        """
        Fully normalizes camera intrinsics by dividing fx, fy, cx, cy
        by the image width and height respectively.

        Parameters:
        - K: np.ndarray of shape (3, 3), original camera intrinsics
        - image_size: tuple (width, height) of the image

        Returns:
        - K_normalized: np.ndarray of shape (3, 3), normalized intrinsics
        """
        width, height = image.size

        K_normalized = K.copy()
        K_normalized[0, 0] /= width   # fx
        K_normalized[1, 1] /= height  # fy
        K_normalized[0, 2] /= width   # cx
        K_normalized[1, 2] /= height  # cy

        return K_normalized
    
    def _compute_road_plane_from_wheels(self, traj: Dict, timestamp_us: int):
        # NOTE: Reduced to dynamic computation for simplicity and memory safety
        wheel_keys = ["cariad_wheel_FL_ground", "cariad_wheel_FR_ground", "cariad_wheel_RL_ground", "cariad_wheel_RR_ground"]
        points = []
        for wk in wheel_keys:
            if wk in traj.get("sensor_to_trajectory_poses", {}):
                w_pose = traj["sensor_to_trajectory_poses"][wk] # x,y,z...
                # Get Vehicle Pose at TS
                v_pose = interpolate_pose(traj, timestamp_us)
                T_veh_world = pose_to_matrix(*v_pose)
                w_veh = np.array([w_pose[0], w_pose[1], w_pose[2], 1.0])
                points.append((T_veh_world @ w_veh)[:3])
        
        if len(points) < 3:
            # Fallback
            vp = interpolate_pose(traj, timestamp_us)
            p = np.array([vp[0], vp[1], vp[2]-1.8], dtype=np.float32)
            return (np.array([0,0,1],dtype=np.float32), p, np.array([p]))

        pts = np.array(points, dtype=np.float32)
        ctr = np.mean(pts, axis=0)
        _, _, vh = np.linalg.svd(pts - ctr)
        print("yessss", vh.shape)
        norm = vh[-1]
        if norm[2] < 0: norm = -norm
        return (norm, ctr, pts)

    def _compute_camera_height_from_ground_plane(self, traj, cam, ts, norm, pt):
        Tcw = self._T_cam_world(traj, cam, ts)
        cam_pt = Tcw[:3, 3]
        return float(np.dot(cam_pt - pt, norm / np.linalg.norm(norm)))

class CARDSetDatasetV2Smalldataset(Dataset):
    def __init__(self, 
                 root_dir: str,
                 mode: str = 'train',
                 down_scale = 4):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        

        self.img_dir = os.path.join(root_dir, 'img', 'cam_1')
        self.depth_dir = os.path.join(root_dir, 'agg_depth', 'cam_1')
        self.image_files = sorted(os.listdir(self.img_dir), key=lambda x: (len(x), x))
        self.depth_files = sorted(os.listdir(self.depth_dir), key=lambda x: (len(x), x))

        self.down_scale = down_scale

        self.base_height = 1.857  # in meter, the reference height of the camera w.r.t. road surface
        self.y_range = 0.5  # in meter, the range of interest above and below the base height， i.e., [-20cm, 20cm]
        self.roi_x = torch.tensor([-1, 0.92])    # in meter, the lateral range of interest (in the horizontal coordinate of camera)
        self.roi_z = torch.tensor([5.16, 10.08])    # in meter, the longitudinal range of interest

        self.grid_res = torch.tensor([0.03, 0.01, 0.03])  # in [x, y(vertical), z] order. The range of interest above should be integer times of resolution here
        

        self.num_grids_x = int((self.roi_x[1] - self.roi_x[0]) / self.grid_res[0])
        self.num_grids_z = int((self.roi_z[1] - self.roi_z[0]) / self.grid_res[2])
        self.num_grids_y = int(self.y_range*2 / self.grid_res[1])

        len_images = len(self.image_files)
        len_depths = len(self.depth_files)
        assert len_images == len_depths, f"The number of images and depth files should be the same.{len_images} != {len_depths}"
        if(mode not in ['train', 'test']):
            raise ValueError("mode should be either 'train' or 'test'")
        
        elif mode == 'train':
            
            self.image_files = self.image_files[:len_images*7//10]
            self.depth_files = self.depth_files[:len_images*7//10]
        else:
            self.image_files = self.image_files #[len_images*7//10:]
            self.depth_files = self.depth_files #[len_images*7//10:]

        # generate the centers of every horizontal grid
        hori_centers = torch.zeros((self.num_grids_z, self.num_grids_x, 2), dtype=torch.float32)
        hori_centers[:, :, 0] = (torch.arange(self.num_grids_x) * self.grid_res[0] + self.roi_x[0] + self.grid_res[0]/2).unsqueeze(0).repeat([self.num_grids_z, 1])
        hori_centers[:, :, 1] = (-torch.arange(self.num_grids_z) * self.grid_res[2] + self.roi_z[1] - self.grid_res[2]/2).unsqueeze(1).repeat([1, self.num_grids_x])
        self.map_centers = hori_centers.reshape(-1, 2)
        self.num_center = self.map_centers.shape[0]

        # generate the centers of every 3D voxel
        voxel_centers = torch.zeros((self.num_grids_z, self.num_grids_x, self.num_grids_y, 3), dtype=torch.float32)
        voxel_centers[:, :, :, [0, 2]] = hori_centers.unsqueeze(2).repeat([1, 1, self.num_grids_y, 1])
        voxel_centers[:, :, :, 1] = (torch.arange(self.num_grids_y) * self.grid_res[1] + self.base_height - self.y_range + self.grid_res[1]/2).unsqueeze(0).unsqueeze(0).repeat([self.num_grids_z, self.num_grids_x, 1])
        self.voxel_centers = voxel_centers.reshape(-1, 3).transpose(1, 0)
        print(fr"voxel_centers shape{voxel_centers.shape}")
        
        self.transform_jpg = transforms.Compose([
            transforms.ToTensor(),  # image --> [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0, 1] --> [-1, 1]
        ])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image from cam_1
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        print(img_path)
        imgs_left = Image.open(img_path)


        #get the camera parameter of the camera
        extrinsic, intrinsic, _, ground_info = self.get_cam_payload(str(img_path))
        extrinsic_inv = np.linalg.inv(extrinsic)
        
        imgs_left, intrinsic = self.crop_image(intrinsic, imgs_left)

        if idx == 0:
            imgs_left.save("mon_image.jpg")
        imgs_left = self.transform_jpg(imgs_left)
                
        print("mbouuuu", imgs_left.shape)


        self.ground_normal_vector = ground_info['plane_normal_world']
        self.camera_height = ground_info['camera_height_above_ground']

        print(self.camera_height)
        # Load point cloud from npz and crop it
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        print(depth_path)
        data = np.load(depth_path)
        points = data['pts_cam'] # N*3 array
        print(f"Height before values (y): min={points[:, 1].min()}, max={points[:, 1].max()}")
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        R = coord_frame.get_rotation_matrix_from_xyz((0.0, 0.0, np.pi))  # (Rx, Ry, Rz) in radians
        coord_frame = coord_frame.rotate(R, center=(0, 0, 0))


        #Transform the pts in plane coordinate system: Hg = Hc*N + delta(camera height - ground height)
        #if idx == 0:
        #draw_voxel_bounding_boxes(img_path, self.voxel_centers, intrinsic, extrinsic, down_scale=self.down_scale)


        ####
        [pitch_cam, roll_cam, yaw_cam] = self.matrix2euler((extrinsic[:3, :3].numpy()).T)  #transpose to get the inverse of the rotation
        print(pitch_cam, roll_cam, yaw_cam)
        pitch_cam -= 1.5708  # pi/2
        R_X = np.array(
            [[1, 0, 0], [0, np.cos(pitch_cam), np.sin(pitch_cam)], [0, -np.sin(pitch_cam), np.cos(pitch_cam)]], dtype=np.float32)
        R_Z = np.array(
            [[np.cos(roll_cam), np.sin(roll_cam), 0], [-np.sin(roll_cam), np.cos(roll_cam), 0], [0, 0, 1]], dtype=np.float32)
        R_cam2vert = R_X @ R_Z  # the rotation matrix from the current camera coord to the vertical status
        R_vert2cam = torch.from_numpy(np.linalg.inv(R_cam2vert))
        ####
        
        R_cam2_vert = R_cam2vert
        #points[:, 1] -= self.camera_height
        R_vert2_cam = np.linalg.inv(R_cam2vert)

        camera_frame = copy.deepcopy(coord_frame)

        camera_frame.rotate(R_cam2vert, center=(0, 0, 0))

        # Ensure voxel_centers is [N, 3]
        voxel_centers = self.voxel_centers  # if currently [3, N]


        voxel_cam_left = torch.tensor(R_vert2_cam) @ voxel_centers

        # Downscale intrinsics
        intrinsic_downscaled = (intrinsic / self.down_scale).to(torch.float32)
        intrinsic_downscaled[2, 2] = 1
        uvz_left =  intrinsic_downscaled @ voxel_cam_left
        voxel_uv_left = torch.floor(uvz_left[:2, :] / uvz_left[2:, :]).type(torch.long)

        print(f"voxel_uv_left{voxel_uv_left.shape}")
        print(voxel_uv_left[:, -1])
        print(voxel_uv_left[:, 16000])

        pcd_cam2vert = o3d.geometry.PointCloud()
        pcd_cam2vert.points = o3d.utility.Vector3dVector(points)
        pcd_cam2vert.rotate(R_cam2_vert, center=(0, 0, 0))
        print("fjklnnini", ((np.asarray(pcd_cam2vert.points))[:, 1]).max())
        crop_bounding = np.array([[self.roi_x[0], 0, self.roi_z[0]],
                                   [self.roi_x[0], 0, self.roi_z[1]],
                                   [self.roi_x[1], 0, self.roi_z[1]],
                                   [self.roi_x[1], 0, self.roi_z[0]]]).astype("float64")
        points = np.asarray(pcd_cam2vert.points)
        print(f"Height after transformation values (y): min={points[:, 1].min()}, max={points[:, 1].max()}")
        vol = o3d.visualization.SelectionPolygonVolume()
        vol.orthogonal_axis = "Y"
        vol.axis_max = 5
        vol.axis_min = 0
        vol.bounding_polygon = o3d.utility.Vector3dVector(crop_bounding)
        self.save_gt_points(pcd_cam2vert, "before_crop" + str(self.image_files[idx]))
        pcd_cam2vert = vol.crop_point_cloud(pcd_cam2vert)
        self.save_gt_as_image(pcd_cam2vert, img_path, intrinsic)
        points = np.asarray(pcd_cam2vert.points)
        self.save_gt_points(pcd_cam2vert, str(self.image_files[idx]))
        o3d.visualization.draw_geometries([camera_frame, coord_frame, pcd_cam2vert])
        print(f"Height after crop values (y): min={points[:, 1].min()}, max={points[:, 1].max()}")

        app = o3d.visualization.gui.Application.instance
        app.initialize()
        #viz = o3d.visualization.O3DVisualizer("Scene", 1280, 720)

        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        enu = copy.deepcopy(camera)
        enu.rotate(extrinsic[:3, :3], center=(0,0,0))
        
        road = copy.deepcopy(camera)
        road.rotate(R_cam2_vert, center=(0,0,0))
        #viz.add_geometry("camera", camera)
        #viz.add_geometry("enu", enu)
        #viz.add_geometry("road", road)
        #viz.show_settings = True
        #app.add_window(viz)
        #app.run()
        # Project to height map
        height_map, mask = self.get_gt_elevation(pcd_cam2vert)
        print(f"croppoint count{np.asarray(pcd_cam2vert.points)[:, 1].max()}")
        self.visualize_height_map_and_mask(height_map, mask, colormap='viridis', save_path='Heightmap/height_map_visualization.png' + str(self.image_files[idx]))
        print(height_map.shape, height_map[:, 1])
        
        # Convert to tensors
        ele_gt = height_map # (H, W)
        if (idx == 0):
            print(f"elevation size{ele_gt.shape}")
        mask = mask.type(torch.float32)
        if idx == 0:
            print(f"elevation mask size{mask.shape}")

        
            print("imgs_left:", type(imgs_left), getattr(imgs_left, 'shape', 'No shape'))
            print("ele_gt:", type(ele_gt), getattr(ele_gt, 'shape', 'No shape'))
            print("mask:", type(mask), getattr(mask, 'shape', 'No shape'))
            print("voxel_uv_left:", type(voxel_uv_left), getattr(voxel_uv_left, 'shape', 'No shape'))
            print("timestamp_us:", ground_info.get('timestamp_us', 'Not found'))
            print(f"voxel_uv_left shape: {voxel_uv_left.shape}")
            print(f"voxel_uv_left min: {voxel_uv_left.min(dim=0).values}, max: {voxel_uv_left.max(dim=0).values}")


            
        return imgs_left, ele_gt, mask, voxel_uv_left, ground_info['timestamp_us']

    def matrix2euler(self, m):
        # order='XYZ'
        d = np.clip
        m = m.reshape(-1)
        a, f, g, k, l, n, e = m[0], m[1], m[2], m[4], m[5], m[7], m[8]
        y = np.arcsin(d(g, -1, 1))
        if 0.99999 > np.abs(g):
            x = np.arctan2(- l, e)
            z = np.arctan2(- f, a)
        else:
            x = np.arctan2(n, k)
            z = 0
        return np.array([x, y, z], dtype=np.float32)

    def crop_image (self, K, image):
        W, H = image.size
        w_c = 960
        h_c = 528
        u_0 = (W - w_c) / 2
        v_0 = (H - h_c) / 2
        u_1 = (W + w_c) / 2
        v_1 = (H + h_c) / 2
        #imgs_left = image.crop((u_0, v_0, u_1, v_1))
        imgs_left = image.resize((w_c, h_c))
        scale_x = W/w_c
        scale_y = H/h_c
        intrinsic = K
        intrinsic[0:2] = intrinsic[0:2] / scale_x  #cx
        intrinsic[1:2] = intrinsic[1:2] / scale_y  #cy
        intrinsic[0:0] = intrinsic[0:0] / scale_x  #fx
        intrinsic[1:1] = intrinsic[1:1] / scale_y  #fy
       

        return imgs_left, intrinsic 
    
    @staticmethod
    def visualize_height_map_and_mask(height_map, mask, colormap='plasma', save_path='Heightmap/height_map_visualization.png'):
        """
        Visualize the height map and mask. Cells without points are black, and cells with height GT are mapped with a colormap.

        Parameters:
            height_map (torch.Tensor): The height map tensor of shape (H, W).
            mask (torch.Tensor): The mask tensor of shape (H, W), where 1 indicates valid points and 0 indicates no points.
            colormap (str): The colormap to use for valid height values (default: 'plasma').
            save_path (str): Path to save the visualization image.
        """
        # Ensure height_map and mask are numpy arrays
        height_map = height_map.cpu().numpy() if isinstance(height_map, torch.Tensor) else height_map
        mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

        # Create an RGBA image where invalid cells are black
        height_map_normalized = (height_map - np.min(height_map[mask > 0])) / (np.max(height_map[mask > 0]) - np.min(height_map[mask > 0]))
        height_map_normalized = np.clip(height_map_normalized, 0, 1)  # Normalize to [0, 1]
        colormap_func = plt.cm.get_cmap(colormap)
        height_map_colored = colormap_func(height_map_normalized)  # Apply colormap
        height_map_colored = (height_map_colored[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB

        # Set invalid cells (mask == 0) to black
        height_map_colored[mask == 0] = [0, 0, 0]

        # Display the visualization
        plt.figure(figsize=(10, 6))
        plt.imshow(height_map_colored)
        plt.axis()
        plt.title('Height Map Visualization')
        plt.tight_layout()
        cbar = plt.colorbar()
        cbar.set_label('cm')
        max = height_map[mask > 0].max()
        min = height_map[mask > 0].min()

        def to_cm(tick_val, pos=None):
            # tick_val is in [0, 255]; convert to cm proportionally to the original range
            return (tick_val / 255.0) * (max - min) + min
        

        from matplotlib.ticker import FuncFormatter

        cbar.formatter = FuncFormatter(lambda v, pos: f"{to_cm(v):.2f}")
        cbar.update_ticks()


        # Save the visualization
        plt.savefig(save_path, dpi=300)
        #plt.show()
        ##print(f"Height map visualization saved to {save_path}")


    def get_cam2vert(self, cam_up, normal_y_up):
        v = np.cross(cam_up, normal_y_up)
        print(f"vsddddd {v}")

        s = np.linalg.norm(v)  # sin(theta)
        c = np.dot(cam_up, normal_y_up)  # cos(theta)

        if s == 0:
            if c > 0:  # vectors are the same
                return np.eye(3)
            else:  # vectors are opposite
                # Pick the x axis for rotation
                perp_axis = np.array([1, 0, 0])
                vx = np.array([[0, -perp_axis[2], perp_axis[1]],
                            [perp_axis[2], 0, -perp_axis[0]],
                            [-perp_axis[1], perp_axis[0], 0]])
                R = np.eye(3) + vx + vx @ vx * ((1 - (-1)) / (1**2))  # 180° rotation
                return R

        vx = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
        return R


    def save_gt_as_image(self, pcd, img_path, intrinsic):
        
        # Assuming you already have your point cloud object `pcd`
        # Extract points as NumPy array
        points = np.asarray(pcd.points)
        img = cv2.imread(img_path)

    # Project points to image plane
        uvz = (intrinsic @ points.T).T  # shape: (N, 3)
        uv = uvz[:, :2] / uvz[:, 2].reshape(-1, 1)

        # Draw points on image
        for (u, v) in uv:
            u_int, v_int = int(u), int(v)
            if 0 <= u_int < img.shape[1] and 0 <= v_int < img.shape[0]:
                cv2.circle(img, (u_int, v_int), 2, (0, 255, 0), -1)  # green dots

        # Save the overlay image
        output_path = "visualization pcd.jpg"
        cv2.imwrite(output_path, img)
        print(f"Overlay image saved to {output_path}")
    
    @staticmethod
    def save_gt_points(pcd, str = " "):

        if isinstance(pcd, o3d.geometry.PointCloud):
            pts = np.asarray(pcd.points)
        else:
            pts = np.asarray(pcd)
        

       # print('Erwan,', pts.shape)
        x = pts[:, 0]  # longitudinal
        y = pts[:, 1]  # height
        z = pts[:, 2]  # lateral

        # Plot f(x, z) = y using color encoding for y
        plt.figure(figsize=(8, 4))
        scatter = plt.scatter(x, z, c=y, cmap='plasma', s=2)
        plt.colorbar(scatter, label='Height (Y)')
        plt.xlabel('X (Lateral)')
        plt.ylabel('Z (Longitidunal)')
        plt.title('f(x, z) = y encoded with plasma colormap')
        plt.tight_layout()

        # Save the plot as a PNG file
        plt.savefig('pointcloud_projection' + str + '.png', dpi=300)
        plt.show()

    def get_gt_elevation(self, xyz):
        #transform in world coordinate
        xyz = np.asarray(xyz.points)


        points_y = xyz[:, 1]*100  # points, m --> cm
        points_xz = xyz[:, [0, 2]]
        grids_y = torch.zeros((self.num_grids_z, self.num_grids_x), dtype=torch.float32)
        grids_count = torch.zeros((self.num_grids_z, self.num_grids_x), dtype=torch.int8)

        for xz, y in zip(points_xz, points_y):
            idx_x = torch.clip(((xz[0] - self.roi_x[0]) / self.grid_res[0]).int(), max=self.num_grids_x-1)
            idx_z = torch.clip(self.num_grids_z - 1 - ((xz[1] - self.roi_z[0]) / self.grid_res[2]).int(), min=0)
            grids_y[idx_z, idx_x] += y
            grids_count[idx_z, idx_x] += 1
        mask = grids_count > 0

        grids_y[mask] = self.camera_height*100 - grids_y[mask] / grids_count[mask]

        return grids_y, mask
    
    def get_transformation_matrices(self, extrinsic, intrinsic):

        """
        Compute the camera projection matrix given intrinsics and extrinsics.
        
        Parameters:
            intrinsic_matrix (np.ndarray): 3x3 camera intrinsic matrix
            extrinsic_matrix (np.ndarray): 4x4 camera extrinsic matrix (pose)
        
        Returns:
            np.ndarray: 3x4 projection matrix
        """
        # Validate shapes
        if intrinsic.shape != (3, 3):
            raise ValueError("Intrinsic matrix must be 3x3.")
        if extrinsic.shape != (4, 4):
            raise ValueError("Extrinsic matrix must be 4x4.")
        
        # Extract rotation and translation from extrinsic (first 3 rows)
        Rt = extrinsic[:3, :]  # 3x4
        
        # Compute projection matrix
        projection_matrix = intrinsic @ Rt  # 3x4
        
        return projection_matrix



    def get_cam_payload(self, img_file):

        path_of_the_sequence, _ = self._seq_root_from_rel_CARD_Reconstruction(img_file)
        traj = self._get_traj(path_of_the_sequence)
        _, relative_path = self._seq_root_from_rel_CARD_Reconstruction(img_file)
        img = self.get_img(Path(img_file))
        img = torch.from_numpy(img).permute(2,0,1).float()  # Normalize to [0, 1]
        ts = self._ts_from_rel_CARD_Reconstruction(str(relative_path))	
        cam_name, _ = self._parse_cam_ts(img_file)
        intrinsic, dist0 = self._K_dist_from_traj(traj, cam_name)
        extrinsic = self._T_cam_world(traj, cam_name, ts)
        norm, pt, wh = self._compute_road_plane_from_wheels(traj, ts)
        h_cam = self._compute_camera_height_from_ground_plane(traj, cam_name, ts, norm, pt)
        gp_info = {
            "plane_normal_world": torch.from_numpy(norm), "plane_point_world": torch.from_numpy(pt),
            "wheel_points_world": torch.from_numpy(wh), "camera_height_above_ground": h_cam, "timestamp_us": ts
        }
        return torch.Tensor(extrinsic), torch.Tensor(intrinsic), torch.Tensor(img), gp_info


    def _seq_root_from_rel_CARD_Reconstruction(self, rel: str) :
        # split at "/img/" and take the left part
        parts = rel.replace("\\","/").split("/img/")
        rel_path = Path(parts[1]) if len(parts) > 1 else None
        return Path(parts[0]), rel_path
    
    def _get_traj(self, seq_root: Path):
        key = str(seq_root)
        traj_path = seq_root.parent / "export" / "output.laz.trajectory.json"
        with open(traj_path, "r") as f:
            d = json.load(f)
        #self._traj_cache[key] = d

        return d
    
    @staticmethod
    def _parse_cam_ts(rel: str) -> tuple[Optional[str], Optional[int]]:
        name = Path(rel).name  # e.g., cam_1_101801867.jpg
        m = re.match(r"(cam_\d+)_(\d+)\.(?:jpg|jpeg|png)$", name)
        if not m: return None, None
        return m.group(1), int(m.group(2))
    
    @staticmethod
    def _K_dist_from_traj(traj: dict, cam: str):
        ci = traj["camera_infos"][cam]
        fx,fy,cx,cy = ci["fx"],ci["fy"],ci["cx"],ci["cy"]
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float32)
        dist = np.array(ci.get("dist_coeffs", [0,0,0,0,0]), np.float32)
        return K, dist
    

    def _ts_from_rel_CARD_Reconstruction(self, rel: str) -> Optional[int]:
        _, ts = self._parse_cam_ts(rel)
        return ts
    
    @staticmethod
    def get_img(path: Path) -> np.ndarray:
        im = Image.open(str(path)).convert("RGB")
        return np.array(im)


    
    def _T_cam_world(self, traj: dict, cam: str, ts_img_us: int):
        ci = traj["camera_infos"][cam]
        ts = ts_img_us + ci.get("timestamp_offset", 0)
        x,y,z, qw,qx,qy,qz = interpolate_pose(traj, ts)
        T_rig_world = pose_to_matrix(x,y,z, qw,qx,qy,qz)
        T_cam_rig = self._T_cam_to_rig_from_traj(traj, cam)
        return T_rig_world @ T_cam_rig

    @staticmethod
    def _T_cam_to_rig_from_traj(traj: dict, cam: str):
        x,y,z, qw,qx,qy,qz = traj["sensor_to_trajectory_poses"][cam]
        return pose_to_matrix(x,y,z, qw,qx,qy,qz)
    

    def get_normalized_intrinsics(K, image):
        """
        Fully normalizes camera intrinsics by dividing fx, fy, cx, cy
        by the image width and height respectively.

        Parameters:
        - K: np.ndarray of shape (3, 3), original camera intrinsics
        - image_size: tuple (width, height) of the image

        Returns:
        - K_normalized: np.ndarray of shape (3, 3), normalized intrinsics
        """
        width, height = image.size

        K_normalized = K.copy()
        K_normalized[0, 0] /= width   # fx
        K_normalized[1, 1] /= height  # fy
        K_normalized[0, 2] /= width   # cx
        K_normalized[1, 2] /= height  # cy

        return K_normalized
    
    def _compute_road_plane_from_wheels(self, traj: Dict, timestamp_us: int):
        # NOTE: Reduced to dynamic computation for simplicity and memory safety
        wheel_keys = ["cariad_wheel_FL_ground", "cariad_wheel_FR_ground", "cariad_wheel_RL_ground", "cariad_wheel_RR_ground"]
        points = []
        for wk in wheel_keys:
            if wk in traj.get("sensor_to_trajectory_poses", {}):
                w_pose = traj["sensor_to_trajectory_poses"][wk] # x,y,z...
                # Get Vehicle Pose at TS
                v_pose = interpolate_pose(traj, timestamp_us)
                T_veh_world = pose_to_matrix(*v_pose)
                w_veh = np.array([w_pose[0], w_pose[1], w_pose[2], 1.0])
                points.append((T_veh_world @ w_veh)[:3])
        
        if len(points) < 3:
            # Fallback
            vp = interpolate_pose(traj, timestamp_us)
            p = np.array([vp[0], vp[1], vp[2]-1.8], dtype=np.float32)
            return (np.array([0,0,1],dtype=np.float32), p, np.array([p]))

        pts = np.array(points, dtype=np.float32)
        ctr = np.mean(pts, axis=0)
        _, _, vh = np.linalg.svd(pts - ctr)
        print("yessss", vh.shape)
        norm = vh[-1]
        if norm[2] < 0: norm = -norm
        return (norm, ctr, pts)

    def _compute_camera_height_from_ground_plane(self, traj, cam, ts, norm, pt):
        Tcw = self._T_cam_world(traj, cam, ts)
        cam_pt = Tcw[:3, 3]
        return float(np.dot(cam_pt - pt, norm / np.linalg.norm(norm)))
    

def project_to_depth_image_numpy(
    pcd_np: np.ndarray,
    intrinsics_np: np.ndarray,
    width: int,
    height: int,
    extrinsics_np: np.ndarray = np.eye(4, dtype=np.float32),
    depth_scale: float = 1000.0,
    depth_max: float = 100
) -> torch.Tensor:
    # Convert NumPy point cloud to Open3D tensor
    pcd_o3d = o3d.t.geometry.PointCloud()
    pcd_o3d.point["positions"] = o3d.core.Tensor(pcd_np.astype(np.float32), dtype=o3d.core.Dtype.Float32)

    # Convert intrinsics and extrinsics to Open3D tensors
    intrinsics_o3d = o3d.core.Tensor(intrinsics_np.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    extrinsics_o3d = o3d.core.Tensor(extrinsics_np.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    # Project to depth image
    depth_image = pcd_o3d.project_to_depth_image(
        width=width,
        height=height,
        intrinsics=intrinsics_o3d,
        extrinsics=extrinsics_o3d,
        depth_scale=depth_scale,
        depth_max=depth_max
    )

    # Convert Open3D image to PyTorch tensor
    depth_np = depth_image.as_tensor().numpy()
    depth_tensor = torch.from_numpy(depth_np)

    return depth_tensor
def draw_voxel_bounding_boxes(img_path, voxel_centers, intrinsic, extrinsic, down_scale=1):
    """
    Draw bounding boxes of voxels on the image.

    Parameters:
        img_path (str): Path to the image file.
        voxel_centers (torch.Tensor): 3D voxel centers of shape [3, N].
        intrinsic (torch.Tensor): Camera intrinsic matrix of shape [3, 3].
        extrinsic (torch.Tensor): Camera extrinsic matrix of shape [4, 4].
        down_scale (float): Downscaling factor for the intrinsic matrix.
    """
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found at {img_path}")
    height, width, _ = img.shape

    # Downscale the intrinsic matrix
    intrinsic_downscaled = intrinsic / down_scale
    intrinsic_downscaled[2, 2] = 1

    # Transform voxel centers to camera coordinates
    voxel_cam = extrinsic[:3, :3] @ voxel_centers + extrinsic[:3, 3].unsqueeze(1)  # [3, N]

    # Project voxel centers to the image plane
    uvz = intrinsic_downscaled @ voxel_cam  # [3, N]
    uv = uvz[:2, :] / uvz[2, :]  # Normalize by depth (z)

    # Convert to integer pixel coordinates
    uv = uv.T.cpu().numpy().astype(np.int32)  # Shape: [N, 2]

    # Draw bounding boxes on the image
    for (u, v) in uv:
        if 0 <= u < width and 0 <= v < height:  # Check if the point is within the image bounds
            cv2.rectangle(img, (u - 5, v - 5), (u + 5, v + 5), (0, 255, 0), 2)  # Green box

    # Save or display the image
    output_path = "voxel_bounding_boxes.jpg"
    cv2.imwrite(output_path, img)
    print(f"Image with voxel bounding boxes saved to {output_path}")


def draw_vectors_planes_axes(vectors=None, plane=None, axes_length=1.0, plane_size=1.0, labels=None):
    """
    Zeichnet Vektoren, eine Ebene und Achsen in einer Open3D-Szene, mit optionalen Beschriftungen.

    Parameter:
        vectors (list of tuples): Liste von Vektoren, z. B. [(start1, end1), (start2, end2), ...].
                                  Jeder Vektor wird durch Start- und Endpunkt definiert.
        plane (tuple): Eine Ebene definiert durch (normal, point), wobei:
                       - normal: Normalenvektor der Ebene (3,)
                       - point: Ein Punkt auf der Ebene (3,)
        axes_length (float): Länge der Achsen (standardmäßig 1.0).
        plane_size (float): Größe der Ebene (standardmäßig 1.0).
        labels (list of str): Liste von Namen/Beschriftungen für die Vektoren.
    """
    geometries = []

    # Zeichne Vektoren
    if vectors:
        for i, (start, end) in enumerate(vectors):
            line = o3d.geometry.LineSet()
            points = [start, end]
            lines = [[0, 1]]
            colors = [[1, 0, 0]]  # Rot für Vektoren
            line.points = o3d.utility.Vector3dVector(points)
            line.lines = o3d.utility.Vector2iVector(lines)
            line.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(line)

            # Füge Beschriftungen hinzu
            if labels and i < len(labels):
                label_pos = (np.array(start) + np.array(end)) / 2  # Position der Beschriftung (Mitte des Vektors)
                text = create_text_3d(labels[i], label_pos, font_size=0.2)
                geometries.append(text)

    # Zeichne Ebene
    if plane:
        normal, point = plane
        normal = np.array(normal)
        point = np.array(point)

        # Erstelle Eckpunkte der Ebene
        d = plane_size / 2
        basis1 = np.cross(normal, [1, 0, 0])
        if np.linalg.norm(basis1) < 1e-6:
            basis1 = np.cross(normal, [0, 1, 0])
        basis1 = basis1 / np.linalg.norm(basis1)
        basis2 = np.cross(normal, basis1)
        basis2 = basis2 / np.linalg.norm(basis2)

        corners = [
            point + d * basis1 + d * basis2,
            point + d * basis1 - d * basis2,
            point - d * basis1 - d * basis2,
            point - d * basis1 + d * basis2,
        ]

        # Erstelle Dreiecke für die Ebene
        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(corners)
        plane_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [2, 3, 0]])
        plane_mesh.paint_uniform_color([0, 1, 0])  # Grün für die Ebene
        geometries.append(plane_mesh)

    # Zeichne Achsen
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_length)
    geometries.append(axes)

    # Visualisiere die Szene
    o3d.visualization.draw_geometries(geometries)


def create_text_3d(text, position, font_size=1.0, color=(0, 0, 0)):
    """
    Erstellt ein 3D-Textobjekt für Open3D.

    Parameter:
        text (str): Der Text, der angezeigt werden soll.
        position (tuple): Die Position des Textes im 3D-Raum.
        font_size (float): Die Schriftgröße.
        color (tuple): Die Farbe des Textes (RGB).

    Rückgabe:
        o3d.geometry.TriangleMesh: Das 3D-Textobjekt.
    """
    text_mesh = o3d.geometry.TriangleMesh.create_text(text, font_size, depth=0.1)
    text_mesh.paint_uniform_color(color)
    text_mesh.translate(position)
    return text_mesh

def align_point_cloud(points_cam, ground_normal, ground_points):
    # --- 1. Ensure ground normal points upward ---
    camera_up = np.array([0, -1, 0])
    if np.dot(ground_normal, camera_up) < 0:
        ground_normal = -ground_normal
    n = ground_normal / np.linalg.norm(ground_normal)

    # --- 2. Build rotation matrix ---
    camera_forward = np.array([0,0,1])
    X = np.cross(camera_forward, n)
    X /= np.linalg.norm(X)
    Y = np.cross(n, X)
    R = np.vstack((X, Y, n))   # 3×3

    # --- 3. Compute translation so ground is Z=0 ---
    ground_heights = []
    for p in ground_points:
        p_rot = R @ p
        ground_heights.append(p_rot[2])

    ground_height = np.mean(ground_heights)

    # --- 4. Transform whole cloud ---
    points_world = (R @ points_cam.T).T
    points_world[:,2] -= ground_height

    return points_world, R, ground_height

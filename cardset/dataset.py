'''The goal of this module is to provide a dataset class for the CARDSet dataset on the RoadBEV model.'''
import os
import open3d as o3d
from typing import Any, Callable, Optional, Dict, Tuple, List, Union
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
import argparse
from torch.utils.data import random_split
from torch.utils.data import Subset
from cardset.utils import _apply_flip, apply_gaussian_noise_and_blur, apply_gt_cutout, npz_to_ply                                                   
import pickle, gzip, random


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
    poses = traj["trajectory_poses"]                                    
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
                 root_dir: Optional[str],
                 split_file: str,
                 mode: str = 'train',
                 down_scale = 4,
                 preprocessed_data = False,
                 augmentation = False,
                 use_static_rotation = True,
                 clamp_gt = False,
                 crop_to_road = False,
                 y_range = None,
                 num_grids_y = None):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.split_file = split_file
        self.clamp_gt = clamp_gt
        self.crop_to_road = crop_to_road
                                                                    
                                                                                
        self.fixed_crop_bbox = (613, 1199, 1607, 1703)
        self.crop_patch_align = 14                                              
                                                                             
        scenes = os.listdir(root_dir)
        scenes_sb = []
        """ for scene in scenes:
            if 'bump' in str(scene):
                scenes_sb.append(os.path.join(root_dir, scene))
        
        self.scenes_sb = scenes_sb[:3] """                                                

        self.down_scale = down_scale
        self.augmentation = augmentation
        self.use_static_rotation = use_static_rotation
        self._global_static_R = None                                                           
        self._traj_cache = {}                                            
        self._folder_payload_cache = {}                                                              
        self.base_height = 1.857                                                                    
        self.y_range = float(y_range) if y_range is not None else 0.2                                                
        self.roi_x = torch.tensor([-1.5, 1.5])                                       
        self.roi_z = torch.tensor([5.01, 15.0])                                                          

        self.grid_res = torch.tensor([0.03, 0.01, 0.03])                                

        self.num_grids_x = int((self.roi_x[1] - self.roi_x[0]) / self.grid_res[0])
        self.num_grids_z = int((self.roi_z[1] - self.roi_z[0]) / self.grid_res[2])
        if num_grids_y is not None:
                                                                                                        
            self.num_grids_y = int(num_grids_y)
            self.grid_res[1] = (self.y_range * 2) / self.num_grids_y
        else:
            self.num_grids_y = int(self.y_range*2 / self.grid_res[1])

                                                       
        hori_centers = torch.zeros((self.num_grids_z, self.num_grids_x, 2), dtype=torch.float32)
        hori_centers[:, :, 0] = (torch.arange(self.num_grids_x) * self.grid_res[0] + self.roi_x[0] + self.grid_res[0]/2).unsqueeze(0).repeat([self.num_grids_z, 1])
        hori_centers[:, :, 1] = (-torch.arange(self.num_grids_z) * self.grid_res[2] + self.roi_z[1] - self.grid_res[2]/2).unsqueeze(1).repeat([1, self.num_grids_x])
        self.hori_centers = hori_centers
        self.map_centers = hori_centers.reshape(-1, 2)
        self.num_center = self.map_centers.shape[0]

                                                
        voxel_centers = torch.zeros((self.num_grids_z, self.num_grids_x, self.num_grids_y, 3), dtype=torch.float32)
        voxel_centers[:, :, :, [0, 2]] = hori_centers.unsqueeze(2).repeat([1, 1, self.num_grids_y, 1])
        voxel_centers[:, :, :, 1] = (torch.arange(self.num_grids_y) * self.grid_res[1] + self.base_height - self.y_range + self.grid_res[1]/2).unsqueeze(0).unsqueeze(0).repeat([self.num_grids_z, self.num_grids_x, 1])
        self.voxel_centers = voxel_centers.reshape(-1, 3).transpose(1, 0)


        with open(self.split_file, "r") as f:
            self.images_names = [line.strip() for line in f if line.strip()]
            
        self.pairs = self.create_image_depth_pairs(self.images_names)            
        
                          
        if mode == 'train': 
            self.pairs = self.pairs                
        else:
            self.pairs = self.pairs                                            

        self.preprocessed_data = preprocessed_data

        
        self.transform_jpg = transforms.Compose([
            transforms.ToTensor(),                    
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                      
        ])

                                                                                   
        if self.use_static_rotation:
            ref_rel = "germany_2days/germany_5_sw/img/cam_1/cam_1_147401867.jpg"
            ref_path = str(Path(self.root_dir) / ref_rel)
            try:
                ext, intr, _, ginfo, _, _ = self.get_cam_payload(ref_path)
                n_w = ginfo['plane_normal_world'].numpy().astype(np.float32)
                R_c2w = ext[:3, :3].numpy()
                self._global_static_R = self._compute_R_vert2cam(n_w, R_c2w)
                pitch = np.degrees(np.arcsin(np.clip(self._global_static_R[0][1, 2], -1, 1)))
                print(f"[STATIC R] Global reference rotation computed from {ref_rel}  "
                      f"(implied pitch ≈ {pitch:.1f}°)")
            except Exception as e:
                print(f"[STATIC R] WARNING: Could not load reference frame {ref_rel}: {e}")
                print(f"[STATIC R] Will fall back to first-seen frame per folder.")
                self._global_static_R = None
    
   
    def filter_pairs_with_points(self, pairs):
        """
        Filter out pairs where the cropped point cloud is empty.

        Args:
            pairs (list): List of (image_path, depth_path) pairs.

        Returns:
            list: Filtered list of pairs.
        """
        valid_pairs = []
        for img_path, depth_path in pairs:
                                                          
            print(depth_path)
            data = np.load(depth_path)
            points = data['pts_cam']               
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

                                  
            cropped_pcd = self.crop_point_cloud(pcd)
            if len(cropped_pcd.points) > 0:                                                 
                valid_pairs.append((img_path, depth_path))
            else:
                print(f"Removed pair with empty point cloud: {img_path}, {depth_path}")

        return valid_pairs

    def crop_point_cloud(self, pcd):
        points = np.asarray(pcd.points)
        crop_bounding = np.array([
            [self.roi_x[0], 0, self.roi_z[0]],
            [self.roi_x[0], 0, self.roi_z[1]],
            [self.roi_x[1], 0, self.roi_z[1]],
            [self.roi_x[1], 0, self.roi_z[0]]
        ]).astype("float64")

        vol = o3d.visualization.SelectionPolygonVolume()
        vol.orthogonal_axis = "Y"
        vol.axis_min= -0.5
        vol.axis_max = 2
        vol.bounding_polygon = o3d.utility.Vector3dVector(crop_bounding)

        sol = vol.crop_point_cloud(pcd)
        return sol


    def create_image_depth_pairs(self, images_name):
        """
        Create a list of (image_name, depth_name) pairs by replacing 'img' with 'agg_depth' in the image name.

        Parameters:
            images_name (list): List of image file names.
            root_dir (str): Root directory where the images and depth files are stored.

        Returns:
            list: A list of tuples containing (image_name, depth_name).
        """
        pairs = []
        number_of_removed_pairs = 0
        for image_name in images_name:
                                                              
            depth_name = image_name.replace("img", "agg_depth")
            depth_name = depth_name.replace(".jpg", ".npz")

            
            image_path = Path(self.root_dir) / image_name
            depth_path = Path(self.root_dir) /depth_name
            
                                            
            if Path(depth_path).exists():
                pairs.append((str(image_path), str(depth_path)))
            else:
               number_of_removed_pairs += 1
        
        print(f"number of removed file pairs due to missing depth files: {number_of_removed_pairs}")
                                                                      

        return pairs
    
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

                               
            agg_index = {p.stem: p for p in agg_dir.iterdir() if p.is_file()}

                                          
            img_files = sorted(
                (p for p in img_dir.iterdir() if p.is_file()),
                key=lambda p: (len(p.name), p.name)
            )

            for img_path in img_files:
                agg_path = agg_index.get(img_path.stem)
                if agg_path:
                    self.pairs.append((img_path, agg_path))

                    
    def __len__(self):
        return len(self.pairs)
            
    
    def matrix2euler(self, m):
                     
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


    def crop_image (self, K, image, crop_box=None, intrinsics_preprocessed=False):
        """
        Either: fixed-crop to the road region (when self.crop_to_road=True) and
        round to a patch_size multiple, OR: resize the full image to 560x560.

        Returns the new image and the intrinsic adjusted accordingly.
        """
        W, H = image.size

                                                                        
        x0, y0, x1, y1 = self.fixed_crop_bbox
        x0 = max(0, min(int(x0), W))
        y0 = max(0, min(int(y0), H))
        x1 = max(x0, min(int(x1), W))
        y1 = max(y0, min(int(y1), H))
        cropped = image.crop((x0, y0, x1, y1))
        cw, ch = cropped.size

        align = self.crop_patch_align
        tw = max(align, (cw // align) * align)
        th = max(align, (ch // align) * align)
        out_img = cropped.resize((tw, th)) if (tw, th) != (cw, ch) else cropped
        sx = tw / float(cw)
        sy = th / float(ch)

        K_new = K.clone()
        if not intrinsics_preprocessed:
            K_new[0, 0] *= sx
            K_new[1, 1] *= sy
            K_new[0, 2] = (K_new[0, 2] - x0) * sx
            K_new[1, 2] = (K_new[1, 2] - y0) * sy
        return out_img, K_new

    def crop_image_square (self, K, image, crop_box=None, intrinsics_preprocessed=False):
        W, H = image.size

                                                                       
        w_c, h_c = 560, 560
        resized_image = image.resize((w_c, h_c))
        scale_x = w_c / W
        scale_y = h_c / H
        K_new = K.clone()
        if not intrinsics_preprocessed:
            K_new[0, 0] *= scale_x
            K_new[1, 1] *= scale_y
            K_new[0, 2] *= scale_x
            K_new[1, 2] *= scale_y
        return resized_image, K_new

    @staticmethod
    def _folder_key_from_path(img_path: str) -> str:
        """Extract scene folder name from an image path.
        Path pattern: .../scene_name/img/cam_X/frame.jpg → returns scene_name"""
        parts = Path(img_path).parts
                                                              
        for i, p in enumerate(parts):
            if p == 'img' and i > 0:
                return parts[i - 1]
                                                  
        return str(Path(img_path).parent.parent.parent.name)

    @staticmethod
    def _compute_R_vert2cam(n_world, R_c2w):
        """Compute R_vert2_cam and R_cam2_vert from road normal in world frame and camera-to-world rotation."""
        n_world = n_world / np.linalg.norm(n_world)
        R_w2c = R_c2w.T

        n_cam = (R_w2c @ n_world).astype(np.float32)
        n_cam = n_cam / np.linalg.norm(n_cam)

                                                                            
        if n_cam[1] > 0:
            n_cam = -n_cam

        y_vert = -n_cam                    
        z_cam = np.array([0, 0, 1], dtype=np.float32)
        z_vert = z_cam - np.dot(z_cam, y_vert) * y_vert
        z_vert = z_vert / np.linalg.norm(z_vert)
        x_vert = np.cross(y_vert, z_vert)
        x_vert = x_vert / np.linalg.norm(x_vert)

        R_vert2_cam = np.column_stack([x_vert, y_vert, z_vert]).astype(np.float32)
        R_cam2_vert = R_vert2_cam.T
        return R_vert2_cam, R_cam2_vert

    def _get_R_vert2cam(self, img_path, ground_info, extrinsic):
        """Get R_vert2_cam and R_cam2_vert.
        If use_static_rotation and global reference R exists: return it for ALL folders.
        Otherwise: compute per-frame from wheel-derived road normal."""
        if self.use_static_rotation and self._global_static_R is not None:
            return self._global_static_R

        n_world = ground_info['plane_normal_world'].numpy().astype(np.float32)
        R_c2w = extrinsic[:3, :3].numpy() if hasattr(extrinsic, 'numpy') else extrinsic[:3, :3]
        return self._compute_R_vert2cam(n_world, R_c2w)

    def get_preprocessed_data(self, path):
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def __getitem__(self, idx):
                                                                                                 
                                                                                                  
        if self.preprocessed_data:
            if self.mode == 'train':
                                                              
                preprocessed_dir = getattr(self, 'preprocessed_dir', "/data/rhf/train_preprocessed_small_data_thesis")
                preprocesed_path = os.path.join(preprocessed_dir, f"data_item_{idx:06d}.pkl.gz")
            else:
                preprocessed_dir = getattr(self, 'preprocessed_dir', "/data/rhf/val_preprocessed_small_data_thesis")
                preprocesed_path = os.path.join(preprocessed_dir, f"data_item_{idx:06d}.pkl.gz")

            data = self.get_preprocessed_data(preprocesed_path)

            ele_gt = torch.tensor(data['ele_gt'], dtype=torch.float32)
            mask = torch.tensor(data['mask'], dtype=torch.int8)
            voxel_uv_left =  torch.tensor(data['voxel_uv_left'], dtype=torch.long)
            timestamp =  data['timestamp_us']
            intrinsic = torch.tensor(data['intrinsics'], dtype=torch.float32)
            data_path = data['path']
            img_left = Image.open(data_path)
                                                                       

            if self.crop_to_road:
                img_left, intrinsic = self.crop_image(intrinsic, img_left, intrinsics_preprocessed=True)
            else:
                img_left, intrinsic = self.crop_image_square(intrinsic, img_left, intrinsics_preprocessed=True)

                                                                  
            feat_H = img_left.height // self.down_scale
            feat_W = img_left.width // self.down_scale
            
                                            
            valid_mask = (voxel_uv_left[0] >= 0) & (voxel_uv_left[0] < feat_W) &\
                        (voxel_uv_left[1] >= 0) & (voxel_uv_left[1] < feat_H)

            valid_ratio = valid_mask.sum().item() / valid_mask.numel()
            preproc_path_str = data.get('path', '')
            is_nardo_preproc = 'nardo' in str(preproc_path_str).lower()
            if False:                                       
                os.makedirs("frustum_test", exist_ok=True)
                tag = f"idx{idx}_ts{timestamp}"

                u = voxel_uv_left[0].numpy()
                v = voxel_uv_left[1].numpy()

                fig, axes = plt.subplots(1, 4, figsize=(32, 7))

                               
                ax = axes[0]
                ax.scatter(u[valid_mask.numpy()], v[valid_mask.numpy()],
                           s=0.3, alpha=0.3, c='green', label='in-frustum')
                ax.scatter(u[~valid_mask.numpy()], v[~valid_mask.numpy()],
                           s=0.3, alpha=0.3, c='red', label='out-of-frustum')
                ax.axhline(0, color='blue', ls='--', lw=0.8)
                ax.axhline(feat_H - 1, color='blue', ls='--', lw=0.8, label=f'feat_H={feat_H}')
                ax.axvline(0, color='blue', ls='--', lw=0.8)
                ax.axvline(feat_W - 1, color='blue', ls='--', lw=0.8, label=f'feat_W={feat_W}')
                ax.set_xlabel('u (px)')
                ax.set_ylabel('v (px)')
                ax.set_title(f'Voxel UV projections — {valid_ratio*100:.1f}% valid')
                ax.legend(markerscale=10, fontsize=8)
                ax.invert_yaxis()

                                                                  
                ax = axes[1]
                img_np = np.array(img_left)
                ax.imshow(img_np)
                scale = self.down_scale
                ax.scatter(u[valid_mask.numpy()] * scale, v[valid_mask.numpy()] * scale,
                           s=0.2, alpha=0.2, c='lime')
                ax.scatter(u[~valid_mask.numpy()] * scale, v[~valid_mask.numpy()] * scale,
                           s=0.2, alpha=0.2, c='red')
                                                            
                K_ds = intrinsic.numpy() / self.down_scale
                K_ds[2, 2] = 1.0
                for z_ref in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
                    v_line = K_ds[1, 1] * self.base_height / z_ref + K_ds[1, 2]
                    v_img = v_line * scale
                    if 0 <= v_img < img_np.shape[0]:
                        ax.axhline(v_img, color='yellow', ls='-', lw=0.6, alpha=0.7)
                        ax.text(5, v_img - 3, f'{z_ref}m', color='yellow', fontsize=7,
                                fontweight='bold', bbox=dict(boxstyle='round,pad=0.1',
                                facecolor='black', alpha=0.5))
                ax.set_title('Projections on SQUEEZED image (560*560)\n+ distance lines')
                ax.axis('off')

                                                                   
                ax = axes[2]
                                                                   
                img_unsqueezed = img_left.resize((532, 532))                                    
                img_unsq_np = np.array(img_unsqueezed)
                ax.imshow(img_unsq_np)
                                                                       
                u_unsq = u * scale * (532.0 / 560.0)                      
                v_unsq = v * scale * (532.0 / 560.0)                                
                valid_np = valid_mask.numpy()
                ax.scatter(u_unsq[valid_np], v_unsq[valid_np],
                           s=0.2, alpha=0.3, c='lime')
                ax.scatter(u_unsq[~valid_np], v_unsq[~valid_np],
                           s=0.2, alpha=0.3, c='red')
                for z_ref in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
                    v_line = K_ds[1, 1] * self.base_height / z_ref + K_ds[1, 2]
                    v_img = v_line * scale
                    if 0 <= v_img < 532:
                        ax.axhline(v_img, color='yellow', ls='-', lw=0.6, alpha=0.7)
                        ax.text(5, v_img - 3, f'{z_ref}m', color='yellow', fontsize=7,
                                fontweight='bold', bbox=dict(boxstyle='round,pad=0.1',
                                facecolor='black', alpha=0.5))
                ax.set_title('UNSQUEEZED (true aspect ratio)\n+ distance lines')
                ax.axis('off')

                                               
                ax = axes[3]
                ax.hist(v, bins=200, color='steelblue', edgecolor='none')
                ax.axvline(0, color='red', ls='--', label='v=0')
                ax.axvline(feat_H - 1, color='red', ls='--', label=f'v={feat_H-1}')
                ax.set_xlabel('v (feature-map px)')
                ax.set_ylabel('count')
                ax.set_title('v-coordinate distribution')
                ax.legend(fontsize=8)

                plt.suptitle(f'{data["path"]}\n'
                             f'K={intrinsic.numpy().tolist()}\n'
                             f'roi_z=[{self.roi_z[0]:.2f}, {self.roi_z[1]:.2f}], '
                             f'base_h={self.base_height:.3f}m, '
                             f'cam_h={data.get("camera_height", "?")}\n'
                             f'Original image: ~2248x2252 (square) -> resized to 952x532 (1.79:1 squeeze)'
                             , fontsize=8)
                plt.tight_layout()
                                                                         
                plt.close()

                print(f"[FRUSTUM DEBUG] saved frustum_test/frustum_{tag}.png  "
                      f"valid={valid_ratio*100:.1f}%  "
                      f"v range=[{v.min()}, {v.max()}]  "
                      f"u range=[{u.min()}, {u.max()}]  "
                      f"feat={feat_W}x{feat_H}")

            if valid_ratio < 0.7:
                raise ValueError(
                    f"Preprocessed data: Only {valid_mask.sum()}/{valid_mask.numel()} "
                    f"({valid_ratio*100:.1f}%) voxels project within camera frustum. "
                    f"Feature map size: {feat_W}x{feat_H}, "
                    f"UV range: [{u.min()},{u.max()}] x [{v.min()},{v.max()}]. "
                    f"Data path: {data_path}")
            
                                                                    
            voxel_uv_left[0] = voxel_uv_left[0].clamp(0, feat_W - 1)
            voxel_uv_left[1] = voxel_uv_left[1].clamp(0, feat_H - 1)
            
            x = transforms.ToTensor()(img_left)
                                     
                                      
            x = x.cpu().permute(1, 2, 0).numpy()
            x = (x * 255.0).clip(0, 255).astype("uint8")
            image_cropped = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
                                                       
            imgs_left = self.transform_jpg(img_left)
            print(data_path)
                                                         
                                                    
            if self.mode == 'train' and self.augmentation:                                

                a = random.random()
                             
                                                                                
                if a < 0.5:
                    if random.random() < 0.5:
                        imgs_left, intrinsic, voxel_uv_left, ele_gt, mask =\
                            _apply_flip(imgs_left, intrinsic,
                                            voxel_uv_left, ele_gt, mask,
                                            down_scale=self.down_scale)

                    if random.random() < 0.5:
                        None                                           
                    imgs_left = apply_gaussian_noise_and_blur(imgs_left)

                    if random.random() < 0.5:
                        ele_gt, mask = apply_gt_cutout(ele_gt, mask)

            if self.clamp_gt:
                ele_gt = torch.clamp(ele_gt, -self.y_range * 100, self.y_range * 100)
            return imgs_left, ele_gt, mask, voxel_uv_left, timestamp


        else:

            
            img_path = self.pairs[idx][0]
                            
            depth_path = self.pairs[idx][1]
            imgs_left = Image.open(img_path)
                                                  
                            
            extrinsic, intrinsic, _, ground_info, neighbours, labels = self.get_cam_payload_cached(str(img_path))
            extrinsic_inv = np.linalg.inv(extrinsic)
            crop_box = (604, 1124, 1696, 1642)
            if self.crop_to_road:
                imgs_left, intrinsic = self.crop_image(intrinsic, imgs_left, crop_box)
            else:
                imgs_left, intrinsic = self.crop_image_square(intrinsic, imgs_left, crop_box)
                    

            self.ground_normal_vector = ground_info['plane_normal_world']
            self.camera_height = ground_info['camera_height_above_ground']

            data = np.load(depth_path)
            points = data['pts_cam']            
                                                     
                                                                                                   
            R_vert2_cam, R_cam2_vert = self._get_R_vert2cam(img_path, ground_info, extrinsic)
            R_cam2vert = R_cam2_vert                                  

                                                      
            voxel_centers = self.voxel_centers                       


            voxel_cam_left = torch.tensor(R_vert2_cam) @ voxel_centers

                                  
            intrinsic_downscaled = (intrinsic / self.down_scale).to(torch.float32)
            intrinsic_downscaled[2, 2] = 1
            uvz_left =  intrinsic_downscaled @ voxel_cam_left
            voxel_uv_left = torch.floor(uvz_left[:2, :] / uvz_left[2:, :]).type(torch.long)

                                                                                   
            feat_H = transforms.ToTensor()(imgs_left).shape[-2] // self.down_scale
            feat_W = transforms.ToTensor()(imgs_left).shape[-1] // self.down_scale
                                                                     
                                                                     
            valid_mask = (voxel_uv_left[0] >= 0) & (voxel_uv_left[0] < feat_W) &\
                        (voxel_uv_left[1] >= 0) & (voxel_uv_left[1] < feat_H)

            valid_ratio = valid_mask.sum().item() / valid_mask.numel()

            folder_key = self._folder_key_from_path(str(img_path))
            is_nardo = 'nardo' in str(img_path).lower()
            if False:                                              
                os.makedirs("frustum_test", exist_ok=True)
                tag = f"online_idx{idx}_{folder_key}_{Path(img_path).stem}"
                u = voxel_uv_left[0].numpy()
                v = voxel_uv_left[1].numpy()

                                                                            
                R_v2c = torch.tensor(R_vert2_cam, dtype=torch.float32)
                K_ds = intrinsic_downscaled
                def vert_to_pixel(x_v, y_v, z_v):
                    """Project a vertical-frame 3D point to (u_img, v_img) in full image coords."""
                    p_vert = torch.tensor([[x_v], [y_v], [z_v]], dtype=torch.float32)
                    p_cam = R_v2c @ p_vert
                    uvz = K_ds @ p_cam
                    u_px = (uvz[0, 0] / uvz[2, 0]).item() * self.down_scale
                    v_px = (uvz[1, 0] / uvz[2, 0]).item() * self.down_scale
                    return u_px, v_px

                fig, axes = plt.subplots(1, 4, figsize=(32, 7))

                               
                ax = axes[0]
                ax.scatter(u[valid_mask.numpy()], v[valid_mask.numpy()],
                           s=0.3, alpha=0.3, c='green', label='in-frustum')
                ax.scatter(u[~valid_mask.numpy()], v[~valid_mask.numpy()],
                           s=0.3, alpha=0.3, c='red', label='out-of-frustum')
                ax.axhline(0, color='blue', ls='--', lw=0.8)
                ax.axhline(feat_H - 1, color='blue', ls='--', lw=0.8, label=f'feat_H={feat_H}')
                ax.axvline(0, color='blue', ls='--', lw=0.8)
                ax.axvline(feat_W - 1, color='blue', ls='--', lw=0.8, label=f'feat_W={feat_W}')
                ax.set_xlabel('u (px)')
                ax.set_ylabel('v (px)')
                ax.set_title(f'ONLINE Voxel UV — {valid_ratio*100:.1f}% valid')
                ax.legend(markerscale=10, fontsize=8)
                ax.invert_yaxis()

                                                                                     
                ax = axes[1]
                img_np = np.array(imgs_left)
                ax.imshow(img_np)
                scale = self.down_scale
                ax.scatter(u[valid_mask.numpy()] * scale, v[valid_mask.numpy()] * scale,
                           s=0.2, alpha=0.2, c='lime')
                ax.scatter(u[~valid_mask.numpy()] * scale, v[~valid_mask.numpy()] * scale,
                           s=0.2, alpha=0.2, c='red')
                                                                                              
                for z_ref in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
                    _, v_img = vert_to_pixel(0.0, self.base_height, float(z_ref))
                    if 0 <= v_img < img_np.shape[0]:
                        ax.axhline(v_img, color='yellow', ls='-', lw=0.6, alpha=0.7)
                        ax.text(5, v_img - 3, f'{z_ref}m', color='yellow', fontsize=7,
                                fontweight='bold', bbox=dict(boxstyle='round,pad=0.1',
                                facecolor='black', alpha=0.5))
                                               
                roi_corners_vert = [
                    (self.roi_x[0].item(), self.base_height, self.roi_z[0].item()),
                    (self.roi_x[1].item(), self.base_height, self.roi_z[0].item()),
                    (self.roi_x[1].item(), self.base_height, self.roi_z[1].item()),
                    (self.roi_x[0].item(), self.base_height, self.roi_z[1].item()),
                ]
                roi_px = [vert_to_pixel(*c) for c in roi_corners_vert]
                for i in range(4):
                    x0, y0 = roi_px[i]
                    x1, y1 = roi_px[(i+1)%4]
                    ax.plot([x0, x1], [y0, y1], color='cyan', lw=1.5, alpha=0.8)
                ax.set_title('ONLINE projections + TRUE distance lines\n(via R_vert2cam)')
                ax.axis('off')

                                                                   
                ax = axes[2]
                W_img, H_img = imgs_left.size if hasattr(imgs_left, 'size') else (img_np.shape[1], img_np.shape[0])
                unsq_size = (H_img, H_img)                                           
                img_unsq = imgs_left.resize(unsq_size) if hasattr(imgs_left, 'resize') else Image.fromarray(img_np).resize(unsq_size)
                img_unsq_np = np.array(img_unsq)
                ax.imshow(img_unsq_np)
                u_unsq = u * scale * (H_img / W_img)
                v_unsq = v * scale
                valid_np = valid_mask.numpy()
                ax.scatter(u_unsq[valid_np], v_unsq[valid_np], s=0.2, alpha=0.3, c='lime')
                ax.scatter(u_unsq[~valid_np], v_unsq[~valid_np], s=0.2, alpha=0.3, c='red')
                                                   
                unsq_ratio = H_img / W_img
                for z_ref in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
                    u_line, v_img = vert_to_pixel(0.0, self.base_height, float(z_ref))
                    if 0 <= v_img < H_img:
                        ax.axhline(v_img, color='yellow', ls='-', lw=0.6, alpha=0.7)
                        ax.text(5, v_img - 3, f'{z_ref}m', color='yellow', fontsize=7,
                                fontweight='bold', bbox=dict(boxstyle='round,pad=0.1',
                                facecolor='black', alpha=0.5))
                                       
                for i in range(4):
                    x0, y0 = roi_px[i]
                    x1, y1 = roi_px[(i+1)%4]
                    ax.plot([x0 * unsq_ratio, x1 * unsq_ratio], [y0, y1], color='cyan', lw=1.5, alpha=0.8)
                ax.set_title('ONLINE UNSQUEEZED + TRUE distance lines')
                ax.axis('off')

                                
                ax = axes[3]
                ax.hist(v, bins=200, color='steelblue', edgecolor='none')
                ax.axvline(0, color='red', ls='--', label='v=0')
                ax.axvline(feat_H - 1, color='red', ls='--', label=f'v={feat_H-1}')
                ax.set_xlabel('v (feature-map px)')
                ax.set_ylabel('count')
                ax.set_title('v-coordinate distribution')
                ax.legend(fontsize=8)

                plt.suptitle(f'ONLINE PATH: {img_path}\n'
                             f'K={intrinsic.numpy().tolist()}\n'
                             f'roi_z=[{self.roi_z[0]:.2f}, {self.roi_z[1]:.2f}], '
                             f'base_h={self.base_height:.3f}m, '
                             f'cam_h={ground_info.get("camera_height_above_ground", "?")}\n'
                             f'Original image: ~2248x2252 (square) -> resized to 952x532'
                             , fontsize=8)
                plt.tight_layout()
                                                                 
                plt.close()
                
                print(f"[FRUSTUM DEBUG ONLINE] saved frustum_test/{tag}.png  "
                      f"valid={valid_ratio*100:.1f}%  "
                      f"v=[{v.min()},{v.max()}] u=[{u.min()},{u.max()}] feat={feat_W}x{feat_H}")
            
            if valid_ratio < 0.7:
                raise ValueError(f"ONLINE: Only {valid_mask.sum()}/{valid_mask.numel()} voxels project within camera frustum. "
                                 f"See frustum_test/online_idx{idx}_*.png")

            pcd_cam2vert = o3d.geometry.PointCloud()
            pcd_cam2vert.points = o3d.utility.Vector3dVector(points)
            pcd_cam2vert.rotate(R_cam2_vert, center=(0, 0, 0))
                                                                     

            points = np.asarray(pcd_cam2vert.points)
                                                                                                                 
            """ crop_bounding = np.array([[self.roi_x[0], 0, self.roi_z[0]],
                                    [self.roi_x[0], 0, self.roi_z[1]],
                                    [self.roi_x[1], 0, self.roi_z[1]],
                                    [self.roi_x[1], 0, self.roi_z[0]]]).astype("float64")
            vol = o3d.visualization.SelectionPolygonVolume()
            vol.orthogonal_axis = "Y"
            vol.axis_max = 5
            vol.axis_min = 0
            vol.bounding_polygon = o3d.utility.Vector3dVector(crop_bounding)
            #self.save_gt_points(pcd_cam2vert, "before_crop" + str(img_path.stem) + ".png")
            pcd_cam2vert = vol.crop_point_cloud(pcd_cam2vert) """
                                                                                                 
            pcd_cam2vert = self.crop_point_cloud(pcd_cam2vert)
            points = np.asarray(pcd_cam2vert.points)
                                                                                 
                                                                                         
            app = o3d.visualization.gui.Application.instance
            app.initialize()
                                                                      

            camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                                   
            height_map, mask = self.get_gt_elevation(pcd_cam2vert)
                                                                                   
                              
            ele_gt = height_map         
        
            mask = mask.type(torch.float32)
                          
                                                           
            imgs_left = self.transform_jpg(imgs_left)
            if self.mode == 'train' and self.augmentation:                                

                a = random.random()
                             
                                                                                
                if a < 0.5:
                    imgs_left, intrinsic, voxel_uv_left, ele_gt, mask =\
                        _apply_flip(imgs_left, intrinsic,
                                        voxel_uv_left, ele_gt, mask)
                    

                if a < 0.5:
                    None                                           
                    imgs_left = apply_gaussian_noise_and_blur(imgs_left)

                if a < 0.5:
                    ele_gt, mask = apply_gt_cutout(ele_gt, mask)
            if self.clamp_gt:
                ele_gt = torch.clamp(ele_gt, -self.y_range * 100, self.y_range * 100)
            return imgs_left, ele_gt, mask, voxel_uv_left, ground_info['timestamp_us']

        return None
                                
                               
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        print(img_path)
        imgs_left = Image.open(img_path)


        extrinsic, intrinsic, _, ground_info = self.get_cam_payload(str(img_path))
        extrinsic_inv = np.linalg.inv(extrinsic)
        
        imgs_left, intrinsic = self.crop_image(intrinsic, imgs_left)

                      
        imgs_left = self.transform_jpg(imgs_left)

        print("mbouuuu", imgs_left.shape)


        self.ground_normal_vector = ground_info['plane_normal_world']
        self.camera_height = ground_info['camera_height_above_ground']

        print(extrinsic)
                                               
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        print(depth_path)
        data = np.load(depth_path)
        points = data['pts_cam']            
        print(f"Height before values (y): min={points[:, 1].min()}, max={points[:, 1].max()}")
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        R = coord_frame.get_rotation_matrix_from_xyz((0.0, 0.0, np.pi))                           
        coord_frame = coord_frame.rotate(R, center=(0, 0, 0))


        [pitch_cam, roll_cam, yaw_cam] = self.matrix2euler((extrinsic[:3, :3].numpy()))                                                        
        print(pitch_cam, roll_cam, yaw_cam)
        pitch_cam -= 1.5708        
        R_X = np.array(
            [[1, 0, 0], [0, np.cos(pitch_cam), np.sin(pitch_cam)], [0, -np.sin(pitch_cam), np.cos(pitch_cam)]], dtype=np.float32)
        R_Z = np.array(
            [[np.cos(roll_cam), np.sin(roll_cam), 0], [-np.sin(roll_cam), np.cos(roll_cam), 0], [0, 0, 1]], dtype=np.float32)
        R_cam2vert = R_X @ R_Z                                                                            
        R_vert2cam = torch.from_numpy(np.linalg.inv(R_cam2vert))
            
        
        R_cam2_vert = R_cam2vert
                                           
        R_vert2_cam = np.linalg.inv(R_cam2vert)

        camera_frame = copy.deepcopy(coord_frame)

        camera_frame.rotate(R_cam2vert, center=(0, 0, 0))

                                        
        voxel_centers = self.voxel_centers                       


        voxel_cam_left = torch.tensor(R_vert2_cam) @ voxel_centers

                              
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
                                                                                      
        pcd_cam2vert = vol.crop_point_cloud(pcd_cam2vert)
                                                                 
        points = np.asarray(pcd_cam2vert.points)
                                                                      
                                                                                     
        print(f"Height after crop values (y): min={points[:, 1].min()}, max={points[:, 1].max()}")

        app = o3d.visualization.gui.Application.instance
        app.initialize()
                                                                  

        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        enu = copy.deepcopy(camera)
        enu.rotate(extrinsic[:3, :3], center=(0,0,0))
        
        road = copy.deepcopy(camera)
        road.rotate(R_cam2_vert, center=(0,0,0))
                                           
                                     
        height_map, mask = self.get_gt_elevation(pcd_cam2vert)
        print(f"croppoint count{np.asarray(pcd_cam2vert.points)[:, 1].max()}")
                                                                                                                                                                  
        print(height_map.shape, height_map[:, 1])
        
                            
        ele_gt = height_map         
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


        if self.clamp_gt:
            ele_gt = torch.clamp(ele_gt, -self.y_range * 100, self.y_range * 100)
        return imgs_left, ele_gt, mask, voxel_uv_left, ground_info['timestamp_us']

    def _load_labels(self, seq_root: Path, cam: str, ts: int, img_path: str):
        p = seq_root / "labels" / f"{cam}_{ts}.txt"
        if not p.is_file(): return None
        img = cv2.imread(img_path); 
        if img is None: return None
        h, w = img.shape[:2]
        out = []
        with open(p, "r") as f:
            for line in f:
                a = line.strip().split()
                if len(a)==5:
                    cls,x_c,y_c,bw,bh = map(float, a)
                    x_c,y_c,bw,bh = x_c*w, y_c*h, bw*w, bh*h
                    x1,y1 = int(x_c-bw/2), int(y_c-bh/2)
                    x2,y2 = int(x_c+bw/2), int(y_c+bh/2)
                    out.append({"class": int(cls), "bbox": (x1,y1,x2,y2)})
        return out
    
    def visualize_voxel_uv(self, img: torch.Tensor, voxel_uv: torch.Tensor, dot_size: int = 3, color: tuple = (255, 0, 0)):
        """
        Visualize voxel UV projections on the image.

        Args:
            img         (3, H, W) torch.Tensor — normalized or uint8 image tensor
            voxel_uv    (2, N) torch.Tensor — UV coordinates (u, v)
            dot_size    radius of each projected point
            color       RGB color for the projections
        """
        print("erwan:", img.shape)
        _, H, W = img.shape
        img_np = img.permute(1, 2, 0).cpu().numpy()

                                            
        if img_np.max() > 1.0:
            img_np = img_np / 255.0

        u_coords = voxel_uv[0].cpu().numpy()*4
        v_coords = voxel_uv[1].cpu().numpy()*4

        in_bounds = (
            (u_coords >= 0) & (u_coords < W) &
            (v_coords >= 0) & (v_coords < H)
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(img_np)
        ax.scatter(
            u_coords[in_bounds], v_coords[in_bounds],
            s=dot_size,
            c=[[color[0]/255, color[1]/255, color[2]/255]],
            alpha=0.5
        )
        ax.set_title(f"{in_bounds.sum()} / {len(u_coords)} voxels in bounds")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig("voxel_uv_debug.png", dpi=150) if in_bounds.sum() == len(u_coords) else plt.savefig("voxel_uv_debug_wrong.png", dpi=150)
        plt.close()

    def visualize_height_map_and_mask(self, height_map, mask, colormap='plasma', save_path='Heightmap/height_map_visualization.png'):
        """
        Visualize the height map and mask with coordinate axes and proper colorbar.

        Parameters:
            height_map (torch.Tensor): The height map tensor of shape (H, W).
            mask (torch.Tensor): The mask tensor of shape (H, W), where 1 indicates valid points and 0 indicates no points.
            colormap (str): The colormap to use for valid height values (default: 'plasma').
            save_path (str): Path to save the visualization image.
        """
        print("save in:", save_path)
        height_map = height_map.cpu().numpy() if isinstance(height_map, torch.Tensor) else height_map
        mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

        valid_heights = height_map[mask > 0]
        h_min, h_max = float(valid_heights.min()), float(valid_heights.max())

                                                                         
        masked = np.ma.array(height_map, mask=(mask == 0))
        cmap = plt.cm.get_cmap(colormap).copy()
        cmap.set_bad(color='black')

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(
            masked,
            extent=[self.roi_x[0], self.roi_x[1], self.roi_z[0], self.roi_z[1]],
            aspect='auto', origin='upper',
            cmap=cmap, vmin=h_min, vmax=h_max,
        )

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Z (m)', fontsize=12)
        ax.set_title('Height Map Visualization', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.colorbar(im, ax=ax, label='Height (cm)', pad=0.02)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Height map visualization saved to {save_path}")


    def get_cam2vert(self, cam_up, normal_y_up):
        v = np.cross(cam_up, normal_y_up)
                             

        s = np.linalg.norm(v)              
        c = np.dot(cam_up, normal_y_up)              

        if s == 0:
            if c > 0:                        
                return np.eye(3)
            else:                        
                                              
                perp_axis = np.array([1, 0, 0])
                vx = np.array([[0, -perp_axis[2], perp_axis[1]],
                            [perp_axis[2], 0, -perp_axis[0]],
                            [-perp_axis[1], perp_axis[0], 0]])
                R = np.eye(3) + vx + vx @ vx * ((1 - (-1)) / (1**2))                 
                return R

        vx = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
        return R


    def save_gt_as_image(self, pcd, img_path, intrinsic):
        
                                                                 
        points = np.asarray(pcd.points)
        img = cv2.imread(img_path)

                                   
        uvz = (intrinsic @ points.T).T                 
        uv = uvz[:, :2] / uvz[:, 2].reshape(-1, 1)

                              
        for (u, v) in uv:
            u_int, v_int = int(u), int(v)
            if 0 <= u_int < img.shape[1] and 0 <= v_int < img.shape[0]:
                cv2.circle(img, (u_int, v_int), 2, (0, 255, 0), -1)              

                                
        output_path = "Pcd_on_image/visualization pcd" + str(img_path.stem) + ".jpg"
        cv2.imwrite(output_path, img)
                                                      
    
    def save_gt_points(self, pcd, str = " "):
        pts = np.asarray(pcd.points)
                                   
        x = pts[:, 0]                
        y = pts[:, 1]          
        z = pts[:, 2]           

                                                     
        plt.figure(figsize=(8, 4))
        scatter = plt.scatter(x, z, c=y, cmap='plasma', s=2)
        plt.colorbar(scatter, label='Height (Y)')
        plt.xlabel('X (Lateral)')
        plt.ylabel('Z (Longitidunal)')
        plt.title('f(x, z) = y encoded with plasma colormap')
        plt.tight_layout()

                                     
        plt.savefig('pointcloud_projection' + str + '.png', dpi=300)
        plt.show()

    def get_gt_elevation(self, xyz, camera_height=None):
                                      
        xyz = np.asarray(xyz.points)
        cam_h = camera_height if camera_height is not None else self.camera_height

        points_y = xyz[:, 1]*100                    
        points_xz = xyz[:, [0, 2]]
        grids_y = torch.zeros((self.num_grids_z, self.num_grids_x), dtype=torch.float32)
        grids_count = torch.zeros((self.num_grids_z, self.num_grids_x), dtype=torch.int32)                         

        for xz, y in zip(points_xz, points_y):
            idx_x = torch.clip(((xz[0] - self.roi_x[0]) / self.grid_res[0]).int(), max=self.num_grids_x-1)
            idx_z = torch.clip(self.num_grids_z - 1 - ((xz[1] - self.roi_z[0]) / self.grid_res[2]).int(), min=0)
            grids_y[idx_z, idx_x] += y
            grids_count[idx_z, idx_x] += 1
        mask = grids_count > 0

        grids_y[mask] = cam_h*100 - grids_y[mask] / grids_count[mask]

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
                         
        if intrinsic.shape != (3, 3):
            raise ValueError("Intrinsic matrix must be 3x3.")
        if extrinsic.shape != (4, 4):
            raise ValueError("Extrinsic matrix must be 4x4.")
        
                                                                        
        Rt = extrinsic[:3, :]       
        
                                   
        projection_matrix = intrinsic @ Rt       
        
        return projection_matrix
    
                                            
    def _cam_dir_list(self, seq_root: Path, cam: str) -> List[Tuple[int,str]]:
        key = (str(seq_root), cam)
        cam_dir = seq_root / "img" / cam
        out = []
        if cam_dir.is_dir():
            for p in cam_dir.iterdir():
                m = re.match(rf"{cam}_(\d+)\.(jpg|jpeg|png)$", p.name)
                if m:
                    out.append((int(m.group(1)), str(p)))
        out.sort(key=lambda x: x[0])
        return out
    
    def find_neighbour_index(self, traj:dict, timestamp_us:int, cam:str, total:int, context: Tuple[int, ...]):
        neighbour_index = {}
        for offset in context:
            ts = timestamp_us + offset
            neighbour_index[offset] = (ts if 0 <= ts < total else -1)

        return neighbour_index

    def get_neighbour_frames(self, traj:dict, seq_root: Path, cam_name:str, timestamp_us:int, context: Tuple[int, ...]):
        frames ={}
        clist = self._cam_dir_list(seq_root, cam_name)
        neighbour_index = self.find_neighbour_index(traj, timestamp_us, cam_name, len(clist), (-1, 1))
        for offset, ts in neighbour_index.items():
            ts_n, path_n = clist[ts]
            rgb_n = Image.open(path_n).convert("RGB")
            rgb_n = self.transform_jpg(rgb_n)
            extrinsics_n = self._T_cam_world(traj, cam_name, ts_n)
            frames[offset] = {
                "timestamp_us": ts_n,
                "path": path_n,
                "extrinsic": extrinsics_n,
                "rgb": rgb_n
            }
        return frames

    def get_cam_payload_cached(self, img_file):
        """Fast path for static rotation: caches trajectory, intrinsic, and ground_info per folder.
        Only the timestamp changes per frame. Falls back to full get_cam_payload if not static."""
        if not self.use_static_rotation:
            return self.get_cam_payload(img_file)

        folder_key = self._folder_key_from_path(img_file)
        seq_root, relative_path = self._seq_root_from_rel_CARD_Reconstruction(img_file)
        traj = self._get_traj(seq_root)          
        ts = self._ts_from_rel_CARD_Reconstruction(str(relative_path))
        cam_name, _ = self._parse_cam_ts(img_file)

        if folder_key in self._folder_payload_cache:
            intrinsic, cached_gp, cached_height = self._folder_payload_cache[folder_key]
                                                             
            gp_info = dict(cached_gp)
            gp_info['timestamp_us'] = ts
        else:
            intrinsic, _ = self._K_dist_from_traj(traj, cam_name)
            extrinsic = self._T_cam_world(traj, cam_name, ts)
            norm, pt, wh = self._compute_road_plane_from_wheels(traj, ts)
            h_cam = self._compute_camera_height_from_ground_plane(traj, cam_name, ts, norm, pt)
            gp_info = {
                "plane_normal_world": torch.from_numpy(norm), "plane_point_world": torch.from_numpy(pt),
                "wheel_points_world": torch.from_numpy(wh), "camera_height_above_ground": h_cam, "timestamp_us": ts,
            }
            self._folder_payload_cache[folder_key] = (intrinsic, gp_info, h_cam)
            print(f"[CACHE] Cached intrinsic + ground_info for folder '{folder_key}'")

                                                                              
        extrinsic = self._T_cam_world(traj, cam_name, ts)
        labels = self._load_labels(seq_root, cam_name, ts, img_file)
        img = self.get_img(Path(img_file))
        img = torch.from_numpy(img).permute(2,0,1).float()
        neighbours = self.get_neighbour_frames(traj, seq_root, cam_name, ts, context=(-1, 1))

        return torch.Tensor(extrinsic), torch.Tensor(intrinsic), img, gp_info, neighbours, labels

    def get_cam_payload(self, img_file):

        path_of_the_sequence, _ = self._seq_root_from_rel_CARD_Reconstruction(img_file)
        traj = self._get_traj(path_of_the_sequence)
        seq_root, relative_path = self._seq_root_from_rel_CARD_Reconstruction(img_file)
        img = self.get_img(Path(img_file))
        img = torch.from_numpy(img).permute(2,0,1).float()                       
        ts = self._ts_from_rel_CARD_Reconstruction(str(relative_path)) 
        cam_name, _ = self._parse_cam_ts(img_file)
        intrinsic, dist0 = self._K_dist_from_traj(traj, cam_name)
        extrinsic = self._T_cam_world(traj, cam_name, ts)
        labels = self._load_labels(seq_root, cam_name, ts, img_file)
        norm, pt, wh = self._compute_road_plane_from_wheels(traj, ts)
        h_cam = self._compute_camera_height_from_ground_plane(traj, cam_name, ts, norm, pt)
        gp_info = {
            "plane_normal_world": torch.from_numpy(norm), "plane_point_world": torch.from_numpy(pt),
            "wheel_points_world": torch.from_numpy(wh), "camera_height_above_ground": h_cam, "timestamp_us": ts,
        }
        neighbours = self.get_neighbour_frames(traj, path_of_the_sequence, cam_name, ts, context=(-1, 1))
        return torch.Tensor(extrinsic), torch.Tensor(intrinsic), torch.Tensor(img), gp_info, neighbours, labels


    def _seq_root_from_rel_CARD_Reconstruction(self, rel: str) :
                                                 
        parts = rel.replace("\\","/").split("/img/")
        rel_path = Path(parts[1]) if len(parts) > 1 else None
        return Path(parts[0]), rel_path
    
    def _get_traj(self, seq_root: Path):
        key = str(seq_root)
        if key in self._traj_cache:
            return self._traj_cache[key]
        traj_path = seq_root / "export" / "output.laz.trajectory.json"
                                                    
        with open(traj_path, "r") as f:
            d = json.load(f)
        self._traj_cache[key] = d
        return d
    
    @staticmethod
    def _parse_cam_ts(rel: str) -> Tuple[Optional[str], Optional[int]]:
        name = Path(rel).name                             
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
        K_normalized[0, 0] /= width       
        K_normalized[1, 1] /= height      
        K_normalized[0, 2] /= width       
        K_normalized[1, 2] /= height      

        return K_normalized
    
    def _compute_road_plane_from_wheels(self, traj: Dict, timestamp_us: int):
                                                                               
        wheel_keys = ["cariad_wheel_FL_ground", "cariad_wheel_FR_ground", "cariad_wheel_RL_ground", "cariad_wheel_RR_ground"]
        points = []
        for wk in wheel_keys:
            if wk in traj.get("sensor_to_trajectory_poses", {}):
                w_pose = traj["sensor_to_trajectory_poses"][wk]           
                                        
                v_pose = interpolate_pose(traj, timestamp_us)
                T_veh_world = pose_to_matrix(*v_pose)
                w_veh = np.array([w_pose[0], w_pose[1], w_pose[2], 1.0])
                points.append((T_veh_world @ w_veh)[:3])
        
        if len(points) < 3:
                      
            vp = interpolate_pose(traj, timestamp_us)
            p = np.array([vp[0], vp[1], vp[2]-1.8], dtype=np.float32)
            return (np.array([0,0,1],dtype=np.float32), p, np.array([p]))

        pts = np.array(points, dtype=np.float32)
        ctr = np.mean(pts, axis=0)
        _, _, vh = np.linalg.svd(pts - ctr)
        norm = vh[-1]
                                                                           
                                                                                   
        vp = interpolate_pose(traj, timestamp_us)
        cam_approx = np.array([vp[0], vp[1], vp[2]], dtype=np.float32)
        if np.dot(norm, cam_approx - ctr) < 0:
            norm = -norm
        return (norm, ctr, pts)

    def _compute_camera_height_from_ground_plane(self, traj, cam, ts, norm, pt):
        Tcw = self._T_cam_world(traj, cam, ts)
        cam_pt = Tcw[:3, 3]
        return float(np.dot(cam_pt - pt, norm / np.linalg.norm(norm)))

    def preprocess_and_save_data(self, output_dir="preprocessed_data", mode = "train", filter_by_labels=True):
        """
        Preprocesses all data items and saves them as pickle files.

        Args:
            output_dir: directory where compressed .pkl.gz files are written.
            mode: 'train' or 'test' - controls which keys end up in the saved dict.
            filter_by_labels: when True (default), keeps the original behaviour - skip
                frames with `labels is None`, and only save the 3 even-offset frames
                inside a 6-frame context window triggered by a labelled frame
                (potholes / speed bumps). When False, every frame in the split is
                processed and saved sequentially using the original split index in
                the filename, with no label-based filtering.
        """
                                                 
        os.makedirs(output_dir, exist_ok=True)
        save_idx = 0
        print(f"Starting preprocessing of {len(self.pairs)} items... (filter_by_labels={filter_by_labels})")
        context_frame = -1
        for idx in range(len(self.pairs)):
                                                                                 
                                                                            
            current_idx = idx if not filter_by_labels else save_idx
            save_path = os.path.join(output_dir, f"data_item_{current_idx:06d}.pkl.gz")

                                       
            if os.path.exists(save_path):
                print(f"Item {idx} already processed, skipping...")
                if not filter_by_labels:
                    pass                                  
                continue

            print(f"Processing item {idx}/{len(self.pairs)}...")

                                                                        
            img_path = self.pairs[idx][0]
            depth_path = self.pairs[idx][1]

                        
            imgs_left = Image.open(img_path)

                                   
            extrinsic, intrinsic, _, ground_info, neighbours, labels = self.get_cam_payload_cached(str(img_path))

            if labels is None:
                if filter_by_labels:
                    continue
                labels = []                                                           

            def process_frame(depth_path, img_path, imgs_left, extrinsic, intrinsic, ground_info, neighbours, labels):
                extrinsic_inv = np.linalg.inv(extrinsic)
                
                                                                         
                imgs_left, intrinsic = self.crop_image_square(intrinsic, imgs_left, intrinsics_preprocessed=False)
                imgs_left = self.transform_jpg(imgs_left)
                
                                          
                self.ground_normal_vector = ground_info['plane_normal_world']
                self.camera_height = ground_info['camera_height_above_ground']
                
                                 
                data = np.load(depth_path)
                points = data['pts_cam']             
                
                                                                                              
                R_vert2_cam, R_cam2_vert = self._get_R_vert2cam(img_path, ground_info, extrinsic)
                R_cam2vert = R_cam2_vert
                
                                       
                voxel_centers = self.voxel_centers
                voxel_cam_left = torch.tensor(R_vert2_cam) @ voxel_centers
                
                                      
                intrinsic_downscaled = (intrinsic / self.down_scale).to(torch.float32)
                intrinsic_downscaled[2, 2] = 1
                uvz_left = intrinsic_downscaled @ voxel_cam_left
                voxel_uv_left = torch.floor(uvz_left[:2, :] / uvz_left[2:, :]).type(torch.long)
                
                                     
                pcd_cam2vert = o3d.geometry.PointCloud()
                pcd_cam2vert.points = o3d.utility.Vector3dVector(points)
                pcd_cam2vert.rotate(R_cam2_vert, center=(0, 0, 0))
                
                points = np.asarray(pcd_cam2vert.points)
                
                                  
                pcd_cam2vert = self.crop_point_cloud(pcd_cam2vert)
                points = np.asarray(pcd_cam2vert.points)
                
                                         
                height_map, mask = self.get_gt_elevation(pcd_cam2vert)
                
                                    
                ele_gt = height_map
                mask = mask.type(torch.float16)
                
                                          
                if mode == 'train':
                    data_item = {
                                                        
                        'ele_gt': ele_gt.numpy(),
                        'mask': mask.numpy(),
                        'voxel_uv_left': voxel_uv_left.numpy(),
                        'timestamp_us': ground_info['timestamp_us'],
                        'ground_normal': ground_info['plane_normal_world'].numpy(),
                        'camera_height': ground_info['camera_height_above_ground'],
                        'extrinsics' : extrinsic.numpy(),
                        'intrinsics' : intrinsic.numpy(),
                                                                       
                                                                   
                        'extrinsic_previous': neighbours[-1]['extrinsic'],
                        'extrinsic_next': neighbours[1]['extrinsic'],
                        'path': str(img_path),                            
                        'depth_path': str(depth_path),
                        'pointcloud': points,
                    }
                else:
                    data_item ={
                        'imgs_left': imgs_left.numpy(),
                        'ele_gt': ele_gt.numpy(),
                        'mask': mask.numpy(),
                        'voxel_uv_left': voxel_uv_left.numpy(),
                        'timestamp_us': ground_info['timestamp_us'],
                        'path': str(img_path),                            
                    }
                                     
                with gzip.open(save_path, 'wb') as f:
                    pickle.dump(data_item, f)
                
                                                            
            if filter_by_labels:
                if len(labels) > 0 and context_frame == -1:
                    print(f"start to save files after potholes or sb have been detected on file{img_path}")
                    context_frame = 6

                if context_frame > -1:
                    if context_frame % 2 == 0:
                        process_frame(depth_path, img_path, imgs_left, extrinsic, intrinsic, ground_info, neighbours, labels)
                        save_idx += 1

                    context_frame -= 1
            else:
                                                             
                process_frame(depth_path, img_path, imgs_left, extrinsic, intrinsic, ground_info, neighbours, labels)
                save_idx += 1


        print(f"Preprocessing complete with {save_idx} saved files!")

class CARDSetDatasetV2Smalldataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 mode: str = 'train',
                 reprojection_loss = False,
                 down_scale = 4,
                 clamp_gt = False,
                 crop_to_road = False,
                 y_range = None,
                 num_grids_y = None):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.reprojection_loss = reprojection_loss
        self.clamp_gt = clamp_gt
        self.crop_to_road = crop_to_road
        self.crop_pad_frac = 0.10
        self.crop_patch_align = 14
        self.crop_target_size = 560
        

        self.img_dir = os.path.join(root_dir, 'img', 'cam_1')
        self.depth_dir = os.path.join(root_dir, 'agg_depth', 'cam_1')
        self.image_files = sorted(os.listdir(self.img_dir), key=lambda x: (len(x), x))
        self.depth_files = sorted(os.listdir(self.depth_dir), key=lambda x: (len(x), x))

        self.down_scale = down_scale

        self.base_height = 1.857                                                                    
        self.y_range = float(y_range) if y_range is not None else 0.2                                                
        self.roi_x = torch.tensor([-1, 0.92])                                                                                      
        self.roi_z = torch.tensor([5.16, 10.08])                                                  

        self.grid_res = torch.tensor([0.03, 0.01, 0.03])                                

        self.num_grids_x = int((self.roi_x[1] - self.roi_x[0]) / self.grid_res[0])
        self.num_grids_z = int((self.roi_z[1] - self.roi_z[0]) / self.grid_res[2])
        if num_grids_y is not None:
                                                                                                        
            self.num_grids_y = int(num_grids_y)
            self.grid_res[1] = (self.y_range * 2) / self.num_grids_y
        else:
            self.num_grids_y = int(self.y_range*2 / self.grid_res[1])

        len_images = len(self.image_files)
                                     
        len_depths = len(self.depth_files)
        assert len_images == len_depths, f"The number of images and depth files should be the same.{len_images} != {len_depths}"
        if(mode not in ['train', 'test']):
            raise ValueError("mode should be either 'train' or 'test'")
        
        elif mode == 'train':
                                   
            self.image_files = self.image_files[:len_images*9//10]
            self.depth_files = self.depth_files[:len_images*9//10]
        else:
            self.image_files = self.image_files[:len_images*9//10]
            self.depth_files = self.depth_files[:len_images*9//10]

                                                       
        hori_centers = torch.zeros((self.num_grids_z, self.num_grids_x, 2), dtype=torch.float32)
        hori_centers[:, :, 0] = (torch.arange(self.num_grids_x) * self.grid_res[0] + self.roi_x[0] + self.grid_res[0]/2).unsqueeze(0).repeat([self.num_grids_z, 1])
        hori_centers[:, :, 1] = (-torch.arange(self.num_grids_z) * self.grid_res[2] + self.roi_z[1] - self.grid_res[2]/2).unsqueeze(1).repeat([1, self.num_grids_x])
        self.map_centers = hori_centers.reshape(-1, 2)
        self.num_center = self.map_centers.shape[0]

                                                
        voxel_centers = torch.zeros((self.num_grids_z, self.num_grids_x, self.num_grids_y, 3), dtype=torch.float32)
        voxel_centers[:, :, :, [0, 2]] = hori_centers.unsqueeze(2).repeat([1, 1, self.num_grids_y, 1])
        voxel_centers[:, :, :, 1] = (torch.arange(self.num_grids_y) * self.grid_res[1] + self.base_height - self.y_range + self.grid_res[1]/2).unsqueeze(0).unsqueeze(0).repeat([self.num_grids_z, self.num_grids_x, 1])
        self.voxel_centers = voxel_centers.reshape(-1, 3).transpose(1, 0)
        
        self.transform_jpg = transforms.Compose([
            transforms.ToTensor(),                    
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])                      
        ])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
                               
                               
        img_path = os.path.join(self.img_dir, self.image_files[idx])
                       
        imgs_left = Image.open(img_path)

                                               
        extrinsic, intrinsic, _, ground_info, neighbours = self.get_cam_payload(str(img_path))
        extrinsic_inv = np.linalg.inv(extrinsic)
        
        imgs_left, intrinsic = self.crop_image(intrinsic, imgs_left)

                      
        imgs_left = self.transform_jpg(imgs_left)

                                          
        self.ground_normal_vector = ground_info['plane_normal_world']
        self.camera_height = ground_info['camera_height_above_ground']

                                  
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
                          
        data = np.load(depth_path)
        points_data = data['pts_cam']            
                                                                                               
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        R = coord_frame.get_rotation_matrix_from_xyz((0.0, 0.0, np.pi))                           
        coord_frame = coord_frame.rotate(R, center=(0, 0, 0))


        [pitch_cam, roll_cam, yaw_cam] = self.matrix2euler((extrinsic[:3, :3].numpy()))                                                        
                                            
        pitch_cam -= 1.5708        
        R_X = np.array(
            [[1, 0, 0], [0, np.cos(pitch_cam), np.sin(pitch_cam)], [0, -np.sin(pitch_cam), np.cos(pitch_cam)]], dtype=np.float32)
        R_Z = np.array(
            [[np.cos(roll_cam), np.sin(roll_cam), 0], [-np.sin(roll_cam), np.cos(roll_cam), 0], [0, 0, 1]], dtype=np.float32)
        R_cam2vert = R_X @ R_Z                                                                            
        R_vert2cam = torch.from_numpy(np.linalg.inv(R_cam2vert))
            
        
        R_cam2_vert = R_cam2vert
                                           
        R_vert2_cam = np.linalg.inv(R_cam2vert)

        camera_frame = copy.deepcopy(coord_frame)

        camera_frame.rotate(R_cam2vert, center=(0, 0, 0))

                                        
        voxel_centers = self.voxel_centers                       


        voxel_cam_left = torch.tensor(R_vert2_cam) @ voxel_centers

                              
        intrinsic_downscaled = (intrinsic / self.down_scale).to(torch.float32)
        intrinsic_downscaled[2, 2] = 1
        uvz_left =  intrinsic_downscaled @ voxel_cam_left
        voxel_uv_left = torch.floor(uvz_left[:2, :] / uvz_left[2:, :]).type(torch.long)

                                                     
        pcd_cam2vert = o3d.geometry.PointCloud()
        pcd_cam2vert.points = o3d.utility.Vector3dVector(points_data)
        pcd_cam2vert.rotate(R_cam2_vert, center=(0, 0, 0))
                                                                            
        crop_bounding = np.array([[self.roi_x[0], 0, self.roi_z[0]],
                                   [self.roi_x[0], 0, self.roi_z[1]],
                                   [self.roi_x[1], 0, self.roi_z[1]],
                                   [self.roi_x[1], 0, self.roi_z[0]]]).astype("float64")
        points = np.asarray(pcd_cam2vert.points)
                                                                                                             
        vol = o3d.visualization.SelectionPolygonVolume()
        vol.orthogonal_axis = "Y"
        vol.axis_max = 5
        vol.axis_min = -1
        vol.bounding_polygon = o3d.utility.Vector3dVector(crop_bounding)
                                                                                      
        pcd_cam2vert = vol.crop_point_cloud(pcd_cam2vert)
                                                                 
        points = np.asarray(pcd_cam2vert.points)
                                                                      
                                                                                     
        app = o3d.visualization.gui.Application.instance
        app.initialize()
                                                                  

        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        enu = copy.deepcopy(camera)
        enu.rotate(extrinsic[:3, :3], center=(0,0,0))
        
        road = copy.deepcopy(camera)
        road.rotate(R_cam2_vert, center=(0,0,0))
                                           
                                     
        height_map, mask = self.get_gt_elevation(pcd_cam2vert)
                                                                               
                                                                                                                                                                  
        ele_gt = height_map         
        if (idx == 0):
            None
                                                  
        mask = mask.type(torch.float32)
        if idx == 0:
            None
                                                     
                      
        if self.clamp_gt:
            ele_gt = torch.clamp(ele_gt, -self.y_range * 100, self.y_range * 100)
        return imgs_left, ele_gt, mask, voxel_uv_left, (None if self.reprojection_loss else ground_info, neighbours, extrinsic, intrinsic, points_data)


    def matrix2euler(self, m):
                     
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
        """
        Resize image and adjust camera intrinsics accordingly.
        Note: This function resizes, not crops (despite the name).
        """
        W, H = image.size                       
        w_c, h_c = 952, 518                     

                      
        resized_image = image.resize((w_c, h_c))

                                                       
        scale_x = w_c / W
        scale_y = h_c / H

                                      
        K_new = K.clone()
        K_new[0, 0] *= scale_x      
        K_new[1, 1] *= scale_y      
        K_new[0, 2] *= scale_x      
        K_new[1, 2] *= scale_y      

        return resized_image, K_new 
    
    @staticmethod
    def visualize_height_map_and_mask(height_map, mask, colormap='plasma', save_path='Heightmap/height_map_visualization.png', vmin= None, vmax=None):
        """
        Visualize the height map and mask. Cells without points are black, and cells with height GT are mapped with a colormap.

        Parameters:
            height_map (torch.Tensor): The height map tensor of shape (H, W).
            mask (torch.Tensor): The mask tensor of shape (H, W), where 1 indicates valid points and 0 indicates no points.
            colormap (str): The colormap to use for valid height values (default: 'plasma').
            save_path (str): Path to save the visualization image.
        """

                      
        if isinstance(height_map, torch.Tensor):
            height_map = height_map.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

                             
        height_map_vis = np.ma.masked_where(mask == 0, height_map)

                                     
        if vmin is None:
            vmin = height_map_vis.min()
            vmax = height_map_vis.max()

              
        plt.figure(figsize=(10, 6))
        im = plt.imshow(
            height_map_vis,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax
        )
        plt.axis("off")
        plt.title("Height Map (valid only)")

                        
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("cm")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def get_cam2vert(self, cam_up, normal_y_up):
        v = np.cross(cam_up, normal_y_up)
                             

        s = np.linalg.norm(v)              
        c = np.dot(cam_up, normal_y_up)              

        if s == 0:
            if c > 0:                        
                return np.eye(3)
            else:                        
                                              
                perp_axis = np.array([1, 0, 0])
                vx = np.array([[0, -perp_axis[2], perp_axis[1]],
                            [perp_axis[2], 0, -perp_axis[0]],
                            [-perp_axis[1], perp_axis[0], 0]])
                R = np.eye(3) + vx + vx @ vx * ((1 - (-1)) / (1**2))                 
                return R

        vx = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
        return R


    def save_gt_as_image(self, pcd, img_path, intrinsic):
        
                                       
        points = np.asarray(pcd.points)
        img = cv2.imread(img_path)

                                   
        uvz = (intrinsic @ points.T).T                 
        uv = uvz[:, :2] / uvz[:, 2].reshape(-1, 1)                

                              
        for (u, v) in uv:
            u_int, v_int = int(u), int(v)
            if 0 <= u_int < img.shape[1] and 0 <= v_int < img.shape[0]:
                cv2.circle(img, (u_int, v_int), 2, (0, 255, 0), -1)              

                                
        output_path = "visualization pcd.jpg"
        cv2.imwrite(output_path, img)
                                                      
    
    @staticmethod
    def save_gt_points(pcd, str = " "):

        if isinstance(pcd, o3d.geometry.PointCloud):
            pts = np.asarray(pcd.points)
        else:
            pts = np.asarray(pcd)
        

        x = pts[:, 0]                
        y = pts[:, 1]          
        z = pts[:, 2]           

                                                     
        plt.figure(figsize=(8, 4))
        scatter = plt.scatter(x, z, c=y, cmap='plasma', s=2)
        plt.colorbar(scatter, label='Height (Y)')
        plt.xlabel('X (Lateral)')
        plt.ylabel('Z (Longitidunal)')
        plt.title('f(x, z) = y encoded with plasma colormap')
        plt.tight_layout()

                                     
        plt.savefig('pointcloud_projection' + str + '.png', dpi=300)
        plt.show()

    def get_gt_elevation(self, xyz, camera_height=None):
                                      
        xyz = np.asarray(xyz.points)
        cam_h = camera_height if camera_height is not None else self.camera_height

        points_y = xyz[:, 1]*100                    
        points_xz = xyz[:, [0, 2]]
        grids_y = torch.zeros((self.num_grids_z, self.num_grids_x), dtype=torch.float32)
        grids_count = torch.zeros((self.num_grids_z, self.num_grids_x), dtype=torch.int32)                         

        for xz, y in zip(points_xz, points_y):
            idx_x = torch.clip(((xz[0] - self.roi_x[0]) / self.grid_res[0]).int(), max=self.num_grids_x-1)
            idx_z = torch.clip(self.num_grids_z - 1 - ((xz[1] - self.roi_z[0]) / self.grid_res[2]).int(), min=0)
            grids_y[idx_z, idx_x] += y
            grids_count[idx_z, idx_x] += 1
        mask = grids_count > 0

        grids_y[mask] = cam_h*100 - grids_y[mask] / grids_count[mask]

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
                         
        if intrinsic.shape != (3, 3):
            raise ValueError("Intrinsic matrix must be 3x3.")
        if extrinsic.shape != (4, 4):
            raise ValueError("Extrinsic matrix must be 4x4.")
        
                                                                        
        Rt = extrinsic[:3, :]       
        
                                   
        projection_matrix = intrinsic @ Rt       
        
        return projection_matrix


    def get_cam_payload(self, img_file):

        path_of_the_sequence, _ = self._seq_root_from_rel_CARD_Reconstruction(img_file)
        traj = self._get_traj(path_of_the_sequence)
        _, relative_path = self._seq_root_from_rel_CARD_Reconstruction(img_file)
        img = self.get_img(Path(img_file))
        img = torch.from_numpy(img).permute(2,0,1).float()                       
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
                                                 
        parts = rel.replace("\\","/").split("/img/")
        rel_path = Path(parts[1]) if len(parts) > 1 else None
        return Path(parts[0]), rel_path
    
    def _get_traj(self, seq_root: Path):
        key = str(seq_root)
        traj_path = seq_root.parent / "export" / "output.laz.trajectory.json"
        with open(traj_path, "r") as f:
            d = json.load(f)
                                  

        return d
    
    @staticmethod
    def _parse_cam_ts(rel: str) -> Tuple[Optional[str], Optional[int]]:
        name = Path(rel).name                             
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
        K_normalized[0, 0] /= width       
        K_normalized[1, 1] /= height      
        K_normalized[0, 2] /= width       
        K_normalized[1, 2] /= height      

        return K_normalized

                                            
    def _cam_dir_list(self, seq_root: Path, cam: str) -> List[Tuple[int,str]]:
        key = (str(seq_root), cam)
        if key in self._camdir_cache:
            return self._camdir_cache[key]
        cam_dir = seq_root / "img" / cam
        out = []
        if cam_dir.is_dir():
            for p in cam_dir.iterdir():
                m = re.match(rf"{cam}_(\d+)\.(jpg|jpeg|png)$", p.name)
                if m:
                    out.append((int(m.group(1)), str(p)))
        out.sort(key=lambda x: x[0])
        self._camdir_cache[key] = out
        return out
    

    def find_neighbour_index(self, traj:dict, timestamp_us:int, cam:str, total:int, context: Tuple[int, ...]):
        neighbour_index = {}
        for offset in context:
            ts = timestamp_us + offset
            neighbour_index[offset] = (ts if 0<= ts < total else -1)
    
    def get_neighbour_frames(self, traj:dict, seq_root: Path, cam_name:str, timestamp_us:int, context: Tuple[int, ...]):
        frames ={}
        clist = self._cam_dir_list(seq_root, cam_name)
        neighbour_index = self.find_neighbour_index(traj, timestamp_us, cam_name, (-1, 1))
        for offset, ts in neighbour_index.items():
            ts_n, path_n = clist[ts]
            rgb_n = Image.open(path_n).convert("RGB")
            extrinsics_n = self._T_cam_world(traj, cam_name, ts_n)
            frames[offset] = {
                "timestamp_us": ts_n,
                "path": path_n,
                "extrinsic": extrinsics_n,
                "rgb": rgb_n
            }
        return frames


    def _compute_road_plane_from_wheels(self, traj: Dict, timestamp_us: int):
                                                                               
        wheel_keys = ["cariad_wheel_FL_ground", "cariad_wheel_FR_ground", "cariad_wheel_RL_ground", "cariad_wheel_RR_ground"]
        points = []
        for wk in wheel_keys:
            if wk in traj.get("sensor_to_trajectory_poses", {}):
                w_pose = traj["sensor_to_trajectory_poses"][wk]           
                                        
                v_pose = interpolate_pose(traj, timestamp_us)
                T_veh_world = pose_to_matrix(*v_pose)
                w_veh = np.array([w_pose[0], w_pose[1], w_pose[2], 1.0])
                points.append((T_veh_world @ w_veh)[:3])
        
        if len(points) < 3:
                      
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
                                                
    pcd_o3d = o3d.t.geometry.PointCloud()
    pcd_o3d.point["positions"] = o3d.core.Tensor(pcd_np.astype(np.float32), dtype=o3d.core.Dtype.Float32)

                                                         
    intrinsics_o3d = o3d.core.Tensor(intrinsics_np.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    extrinsics_o3d = o3d.core.Tensor(extrinsics_np.astype(np.float32), dtype=o3d.core.Dtype.Float32)
                            
    depth_image = pcd_o3d.project_to_depth_image(
        width=width,
        height=height,
        intrinsics=intrinsics_o3d,
        extrinsics=extrinsics_o3d,
        depth_scale=depth_scale,
        depth_max=depth_max
    )

                                            
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
                    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found at {img_path}")
    height, width, _ = img.shape

                                    
    intrinsic_downscaled = intrinsic / down_scale
    intrinsic_downscaled[2, 2] = 1

                                                   
    voxel_cam = extrinsic[:3, :3] @ voxel_centers + extrinsic[:3, 3].unsqueeze(1)          

                                              
    uvz = intrinsic_downscaled @ voxel_cam          
    uv = uvz[:2, :] / uvz[2, :]                          

                                          
    uv = uv.T.cpu().numpy().astype(np.int32)                 

                                      
    for (u, v) in uv:
        if 0 <= u < width and 0 <= v < height:                                                 
            cv2.rectangle(img, (u - 5, v - 5), (u + 5, v + 5), (0, 255, 0), 2)             

                               
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

                      
    if vectors:
        for i, (start, end) in enumerate(vectors):
            line = o3d.geometry.LineSet()
            points = [start, end]
            lines = [[0, 1]]
            colors = [[1, 0, 0]]                    
            line.points = o3d.utility.Vector3dVector(points)
            line.lines = o3d.utility.Vector2iVector(lines)
            line.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(line)

                                       
            if labels and i < len(labels):
                label_pos = (np.array(start) + np.array(end)) / 2                                                 
                text = create_text_3d(labels[i], label_pos, font_size=0.2)
                geometries.append(text)

                   
    if plane:
        normal, point = plane
        normal = np.array(normal)
        point = np.array(point)

                                      
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

                                         
        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(corners)
        plane_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [2, 3, 0]])
        plane_mesh.paint_uniform_color([0, 1, 0])                      
        geometries.append(plane_mesh)

                    
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_length)
    geometries.append(axes)

                            
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
                                                   
    camera_up = np.array([0, -1, 0])
    if np.dot(ground_normal, camera_up) < 0:
        ground_normal = -ground_normal
    n = ground_normal / np.linalg.norm(ground_normal)

                                      
    camera_forward = np.array([0,0,1])
    X = np.cross(camera_forward, n)
    X /= np.linalg.norm(X)
    Y = np.cross(n, X)
    R = np.vstack((X, Y, n))        

                                                     
    ground_heights = []
    for p in ground_points:
        p_rot = R @ p
        ground_heights.append(p_rot[2])

    ground_height = np.mean(ground_heights)

                                      
    points_world = (R @ points_cam.T).T
    points_world[:,2] -= ground_height

    return points_world, R, ground_height


def save_image_path_in_list(input_dir, output_dir):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    with open(output_dir, "w") as out_f:
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".gz"):
                file_path = os.path.join(input_dir, file_name)

                with gzip.open(file_path, "rb") as f:
                    data = pickle.load(f)
                    path = data["path"]
                    if "cariad dataset" in path:
                        parts = path.split("cariad dataset/")
                        relative_path = parts[1]
                        out_f.write(relative_path + "\n")

        
    print("finished making small_data_training_list")

if __name__ == "__main__":
                                                                                                       
                                                                                                
    """ import numpy as np
    import cv2
    print(os.listdir())
    # ---------- load npz ----------
    data = np.load("cardset/cam_1_3501825.npz")

    # show keys inside npz
    print("data", data['pts_cam'])

    # choose array (change key if needed)
    arr = data['pts_cam']   # first array inside npz
    # Extract x, y, z coordinates
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]  # depth

    # Normalize x, y to pixel coordinates (0 to image_width/height)
    image_width = 640
    image_height = 480

    x_norm = ((x - x.min()) / (x.max() - x.min()) * (image_width - 1)).astype(int)
    y_norm = ((y - y.min()) / (y.max() - y.min()) * (image_height - 1)).astype(int)

    # Clamp to valid range
    x_norm = np.clip(x_norm, 0, image_width - 1)
    y_norm = np.clip(y_norm, 0, image_height - 1)

    # Create empty depth map
    depth_map = np.full((image_height, image_width), np.nan, dtype=np.float32)

    # Fill depth map (last point wins if multiple points map to same pixel)
    for i in range(len(arr)):
        depth_map[y_norm[i], x_norm[i]] = z[i]

    # Normalize depth values to 0-255 range
    depth_min = np.nanmin(depth_map)
    depth_max = np.nanmax(depth_map)
    depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255)
    depth_normalized = np.nan_to_num(depth_normalized, nan=0).astype(np.uint8)

    # Save as PNG
    cv2.imwrite("depth_map.png", depth_normalized)
    print(f"Saved: depth_map.png ({image_height}x{image_width})")
    print(f"Depth range: {depth_min:.2f} to {depth_max:.2f}")

    """
    save_image_path_in_list(input_dir = "/data/rhf/val_preprocessed_data_final", output_dir = "/data/rhf/val_dataset_paper.txt")
    save_image_path_in_list(input_dir = "/data/rhf/train_preprocessed_data_final", output_dir = "/data/rhf/train_dataset_paper.txt")
 
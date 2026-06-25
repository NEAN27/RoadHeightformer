import numpy as np
import math
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import PIL.Image
from torchvision import transforms
import open3d as o3d
import copy, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from RSRD_dev_toolkitmain.cam_extrinsic import Extrinsic

class RSRD(Dataset):
    def __init__(self, training=True, stereo=False, down_scale=2, backbone=None):
        super(RSRD, self).__init__()
        self.training = training
        self.stereo = stereo
        self.down_scale = down_scale
        self.backbone = backbone or ''
                                                                             
                                                                            
        self.is_rhf = 'DINOv2' in self.backbone
        if self.is_rhf:
            self.rhf_pad_stride = 14 * self.down_scale // math.gcd(14, self.down_scale)

        self.calib_path = 'calibration_files'                              
                                                                                                                             
        self.data_path = '/data/RSRD/RSRD-dense/train'   
        preprocessed_path = './preprocessed/'                                 

        if self.training:
            self.load_dataset_names('./filenames/train/')
            self.preprocessed_path = os.path.join(preprocessed_path, 'train')
        else:
                                                                    
            self.load_dataset_names('./filenames/train/')
            self.preprocessed_path = os.path.join(preprocessed_path, 'train')

                               
        self.base_height = 1.1                                                                    
        self.y_range = 0.2                                                                                        
        self.roi_x = torch.tensor([-1, 0.92])                                                                                      
        self.roi_z = torch.tensor([2.16, 7.08])                                                  
                               
        
        self.grid_res = torch.tensor([0.03, 0.01, 0.03])                                                                                                        
        

        self.num_grids_x = int((self.roi_x[1] - self.roi_x[0]) / self.grid_res[0])
        self.num_grids_z = int((self.roi_z[1] - self.roi_z[0]) / self.grid_res[2])
        self.num_grids_y = int(self.y_range*2 / self.grid_res[1])

                                                       
        hori_centers = torch.zeros((self.num_grids_z, self.num_grids_x, 2), dtype=torch.float32)
        hori_centers[:, :, 0] = (torch.arange(self.num_grids_x) * self.grid_res[0] + self.roi_x[0] + self.grid_res[0]/2).unsqueeze(0).repeat([self.num_grids_z, 1])
        hori_centers[:, :, 1] = (-torch.arange(self.num_grids_z) * self.grid_res[2] + self.roi_z[1] - self.grid_res[2]/2).unsqueeze(1).repeat([1, self.num_grids_x])
        self.map_centers = hori_centers.reshape(-1, 2)
        self.num_center = self.map_centers.shape[0]
        self.hori_centers = hori_centers
                                                
        voxel_centers = torch.zeros((self.num_grids_z, self.num_grids_x, self.num_grids_y, 3), dtype=torch.float32)
        voxel_centers[:, :, :, [0, 2]] = hori_centers.unsqueeze(2).repeat([1, 1, self.num_grids_y, 1])
        voxel_centers[:, :, :, 1] = (torch.arange(self.num_grids_y) * self.grid_res[1] + self.base_height - self.y_range + self.grid_res[1]/2).unsqueeze(0).unsqueeze(0).repeat([self.num_grids_z, self.num_grids_x, 1])
        self.voxel_centers = voxel_centers.reshape(-1, 3).transpose(1, 0)

                                                                    
        calib_files = ['calib_20230317_half.pkl', 'calib_20230321_half.pkl', 'calib_20230406_half.pkl', 'calib_20230408_half.pkl', 'calib_20230409_half.pkl']
        self.calib_params_all = {}
        for file in calib_files:
            with open(os.path.join(self.calib_path, file), 'rb') as f:
                calib_params = pickle.load(f)
            calib_params['K'] = calib_params['K'].astype(np.float32)
            calib_params['R'] = calib_params['R'].astype(np.float32)
            calib_params['T'] = calib_params['T'].astype(np.float32)
            calib_params['B'] = calib_params['B']/1000            
                                                                    
            calib_params['K_feat_T'] = torch.from_numpy(calib_params['K'] / self.down_scale)

            calib_params['K_feat_T'][2, 2] = 1
            calib_params['R_inv'] = np.linalg.inv(calib_params['R']).astype(np.float32)
            date = file[6:14]
            self.calib_params_all[date] = calib_params

        self.transform_jpg = transforms.Compose([
            transforms.ToTensor(),                    
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])                      
        ])

    def _pad_for_rhf(self, img):
        """Right/bottom-pad a [C, H, W] tensor so H and W are multiples of
        lcm(14, down_scale). Padding (vs. resize) preserves the intrinsics, so
        K_feat_T and the precomputed voxel_uv projections remain valid — the
        original image content stays at the top-left and padded pixels lie
        outside the road ROI."""
        if not self.is_rhf:
            return img
        s = self.rhf_pad_stride
        _, H, W = img.shape
        pad_h = (s - H % s) % s
        pad_w = (s - W % s) % s
        if pad_h == 0 and pad_w == 0:
            return img
        return F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0.0)

    def load_dataset_names(self, sample_path):
        data_all = []
        files = sorted(os.listdir(sample_path))
        for file in files:
            with open(os.path.join(sample_path, file), 'rb') as f:
                data = pickle.load(f)
            data_all += data
        self.data_all = data_all
                                                                        

    def get_lidar2cam(self, date_stamp):
                                            
        date = date_stamp[:8]
        return self.calib_params_all[date]

    def yaw_convert(self, yaw):
        '''
            convert the yaw data from [0, 2pi] to [-pi, pi]
        '''
        if np.pi <= yaw <= 2 * np.pi:
            yaw -= 2 * np.pi
        return yaw

    def lla_to_enu(self, C_lat, C_lon, C_alt, O_lat, O_lon, O_alt):
        '''
            Calculate the relative location with respect to the selected origin in the local ENU coordinate. unit: meter
            C_lat, C_lon, C_alt: current location
            O_lat, O_lon, O_alt: origin location
        '''
        Ea = 6378137
        Eb = 6356752.3142
        C_lat = math.radians(C_lat)
        C_lon = math.radians(C_lon)
        O_lat = math.radians(O_lat)
        O_lon = math.radians(O_lon)
        Ec = Ea * (1 - (Ea - Eb) / Ea * (math.sin(C_lat)) ** 2) + C_alt
        d_lat = C_lat - O_lat
        d_lon = C_lon - O_lon
        e = d_lon * Ec * math.cos(C_lat)
        n = d_lat * Ec
        u = C_alt - O_alt
        return np.array([e, n, u])

    def get_RT_lidar(self, loc_pose):
                        
                                                                    
        rotX_cur = 0.017453 * loc_pose['pitch']
        rotY_cur = 0.017453 * loc_pose['roll']
        rotZ_cur = 0.017453 * loc_pose['yaw']
        rotZ_cur = self.yaw_convert(rotZ_cur)

                                                                                             
        R_X1 = np.array([[1, 0, 0], [0, np.cos(rotX_cur), -np.sin(rotX_cur)], [0, np.sin(rotX_cur), np.cos(rotX_cur)]])
        R_Y1 = np.array([[np.cos(rotY_cur), 0, np.sin(rotY_cur)], [0, 1, 0], [-np.sin(rotY_cur), 0, np.cos(rotY_cur)]])
        R_Z1 = np.array([[np.cos(rotZ_cur), -np.sin(rotZ_cur), 0], [np.sin(rotZ_cur), np.cos(rotZ_cur), 0], [0, 0, 1]])
        R_cur2enu = R_Z1 @ R_X1 @ R_Y1                                           

        return R_cur2enu

    def get_gt_elevation(self, xyz):
        xyz = np.asarray(xyz.points)
        N, _ = xyz.shape
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
        grids_y[mask] = self.base_height*100 - grids_y[mask] / grids_count[mask]

        return grids_y, mask

    def get_gt_preprocessed(self, time):
        with open(os.path.join(self.preprocessed_path, time)+'.pkl', 'rb') as f:
            [ele_gt, ele_mask] = pickle.load(f)
        return ele_gt, ele_mask

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

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        sample_cur = self.data_all[index]
        l2c_calib_cur = self.get_lidar2cam(sample_cur['time'])
        path_base = sample_cur['path']
        idx_str = path_base.find('/')
        path_base = path_base[idx_str + 1:]

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                                                                                                   
                                                              
        R_cur2enu = self.get_RT_lidar(sample_cur)
                                                  
        [pitch_cam, roll_cam, _] = self.matrix2euler(l2c_calib_cur['R'] @ np.linalg.inv(R_cur2enu))
        pitch_cam -= 1.5708        
        R_X = np.array(
            [[1, 0, 0], [0, np.cos(pitch_cam), np.sin(pitch_cam)], [0, -np.sin(pitch_cam), np.cos(pitch_cam)]], dtype=np.float32)
        R_Z = np.array(
            [[np.cos(roll_cam), np.sin(roll_cam), 0], [-np.sin(roll_cam), np.cos(roll_cam), 0], [0, 0, 1]], dtype=np.float32)
        R_cam2vert = R_X @ R_Z                                                                            
        R_vert2cam = torch.from_numpy(np.linalg.inv(R_cam2vert))

        mou = copy.deepcopy(coord_frame)
        mou = mou.rotate(l2c_calib_cur['R'], center=(0, 0, 0))
        mou = mou.translate(tuple(l2c_calib_cur['T'].reshape(-1)))

        road_frame = copy.deepcopy(mou)
        road_frame.rotate(R_cam2vert, center=(0, 0 , 0))
                                                            

        ele_gt, ele_mask = self.get_gt_preprocessed(sample_cur['time'])
                                           

        path_img = os.path.join(self.data_path, path_base, 'left_half', sample_cur['time']) + '.jpg'
        img = PIL.Image.open(path_img).crop((0, 0, 960, 528))
        imgs_left = self._pad_for_rhf(self.transform_jpg(img))
                                       

        voxel_cam_left = R_vert2cam @ self.voxel_centers
        if self.stereo:
                                                                                                         
            voxel_cam_right = copy.deepcopy(voxel_cam_left)
                                              
            voxel_cam_right[0, :] = voxel_cam_right[0, :] - l2c_calib_cur['B']
            uvz_left = l2c_calib_cur['K_feat_T'] @ voxel_cam_left
            uvz_right = l2c_calib_cur['K_feat_T'] @ voxel_cam_right                                         
            voxel_uv_left = torch.floor(uvz_left[:2, :] / uvz_left[2:, :]).type(torch.long)
            voxel_uv_right = torch.floor(uvz_right[:2, :] / uvz_right[2:, :]).type(torch.long)

            path_img = os.path.join(self.data_path, path_base, 'right_half', sample_cur['time']) + '.jpg'
            img = PIL.Image.open(path_img).crop((0, 0, 960, 528))
            imgs_right = self._pad_for_rhf(self.transform_jpg(img))

            return imgs_left, imgs_right, ele_gt, ele_mask, voxel_uv_left, voxel_uv_right, sample_cur['time']
        else:
            uvz_left = l2c_calib_cur['K_feat_T'] @ voxel_cam_left
            voxel_uv_left = torch.floor(uvz_left[:2, :] / uvz_left[2:, :]).type(torch.long)

                                                                  
            feat_H = 528 // self.down_scale
            feat_W = 960 // self.down_scale
            
                                            
            valid_mask = (voxel_uv_left[0] >= 0) & (voxel_uv_left[0] < feat_W) &\
                        (voxel_uv_left[1] >= 0) & (voxel_uv_left[1] < feat_H)

            valid_ratio = valid_mask.sum().item() / valid_mask.numel()
            if valid_ratio < 0.7 or index < 3:                                               
                                                                            
                os.makedirs("frustum_debug", exist_ok=True)
                tag = f"rsrd_idx{index}_ts{sample_cur['time']}"

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
                img_np = np.array(img)
                ax.imshow(img_np)
                scale = self.down_scale
                ax.scatter(u[valid_mask.numpy()] * scale, v[valid_mask.numpy()] * scale,
                           s=0.2, alpha=0.2, c='lime')
                ax.scatter(u[~valid_mask.numpy()] * scale, v[~valid_mask.numpy()] * scale,
                           s=0.2, alpha=0.2, c='red')
                                                            
                K_ds = l2c_calib_cur['K_feat_T'].numpy()
                for z_ref in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
                    v_line = K_ds[1, 1] * self.base_height / z_ref + K_ds[1, 2]
                    v_img = v_line * scale
                    if 0 <= v_img < img_np.shape[0]:
                        ax.axhline(v_img, color='yellow', ls='-', lw=0.6, alpha=0.7)
                        ax.text(5, v_img - 3, f'{z_ref}m', color='yellow', fontsize=7,
                                fontweight='bold', bbox=dict(boxstyle='round,pad=0.1',
                                facecolor='black', alpha=0.5))
                ax.set_title('Projections on image + distance lines')
                ax.axis('off')

                                                
                ax = axes[2]
                img_unsqueezed = img.resize((528, 528))                  
                img_unsq_np = np.array(img_unsqueezed)
                ax.imshow(img_unsq_np)
                                                                       
                u_unsq = u * scale * (528.0 / 960.0)                      
                v_unsq = v * scale                       
                valid_np = valid_mask.numpy()
                ax.scatter(u_unsq[valid_np], v_unsq[valid_np],
                           s=0.2, alpha=0.3, c='lime')
                ax.scatter(u_unsq[~valid_np], v_unsq[~valid_np],
                           s=0.2, alpha=0.3, c='red')
                for z_ref in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
                    v_line = K_ds[1, 1] * self.base_height / z_ref + K_ds[1, 2]
                    v_img = v_line * scale
                    if 0 <= v_img < 528:
                        ax.axhline(v_img, color='yellow', ls='-', lw=0.6, alpha=0.7)
                        ax.text(5, v_img - 3, f'{z_ref}m', color='yellow', fontsize=7,
                                fontweight='bold', bbox=dict(boxstyle='round,pad=0.1',
                                facecolor='black', alpha=0.5))
                ax.set_title('Unsqueezed + distance lines')
                ax.axis('off')

                                               
                ax = axes[3]
                ax.hist(v, bins=200, color='steelblue', edgecolor='none')
                ax.axvline(0, color='red', ls='--', label='v=0')
                ax.axvline(feat_H - 1, color='red', ls='--', label=f'v={feat_H-1}')
                ax.set_xlabel('v (feature-map px)')
                ax.set_ylabel('count')
                ax.set_title('v-coordinate distribution')
                ax.legend(fontsize=8)

                plt.suptitle(f'{path_img}\n'
                             f'K={l2c_calib_cur["K"].tolist()}\n'
                             f'roi_z=[{self.roi_z[0]:.2f}, {self.roi_z[1]:.2f}], '
                             f'base_h={self.base_height:.3f}m\n'
                             f'Original image: 960x528 cropped'
                             , fontsize=8)
                plt.tight_layout()
                                                                  
                plt.close()

                                                                               
            if valid_ratio < 0.7:
                raise ValueError(
                    f"RSRD: Only {valid_mask.sum()}/{valid_mask.numel()} "
                    f"({valid_ratio*100:.1f}%) voxels project within camera frustum. "
                    f"Feature map size: {feat_W}x{feat_H}, "
                    f"UV range: [{u.min()},{u.max()}] x [{v.min()},{v.max()}]. "
                    f"See frustum_debug/{tag}.png "
                    f"Data path: {path_img}")
            
                              
            voxel_uv_left[0] = voxel_uv_left[0].clamp(0, feat_W - 1)
            voxel_uv_left[1] = voxel_uv_left[1].clamp(0, feat_H - 1)

            if index == 0:
                                                   
                                                       
                T_l2c = l2c_calib_cur['T']
                R_l2c = l2c_calib_cur['R']
                extrinsic_matrix = np.eye(4, dtype=np.float32)
                extrinsic_matrix[:3, :3] = R_l2c
                extrinsic_matrix[:3, 3] = T_l2c.flatten()
                                         
                                                                      
            path_pcd = os.path.join(self.data_path, path_base, 'pcd', sample_cur['time']) + '.pcd'
            cloud = o3d.io.read_point_cloud(path_pcd)
            cloud = cloud.rotate(l2c_calib_cur['R'], center=(0, 0, 0))
            cloud = cloud.translate(tuple(l2c_calib_cur['T'].reshape(-1)))                                         
            
                                     
            cloud_camvert = cloud.rotate(R_cam2vert, center=(0, 0, 0))
                                                                                                                 
                                                     
            crop_bounding = np.array([[self.roi_x[0], 0, self.roi_z[0]],
                                   [self.roi_x[0], 0, self.roi_z[1]],
                                   [self.roi_x[1], 0, self.roi_z[1]],
                                   [self.roi_x[1], 0, self.roi_z[0]]]).astype("float64")
            
            vol_roi = o3d.visualization.SelectionPolygonVolume()
            vol_roi.orthogonal_axis = "Y"
            vol_roi.axis_max = 1.5
            vol_roi.axis_min = 0.5
            vol_roi.bounding_polygon = o3d.utility.Vector3dVector(crop_bounding)

            cloud_camvert = vol_roi.crop_point_cloud(cloud_camvert)

                                                                 
            return imgs_left, ele_gt, ele_mask, voxel_uv_left, sample_cur['time']
    
    def save_gt_as_image(self, pcd, img_path, intrinsic, addi = ' ', index = 0):
        
                                                                 
        points = np.asarray(pcd.points)
        img = cv2.imread(img_path)

                                   
        uvz = (intrinsic @ points.T).T                 
        uv = uvz[:, :2] / uvz[:, 2].reshape(-1, 1)

                              
        for (u, v) in uv:
            u_int, v_int = int(u), int(v)
            if 0 <= u_int < img.shape[1] and 0 <= v_int < img.shape[0]:
                cv2.circle(img, (u_int, v_int), 2, (0, 255, 0), -1)              

                                
    def save_gt_points(self, pcd, str=''):
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

                                    
    intrinsic_downscaled = intrinsic / 1

                                                   
    voxel_cam = extrinsic[:3, :3] @ voxel_centers + extrinsic[:3, 3].unsqueeze(1)          

                                              
    uvz = intrinsic_downscaled @ voxel_cam          
    uv = uvz[:2, :] / uvz[2, :]                          

                                          
    uv = uv.T.cpu().numpy().astype(np.int32)                 

                                      
    for (u, v) in uv:
        if 0 <= u < width and 0 <= v < height:                                                 
            cv2.rectangle(img, (u - 5, v - 5), (u + 5, v + 5), (0, 255, 0))             

                               
if __name__ == '__main__':
    dataset = RSRD(down_scale=2, training=False, stereo=False)
    for i in range(len(dataset)):
        dataset.__getitem__(i)

    
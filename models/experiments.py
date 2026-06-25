"""
Goal of this file is to test the different stage of the reprojection loss:
- homography estimation and application
- flow residual application
"""

from pathlib import Path
import os, sys
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import torch
from reprojection_loss import HomographyWarp, ReprojectionLoss
import re
from PIL import Image
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from CARDSet.dataset import CARDSetDataset


def get_camera_and_ground_info(img_file, root_dir, depth_name):
    """
    Extract camera parameters and ground normal vector from an image file.

    Args:
        img_file (str): Path to the image file.
        root_dir (str): Root directory of the dataset.

    Returns:
        dict: A dictionary containing:
            - 'intrinsics': Camera intrinsic matrix (3, 3)
            - 'extrinsics': Camera extrinsic matrix (4, 4)
            - 'ground_normal': Ground plane normal vector (3,)
            - 'ground_point': A point on the ground plane (3,)
            - 'camera_height': Height of camera above ground (float)
            - 'timestamp_us': Image timestamp in microseconds (int)
            - 'image': Image as tensor (3, H, W)
    """
 
    
    # Helper functions (from the dataset class)
    def pose_to_matrix(x, y, z, qw, qx, qy, qz):
        """Convert pose (position + quaternion) to 4x4 transformation matrix."""
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def interpolate_pose(traj, ts_us):
        """Interpolate pose at a given timestamp."""
        poses = traj["trajectory_poses"]
        i = 0
        while i < len(poses) and poses[i][0] <= ts_us:
            i += 1
        if i == 0:
            return poses[0][1]
        if i >= len(poses):
            return poses[-1][1]
        ts0, p0 = poses[i-1]
        ts1, p1 = poses[i]
        t = float(np.clip((ts_us - ts0) / max(1, (ts1 - ts0)), 0.0, 1.0))
        trans = (1.0 - t) * np.array(p0[:3]) + t * np.array(p1[:3])
        
        # SLERP for quaternion
        from scipy.spatial.transform import Slerp
        q0_xyzw = np.array([p0[4], p0[5], p0[6], p0[3]])
        q1_xyzw = np.array([p1[4], p1[5], p1[6], p1[3]])
        Rends = Rotation.from_quat(np.stack([q0_xyzw, q1_xyzw], 0))
        s = Slerp([0.0, 1.0], Rends)
        q_xyzw = s([t]).as_quat()[0]
        quat = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]
        
        return [*trans.astype(float).tolist(), *quat]

    def _seq_root_from_rel_CARD_Reconstruction(rel):
        """Extract sequence root and relative path from image file path."""
        parts = rel.replace("\\", "/").split("/img/")
        rel_path = Path(parts[1]) if len(parts) > 1 else None
        return Path(parts[0]), rel_path

    def _parse_cam_ts(rel):
        """Parse camera name and timestamp from image filename."""
        import re
        name = Path(rel).name
        m = re.match(r"(cam_\d+)_(\d+)\.(?:jpg|jpeg|png)$", name)
        if not m:
            return None, None
        return m.group(1), int(m.group(2))

    def _get_traj(seq_root):
        """Load trajectory JSON file."""
        traj_path = seq_root / "export" / "output.laz.trajectory.json"
        with open(traj_path, "r") as f:
            return json.load(f)

    def _K_dist_from_traj(traj, cam):
        """Extract intrinsic matrix and distortion coefficients from trajectory."""
        ci = traj["camera_infos"][cam]
        fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
        dist = np.array(ci.get("dist_coeffs", [0, 0, 0, 0, 0]), np.float32)
        return K, dist

    def _ts_from_rel_CARD_Reconstruction(rel):
        """Extract timestamp from relative path."""
        _, ts = _parse_cam_ts(rel)
        return ts

    def _T_cam_to_rig_from_traj(traj, cam):
        """Get transformation from camera to rig."""
        x, y, z, qw, qx, qy, qz = traj["sensor_to_trajectory_poses"][cam]
        return pose_to_matrix(x, y, z, qw, qx, qy, qz)

    def _T_cam_world(traj, cam, ts_img_us):
        """Get transformation from camera to world."""
        ci = traj["camera_infos"][cam]
        ts = ts_img_us + ci.get("timestamp_offset", 0)
        x, y, z, qw, qx, qy, qz = interpolate_pose(traj, ts)
        T_rig_world = pose_to_matrix(x, y, z, qw, qx, qy, qz)
        T_cam_rig = _T_cam_to_rig_from_traj(traj, cam)
        return T_rig_world @ T_cam_rig

    def get_img(path):
        """Load image as numpy array."""
        from PIL import Image
        im = Image.open(str(path)).convert("RGB")
        return np.array(im)

    def find_neighbour_index(clist, timestamp_us, offsets=(-1, 1)):
        """Find indices of neighbor frames."""
        neighbour_index = {}
        
        # Find current index
        current_idx = None
        for idx, (ts, _) in enumerate(clist):
            if ts == timestamp_us:
                current_idx = idx
                break
        
        if current_idx is None:
            return {offset: -1 for offset in offsets}
        
        # Find neighbor indices based on offsets
        for offset in offsets:
            neighbor_idx = current_idx + offset
            if 0 <= neighbor_idx < len(clist):
                neighbour_index[offset] = neighbor_idx
            else:
                neighbour_index[offset] = -1
        
        return neighbour_index
    
    def compute_relative_extrinsics(extrinsic_current, extrinsic_neighbor):
        """
        Compute the relative extrinsic transformation between two camera poses.
        """
        # Convert to numpy if needed
        if isinstance(extrinsic_current, torch.Tensor):
            extrinsic_current = extrinsic_current.cpu().numpy()
        if isinstance(extrinsic_neighbor, torch.Tensor):
            extrinsic_neighbor = extrinsic_neighbor.cpu().numpy()
        
        # Ensure they are float32
        extrinsic_current = extrinsic_current.astype(np.float32)
        extrinsic_neighbor = extrinsic_neighbor.astype(np.float32)
        
        # Compute inverse of current extrinsic
        extrinsic_current_inv = np.linalg.inv(extrinsic_current)
        
        # Compute relative transformation
        relative_extrinsic = extrinsic_neighbor @ extrinsic_current_inv
        relative_extrinsic_inv = np.linalg.inv(relative_extrinsic)
        
        return relative_extrinsic, relative_extrinsic_inv
    
    def _cam_dir_list(seq_root, cam_name):
        """Get list of image files for a camera."""
        img_dir = seq_root / "img" / cam_name
        image_files = sorted(img_dir.glob("*.jpg"), key=lambda x: int(re.search(r'_(\d+)', x.name).group(1)))
        
        clist = []
        for img_path in image_files:
            _, ts = _parse_cam_ts(img_path.name)
            clist.append((ts, img_path))
        return clist
    
    def get_neighbour_frames_with_relative_extrinsics(traj, seq_root, cam_name, timestamp_us, offsets=(-1, 1)):
        """Get neighbor frames with relative extrinsics."""
        frames = {}
        clist = _cam_dir_list(seq_root, cam_name)
        
        # Get current frame extrinsic
        current_extrinsic = _T_cam_world(traj, cam_name, timestamp_us)
        
        # Find neighbor indices
        neighbour_index = find_neighbour_index(clist, timestamp_us, offsets)
        
        for offset, ts_idx in neighbour_index.items():
            if ts_idx == -1:  # Invalid index
                continue
                
            ts_n, path_n = clist[ts_idx]
            rgb_n = Image.open(path_n)
            points = np.load(str(path_n).replace("img", "agg_depth").replace(".jpg", ".npz"))['pts_cam']
            rgb_n = torch.from_numpy(np.array(rgb_n)).permute(2, 0, 1).float()
            rgb_n = rgb_n[[2, 1, 0], :, :]
            extrinsic_n = _T_cam_world(traj, cam_name, ts_n)
            
            # Compute relative extrinsics
            relative_extrinsic, relative_extrinsic_inv = compute_relative_extrinsics(
                current_extrinsic, extrinsic_n
            )
            
            gt_height =  get_gt_elevation()
            frames[offset] = {
                "timestamp_us": ts_n,
                "path": str(path_n),
                "extrinsic": extrinsic_n,
                "relative_extrinsic": relative_extrinsic,
                "relative_extrinsic_inv": relative_extrinsic_inv,
                "rgb": rgb_n,
                "depth": points,
            }
        
        return frames
    
    def _compute_road_plane_from_wheels(traj, timestamp_us):
        """Compute road plane from wheel positions."""
        wheel_keys = [
            "cariad_wheel_FL_ground", "cariad_wheel_FR_ground",
            "cariad_wheel_RL_ground", "cariad_wheel_RR_ground"
        ]
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
            p = np.array([vp[0], vp[1], vp[2] - 1.8], dtype=np.float32)
            return (np.array([0, 0, 1], dtype=np.float32), p, np.array([p]))

        pts = np.array(points, dtype=np.float32)
        ctr = np.mean(pts, axis=0)
        _, _, vh = np.linalg.svd(pts - ctr)
        norm = vh[-1]
        if norm[2] < 0:
            norm = -norm
        return (norm, ctr, pts)

    def _compute_camera_height_from_ground_plane(traj, cam, ts, norm, pt):
        """Compute camera height above ground plane."""
        Tcw = _T_cam_world(traj, cam, ts)
        cam_pt = Tcw[:3, 3]
        return float(np.dot(cam_pt - pt, norm / np.linalg.norm(norm)))

    def get_depth(path):
        data = np.load(path)['pts_cam']
        return data
    # Main processing
    path_of_the_sequence, relative_path = _seq_root_from_rel_CARD_Reconstruction(os.path.join(root_dir, img_file))
    traj = _get_traj(path_of_the_sequence)

    # Parse image file
    img = get_img(os.path.join(root_dir, img_file))
    pts = get_depth(os.path.join(root_dir, depth_name))
    
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    img_tensor = img_tensor[[2, 1, 0], :, :]
    cv2.imwrite("current_img.png", img_tensor.permute(1, 2, 0).byte().numpy())
    # Extract timestamp and camera name
    ts = _ts_from_rel_CARD_Reconstruction(str(relative_path))
    cam_name, _ = _parse_cam_ts(img_file)

    # Get camera parameters
    intrinsic, dist_coeffs = _K_dist_from_traj(traj, cam_name)
    extrinsic = _T_cam_world(traj, cam_name, ts)

    # Compute ground plane
    norm, pt, wheel_pts = _compute_road_plane_from_wheels(traj, ts)
    camera_height = _compute_camera_height_from_ground_plane(traj, cam_name, ts, norm, pt)
    depth = _project_pts_to_depth(pts, intrinsic, img.shape)
    # Get neighbor frames with relative extrinsics
    neighbours = get_neighbour_frames_with_relative_extrinsics(
        traj, path_of_the_sequence, cam_name, ts, offsets=(-1, 1)
    )

    gt_height = get_gt_elevation(pts)
    # Return results
    return {
        'intrinsics': torch.Tensor(intrinsic),
        'extrinsics': torch.Tensor(extrinsic),
        'ground_normal': torch.Tensor(norm),
        'ground_point': torch.Tensor(pt),
        'wheel_points': torch.Tensor(wheel_pts),
        'camera_height': camera_height,
        'timestamp_us': ts,
        'image': img_tensor,
        'depth': depth,
        'distortion_coeffs': torch.Tensor(dist_coeffs),
        'neighbours': neighbours,
        'gt_height': gt_height,
    }

def _project_pts_to_depth(pts_cam: np.ndarray, K: np.ndarray, HW: tuple[int,int]) -> np.ndarray:
    H,W = HW; dep = np.zeros((H,W), np.float32)
    if pts_cam is None or len(pts_cam)==0: return dep
    z = pts_cam[:,2]; ok = z > 0
    if not np.any(ok): return dep
    x,y,z = pts_cam[ok,0], pts_cam[ok,1], pts_cam[ok,2]
    u = (K[0,0]*x/z) + K[0,2]; v = (K[1,1]*y/z) + K[1,2]
    m = (u>=0)&(u<W)&(v>=0)&(v<H)
    if not np.any(m): return dep
    u,v,z = u[m], v[m], z[m]
    order = np.argsort(z)[::-1]
    u,v,z = u[order], v[order], z[order]
    dep[v.astype(int), u.astype(int)] = z
    return dep

def _get_height_(ele) -> np.ndarray:
    
    world_points = torch.stack([
        x_world.reshape(-1),
        ele.reshape(-1),
        z_world.reshape(-1)
    ], dim=1) #(N, 3)

def project_height(height, extrinsic, HW):
    H, W = HW
    t = extrinsic[3,:3]
    R = extrinsic[:3, :3]
    cam_points = (R @ height.T + t).T
    points_2d = (K @ cam_points.T).T  # (N, 3)
    z = cam_points[:, 2]
    u = (points_2d[:, 0] / z).long()
    v = (points_2d[:, 1] / z).long()
    
    # Fill
    height_map = torch.zeros(H, W)
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    height_map[v[valid], u[valid]] = height[1][valid]

def get_gt_elevation(xyz):
        #transform in world coordinate
        xyz = np.asarray(xyz.points)


        points_y = xyz[:, 1]*100  # points, m --> cm
        points_xz = xyz[:, [0, 2]]
        grids_y = torch.zeros((num_grids_z, num_grids_x), dtype=torch.float32)
        grids_count = torch.zeros((num_grids_z, num_grids_x), dtype=torch.int8)

        for xz, y in zip(points_xz, points_y):
            idx_x = torch.clip(((xz[0] - roi_x[0]) / grid_res[0]).int(), max=num_grids_x-1)
            idx_z = torch.clip(num_grids_z - 1 - ((xz[1] - roi_z[0]) / grid_res[2]).int(), min=0)
            grids_y[idx_z, idx_x] += y
            grids_count[idx_z, idx_x] += 1
        mask = grids_count > 0

        grids_y[mask] = 1.67*100 - grids_y[mask] / grids_count[mask]

        return grids_y, mask

def compute_target_stats(train_loader):
    all_targets = []
    max = 0
    
    for batch in train_loader:
        (imgs_left, target, ele_mask, proj_index_left,_) = batch
        elerange = 20
        roi = torch.logical_and(target < elerange , target > -elerange)
        ele_mask = torch.logical_and(ele_mask, roi)
        print("number of remaining values:", target[ele_mask].numel())
        all_targets.append(target[ele_mask].flatten())
        
    
    all_targets = torch.cat(all_targets)
    all_targets = all_targets.numpy()
    
    mean = all_targets.mean()
    std  = all_targets.std()
    
    print(f"mean:  {mean:.4f}")
    print(f"std:   {std:.4f}")
    print(f"min:   {all_targets.min():.4f}")
    print(f"max:   {all_targets.max():.4f}")
    print(f"p1:    {np.quantile(all_targets, 0.01):.4f}")
    print(f"p99:   {np.quantile(all_targets, 0.99):.4f}")
    # Naive baseline: predict dataset mean for every pixel
    naive_l1 = np.mean(np.abs(all_targets - mean))
    print(f"naive baseline L1 (predict mean): {naive_l1:.4f}")

    print("normalize: meand and deviation", mean, std)
    return mean, std



if __name__ == "__main__":

    """ y_range = 0.2 
    roi_x = torch.tensor([-1, 0.92])    # in meter, the lateral range of interest (in the horizontal coordinate of camera)
    roi_z = torch.tensor([2.16, 7.08])    # in meter, the longitudinal range of interest

    grid_res = torch.tensor([0.03, 0.01, 0.03])  # in [x, y(vertical), z] order. The range of interest above should be integer times of resolution here
    

    num_grids_x = int((roi_x[1] - roi_x[0]) / grid_res[0])
    num_grids_z = int((roi_z[1] - roi_z[0]) / grid_res[2])
    num_grids_y = int(y_range*2 / grid_res[1])
    x_world = (torch.arange(num_grids_x) * grid_res[0] + roi_x[0] + grid_res[0]/2).unsqueeze(0).repeat([num_grids_z, 1])
    z_world = (-torch.arange(num_grids_z) * grid_res[2] + roi_z[1] - grid_res[2]/2).unsqueeze(1).repeat([1, num_grids_x])
 

    root_dir = '/data/T7/cariad dataset'
    image_name = "germany_2days/germany_5_sw/img/cam_1/cam_1_126301867.jpg"
    depth_name = "germany_2days/germany_5_sw/agg_depth/cam_1/cam_1_126301867.npz"
    camera_info = get_camera_and_ground_info(image_name, root_dir, depth_name)
    camera_previous = camera_info['neighbours'][-1]
    homography = HomographyWarp(2252, 2248)
    reprojection_loss = ReprojectionLoss((2252, 2248))

    d_scalar = camera_info['camera_height']
    n = camera_info['ground_normal']
    R = camera_previous['relative_extrinsic'][:3, :3]
    t = camera_previous['relative_extrinsic'][:3, 3]
    K = camera_info['intrinsics']
    inv_K = torch.linalg.inv(K)

       # Add batch dimension
    d = torch.tensor([d_scalar]).unsqueeze(0)  # (1, 1, 1) - scalar in batch
    print("d", d)
    n = n.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
    R = torch.tensor(R).unsqueeze(0)  # (1, 3, 3)
    t = torch.tensor(t).unsqueeze(0).unsqueeze(-1) #(3,)
    n = (n @ camera_previous['extrinsic'][:3, :3])  # Transform t to current camera frame
    camera_extrinsic = torch.tensor(camera_info['extrinsics'])
    camera_previous_extrinsic = torch.tensor(camera_previous['extrinsic'])
    K = K.unsqueeze(0)  # (1, 3, 3)
    inv_K = torch.linalg.inv(K)  # (1, 3, 3)
    image_previous = camera_previous['rgb'].unsqueeze(0)

    cv2.imwrite("previous_img.png", image_previous.squeeze(0).permute(1, 2, 0).numpy())


    print(f"d shape: {d.shape}")
    print(f"n shape: {n.shape}")
    print(f"R shape: {R.shape}")
    print(f"t shape: {t.shape}")
    print(f"K shape: {K.shape}")
    print(f"inv_K shape: {inv_K.shape}")

    pix_coords, padding_mask, _ = homography(d, n, R, t, K, inv_K)
    
    print(pix_coords.shape, padding_mask.shape)

    target_warped = F.grid_sample(image_previous, pix_coords, padding_mode="border", align_corners=True)

    print(target_warped.shape)
    warped = target_warped * padding_mask.squeeze(1)

    #########pp_flow
    epipoles, _ = reprojection_loss.get_epipoles(K, camera_previous_extrinsic, camera_extrinsic)

    #get the ground_truth_depth and height from the gt pointcloud
    ground_truth_depth = camera_info['depth']
    print("g depth shape", ground_truth_depth.shape)
    gt_height = camera_info['gt_height']
    height = _get_height_(gt_height)
    
    
    print("height shape: ", height.shape)
    gt_height_pixels = project_height(height, camera_previous_extrinsic, (2252, 2248))
    gt__height = gt_height_pixels
    pp_flow = reprojection_loss.get_pp_flow(epipoles, R, t, K, gt__height, ground_truth_depth, d)

    cv2.imwrite("warped_image.png", target_warped.squeeze(0).permute(1, 2, 0).byte().numpy()) """
    # test_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', split_file='/data/T7/cariad dataset/val_all_data_clean_NN_RHF.txt', mode='val', down_scale=2, preprocessed_data = True)

    # train_loader = DataLoader(test_set, 1, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    
    # compute_target_stats(train_loader)

    w1 = torch.load("patch_embed_proj.pt")
    w2 = torch.load("patch_embed_proj_da3.pt")
    print(w1, "/w2",w2)
    all_equal = True

    for k in w1:
        if not torch.equal(w1[k], w2[k]):
            print(f"❌ Different weight: {k}")
            all_equal = False
            diff = (w1[k] - w2[k]).abs().max()
            print(k, diff)

    if all_equal:
        print("✅ All weights are identical!")
from tqdm import tqdm
import argparse
from models.model import Elevation
from utils.dataset import RSRD
import torch
import pickle
import numpy as np
import open3d as o3d
import os


def get_item(index):
    sample_cur = dataset.data_all[index]
    l2c_calib_cur = dataset.get_lidar2cam(sample_cur['time'])
    path_base = sample_cur['path']
    idx_str = path_base.find('/')
    path_base = path_base[idx_str + 1:]

    R_cur2enu = dataset.get_RT_lidar(sample_cur)

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    viz = o3d.visualization.O3DVisualizer("Scene", 1280, 720)
    
    lidar_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    R = coord_frame.get_rotation_matrix_from_xyz((0.0, 0.0, np.pi))  # (Rx, Ry, Rz) in radians

    coord_frame = coord_frame.rotate(R, center=(0, 0, 0))
    ########   calculate the euler angles of the camera (relative to local ENU coord)   ########
    [pitch_cam, roll_cam, yaw_cam] = dataset.matrix2euler(l2c_calib_cur['R'] @ np.linalg.inv(R_cur2enu))
    pitch_cam -= 1.5708  # pi/2
    R_X = np.array(
        [[1, 0, 0], [0, np.cos(pitch_cam), np.sin(pitch_cam)], [0, -np.sin(pitch_cam), np.cos(pitch_cam)]],
        dtype=np.float32)
    R_Z = np.array(
        [[np.cos(roll_cam), np.sin(roll_cam), 0], [-np.sin(roll_cam), np.cos(roll_cam), 0], [0, 0, 1]],
        dtype=np.float32)
    R_cam2vert = R_X @ R_Z  # the rotation matrix from the current camera coord to the vertical status

    ########  read point cloud and transform into camera's coord, then crop the ROI  #######
    path_pcd = os.path.join(dataset.data_path, path_base, 'pcd', sample_cur['time']) + '.pcd'
    cloud = o3d.io.read_point_cloud(path_pcd)
    cloud = cloud.rotate(l2c_calib_cur['R'], center=(0, 0, 0))
    cloud = cloud.translate(tuple(l2c_calib_cur['T'].reshape(-1)))  # the point cloud in the camera's coord
    points = np.asarray(cloud.points)
    print(f"Height before values (y): min={points[:, 1].min()}, max={points[:, 1].max()}")

    import copy
    camera = copy.deepcopy(lidar_frame)
    camera = camera.rotate(l2c_calib_cur['R'], center=(0, 0, 0))
    camera = camera.translate(tuple(l2c_calib_cur['T'].reshape(-1)))
    print("lidar_camera", l2c_calib_cur['R'])
    print("lidar_enu", R_cur2enu)
    print("translation lidar_camera", l2c_calib_cur['T'])
    enu = copy.deepcopy(lidar_frame)
    enu.rotate(R_cur2enu, center=(0, 0 , 0))
    #road_frame.translate((0, -1.73, 0))
    roadZ = copy.deepcopy(camera)
    roadZ = roadZ.rotate(R_Z, center=(0, 0, 0))

    roadX = copy.deepcopy(roadZ)
    roadX = roadX.rotate(R_X, center=(0, 0, 0))
    print(R_cam2vert)
    #points = np.asarray(.points)
    #print(f"Height values (y): min={points[:, 1].min()}, max={points[:, 1].max()}")

    viz.add_geometry("lidar", lidar_frame)
    viz.add_geometry("road_frameZ", roadZ)
    viz.add_geometry("road_frameX", roadX)
    viz.add_geometry("camera", camera)
    viz.add_geometry("enu", enu)

    viz.show_settings = True
    app.add_window(viz)
    #app.run()
    #o3d.visualization.draw_geometries([road_frame, mou])

    # crop the point cloud according to the given range of interest
    cloud_camvert = cloud.rotate(R_cam2vert, center=(0, 0, 0))
    points = np.asarray(cloud_camvert.points)
    print(f"Height after alignment values (y): min={points[:, 1].min()}, max={points[:, 1].max()}")

    cloud_camvert = vol_roi.crop_point_cloud(cloud_camvert)

    points = np.asarray(cloud_camvert.points)
    print(f"Height after cropping values (y): min={points[:, 1].min()}, max={points[:, 1].max()}")
    if(index == 0):
        print("check cm or m", np.asarray(cloud_camvert.points[1]))

    ele_gt, ele_mask = dataset.get_gt_elevation(cloud_camvert)

    return ele_gt, ele_mask, sample_cur['time']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RoadBEV: Road Surface Reconstruction in Bird\'s Eye View')
    parser.add_argument('--save_dir', type=str, required=True, help='save path for preprocessed GT maps')
    parser.add_argument('--dataset', type=str, required=True, choices=['train', 'test'], help='generating for train or test sets')
    args = parser.parse_args()

    training = args.dataset == 'train'
    dataset = RSRD(training=training)
    if training:
        path = os.path.join(args.save_dir, 'train')
    else:
        path = os.path.join(args.save_dir, 'test')
    os.makedirs(path, exist_ok=True)

    # parameters for cropping ROI point clouds
    crop_bounding = np.array([[dataset.roi_x[0], 0, dataset.roi_z[0]],
                                   [dataset.roi_x[0], 0, dataset.roi_z[1]],
                                   [dataset.roi_x[1], 0, dataset.roi_z[1]],
                                   [dataset.roi_x[1], 0, dataset.roi_z[0]]]).astype("float64")
    vol_roi = o3d.visualization.SelectionPolygonVolume()
    vol_roi.orthogonal_axis = "Y"
    vol_roi.axis_max = 1.5
    vol_roi.axis_min = 0.5
    vol_roi.bounding_polygon = o3d.utility.Vector3dVector(crop_bounding)

    for i in tqdm(range(len(dataset))):
        ele_gt, ele_mask, stamp = get_item(i)
        with open(os.path.join(path, stamp + '.pkl'), 'wb') as f:
            pickle.dump([ele_gt, ele_mask], f)
        print(stamp)

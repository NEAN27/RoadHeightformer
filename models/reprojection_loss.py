import numpy as np
import torch
from matplotlib.path import Path
from torch import nn
import torch.nn.functional as F

class ReprojectionLoss(torch.nn.Module):
    def __init__(self, image_shape):
        super(ReprojectionLoss, self).__init__()
        self.image_shape = image_shape  # Example image shape (height, width)
        self.homography = HomographyWarp(image_shape[0], image_shape[1])


    def project_grid_to_pixel(hori_centers, K, R_vert2cam, camera_height):
        """
        Project the corners of a grid onto the pixel coordinate space.
        Assumption: The world coordinate system origin is at the camera center, with the y-axis pointing upwards, the x-axis pointing to the right, and the z-axis pointing forward.
        Args:
            hori_centers (torch.Tensor): Horizontal centers of the grid (N, 2), where each row is [x, z].
            K (torch.Tensor): Intrinsic matrix of the camera (3, 3).
            R_vert2cam (torch.Tensor): Rotation matrix from vertical to camera coordinates (3, 3).
            camera_height (float): Height of the camera above the ground.

        Returns:
            torch.Tensor: Pixel coordinates of the 4 corners of the grid (4, 2).
        """
        # Ensure inputs are tensors
        if not isinstance(hori_centers, torch.Tensor):
            hori_centers = torch.tensor(hori_centers, dtype=torch.float32)
        if not isinstance(K, torch.Tensor):
            K = torch.tensor(K, dtype=torch.float32)
        if not isinstance(R_vert2cam, torch.Tensor):
            R_vert2cam = torch.tensor(R_vert2cam, dtype=torch.float32)

        # Add the y-coordinate (camera height) to the horizontal centers
        num_points = hori_centers.shape[0]
        grid_3d = torch.zeros((num_points, 3), dtype=torch.float32)
        grid_3d[:, 0] = hori_centers[:, 0]  # x-coordinates
        grid_3d[:, 1] = camera_height       # y-coordinate (camera height)
        grid_3d[:, 2] = hori_centers[:, 1]  # z-coordinates

        # Transform the grid points to the camera coordinate system
        grid_cam = (R_vert2cam @ grid_3d.T).T  # Shape: (N, 3)

        # Project the 3D points to the 2D pixel space
        uvz = (K @ grid_cam.T).T  # Shape: (N, 3)
        uv = uvz[:, :2] / uvz[:, 2:]  # Normalize by depth (z), Shape: (N, 2)

        # Extract the 4 corners of the grid in pixel coordinates
        top_left = uv[0]
        top_right = uv[1]
        bottom_right = uv[2]
        bottom_left = uv[3]

        # Combine the corners into a single tensor
        corners_pixel = torch.stack([top_left, top_right, bottom_right, bottom_left], dim=0)

        return corners_pixel

    def get_coordinates_from_corners(corners_pixel, image_shape):
        """
        Get the coordinates of all the pixel in pixel in the region defined by the corners.
        method: return a tensor of shape (B, H, W) where every pixel in the polygone is 1 <nd the rest is 0. Then we can use this mask to compute the loss only on the pixel in the region of interest.
        
        """

        mask = torch.zeros(image_shape, dtype=torch.bool)
        #ensure the input is a tensor
        if not isinstance(corners_pixel, torch.Tensor):
            corners_pixel = torch.tensor(corners_pixel, dtype=torch.float32)
            #Get the bounding box of the corners
            u_min = max(int(torch.min(corners_pixel[:, 0]).item()), 0)
            u_max = min(int(torch.max(corners_pixel[:, 0]).item()), image_shape[1] - 1)
            v_min = max(int(torch.min(corners_pixel[:, 1]).item()), 0)
            v_max = min(int(torch.max(corners_pixel[:, 1]).item()), image_shape[0] - 1)

            # Create a grid of pixel coordinates
        u_coords, v_coords = torch.meshgrid(
            torch.arange(u_min, u_max + 1),
            torch.arange(v_min, v_max + 1),
            indexing="xy"
        )
        pixel_coords = torch.stack([u_coords.flatten(), v_coords.flatten()], dim=1)  # Shape: (N, 2)

        # Check if each pixel is inside the polygon defined by the corners
        polygon = Path(corners_pixel.numpy())  # Convert corners to a Path object
        inside = polygon.contains_points(pixel_coords.numpy())  # Check if pixels are inside the polygon

        inside = torch.tensor(inside)
        mask[pixel_coords[inside][:, 1],
              pixel_coords[inside][:, 0]] = True  # Set the mask for pixels inside the polygon


        return mask

    def get_epipoles(self, K, ext_0, ext_1):
        """
        Get the epipoles in the image plane for two cameras.
        Args:
            K (torch.Tensor): Intrinsic matrix of the camera (B, 3, 3) camera to world matrice.
            ext_0 (torch.Tensor): Extrinsic parameters of camera 0 (B, 4, 4).
            ext_1 (torch.Tensor): Extrinsic parameters of camera 1 (B, 4, 4).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Epipole coordinates in the image plane for cameras (B, 2) for both cameras.
        """
        K = K.detach().clone()
        ext_0 = ext_0.detach().clone()
        ext_1 = ext_1.detach().clone()

        # Extract translations
        t0 = ext_0[..., :3, 3]  # (B, 3)
        t1 = ext_1[..., :3, 3]  # (B, 3)

        # Relative translation (from frame 0 to frame 1)
        t_rel = t1 - t0  # (B, 3)

        epipole_0 = (K @ (-t_rel).unsqueeze(-1)).squeeze(-1)  # Epipole in camera 0's image plane
        epipole_1 = (K @ t_rel.unsqueeze(-1)) .squeeze(-1) # Epipole in camera 1's image plane

        epipole_0 = epipole_0[:, :2] / (epipole_0[:, 2:] + 1e-8) # Normalize by depth (z)
        epipole_1 = epipole_1[:, :2] / (epipole_1[:, 2:] + 1e-8) # Normalize by depth (z)

        return epipole_0, epipole_1
    
    def compute_flow_to_epipole(epipoles, image_shape):
        """
        Compute the flow from each pixel to the epipole.
        Args:
            epipoles (torch.Tensor): Epipole coordinates in the image plane (2,).
            pixel_coords (torch.Tensor): Pixel coordinates in the image plane (N, 2).

        """
        B = epipoles.shape[0]

        #Create pixel grid
        grid_x, grid_y = torch.meshgrid(
            torch.arange(image_shape[0]),
            torch.arange(image_shape[1]),
            indexing = 'ij'
        )
        grid = torch.stack([grid_x, grid_y], dim= 0).float() #(2, H, W)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1) #(B, 2, H, W)

        #expand epipole to match the grid shape
        epipoles = epipoles.view(B, 2, 1, 1).expand(-1, -1, image_shape[0], image_shape[1]) #(B, 2, H, W)

        #compute the flow for eac pixel
        flow = epipoles - grid 

        return flow
    
    def get_relative_from_extrinsics(ext_0, ext_1):
        """
        Get the relative rotation and translation from the extrinsic parameters of two cameras.
        Args:
            ext_0 (torch.Tensor): Extrinsic parameters of camera 0 (B, 4, 4).
            ext_1 (torch.Tensor): Extrinsic parameters of camera 1 (B, 4, 4).
        Returns:
            Relative rotation and translation, also inverse of the relative rotation and translation
        """
        R0 = ext_0[..., :3, :3]  # (B, 3, 3)
        t0 = ext_0[..., :3, 3]    # (B, 3)
        R1 = ext_1[..., :3, :3]  # (B, 3, 3)
        t1 = ext_1[..., :3, 3]    # (B, 3)

        # Relative rotation and translation from camera 0 to camera 1
        R_rel = R1 @ R0.transpose(-2, -1)  # Relative rotation
        t_rel = t1 - (R_rel @ t0)           # Relative translation

        # Inverse of the relative rotation and translation (from camera 1 to camera 0)
        R_rel_inv = R_rel.transpose(-2, -1) # Inverse rotation
        t_rel_inv = -(R_rel_inv @ t_rel)    # Inverse translation

        return R_rel, t_rel, R_rel_inv, t_rel_inv
    
    
    def get_pp_flow(self, epipoles, R_rel, t_rel, K, Height_pred, ground_truth_depth, camera_height):
        """
        compute the residual parralax flow for each pixel
        Args:
        """
        flow_to_epipole = self.compute_flow_to_epipole(epipoles, self.image_shape) #(B, 2, H, W)
        fct = t_rel/camera_height      #translation along the z axis/camera_height
        gamma = Height_pred / ground_truth_depth #the ratio between the predicted height and the ground truth depth
        #ToDo: expand the fct and the gamma to match the shape of the flow
        gamma_fct = gamma * fct

        pp_flow = ((-gamma_fct)/(1 - gamma_fct + 1e-5)) * flow_to_epipole

        return pp_flow #[B, 2, H, W]

    def generate_images_pred_flow(self, input_image, pp_flow):
        """
        Generate the predicted pixel coordinates by adding the parralax flow to the original pixel coordinates.
        Args:
    """
        u, v = torch.split(pp_flow, 1, dim=1)
        y, x = torch.meshgrid([torch.arrange(0, self.image_shape[0]), torch.arrange(0, self.image_shape[1])])
        x_t = x - u
        y_t = y - v
        pix_coords = torch.stack([x_t, y_t], dim= 0)
        pix_coords= pix_coords.squeeze(2).permute(1,2,3,0) #this is to reshape such that it should be in the shape ([3, 192, 640, 2])

        pix_coords[..., 0] /= self.image_shape[1] - 1
        pix_coords[..., 1] /= self.image_shape[0] - 1
        pix_coords = (pix_coords - 0.5) *2
        warped_image = F.grid_sample(input_image, pix_coords, padding_mode="border", align_corners=True)


        return warped_image
    
    def generate_warped_from_homography(self, input_image, normal_vector, height, R, t, K, inv_K):
        """
        Generate image warped by homography, The homography is generated by the normal vector and height of the plane, as well as the relative pose between the two cameras.
        
        """
        pix_coords, padding_mask, H_s2t = self.homography(height, normal_vector, R, t, K, inv_K)
        warped_image = F.grid_sample(input_image, pix_coords, padding_mode="border", align_corners=True)
        return warped_image, padding_mask, H_s2t

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    

    def forward(self, height_pred, gt_pcd, I, I_previous, I_next, ground_info, extrinsics_t, extrinsics_next, extrinsics_previous, K):
        """" compute the warped image the the loss between the warped image and the input image.
        Args:
            height_pred: the predicted height of the plane (B, H_grid, W_grid)
            gt_pcd: the ground truth point cloud (B, N, 3)
            I, I_previous, I_next: the input images (B, C, H, W)
            ground_info: the information of the ground plane, including the normal vector and the height of the camera above the ground
            extrinsics_t, extrinsics_next, extrinsics_previous: the extrinsic parameters of the current frame, the next frame and the previous frame (B, 4, 4)
            K: the intrinsic matrix of the camera (B, 3, 3)
        Return: a loss value per grid prediction
        """
        inv_K = torch.inverse(K)   #shape (B, 3, 3)
        gt_depth = gt_pcd[..., 2]  #shape (B, N) N is the number of points in the croped point cloud
        R_rel, t_rel, _, _ = self.get_relative_from_extrinsics(extrinsics_previous, extrinsics_t)   #shape (B, 3, 3), (B, 3)
        epipole_previous, epipole_t = self.get_epipoles(K, extrinsics_previous, extrinsics_t)       #shape (B, 2), (B, 2)
        pp_flow = self.get_pp_flow(epipole_previous, R_rel, t_rel, K, height_pred, gt_depth, ground_info['camera_height_above_ground'])     #shape (B, 2, H, W)
        warped_homography, padding_mask, H_s2t = self.generate_warped_from_homography(I_previous, ground_info['plane_normal_world'], ground_info['camera_height_above_ground'], R_rel, t_rel, K, inv_K)
        warped_flow = self.generate_images_pred_flow(warped_homography, pp_flow)

        loss = self.compute_reprojection_loss(warped_flow, I)
        print("reprojection loss: ", loss.shape)

        return loss



class HomographyWarp(nn.Module):
    """Layer to generate the homography by pose
    """
    def __init__(self, height, width):
        super(HomographyWarp, self).__init__()

        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False).cuda()

        self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
                                 requires_grad=False).cuda()

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, d, n, R, t, K, inv_K):
        """
        d: B, N [batch_size, Number of heights]
        n: B, N, 3 [batch_size, number of heights, 3 (the normal vector ;) ]
        # d --> negative , n = [0,1,0], Tz --> positive
        """
        print(d, t, n, K)
        B, N = d.shape
        d = d.reshape(B*N, 1, 1)
        n = n.reshape(B*N, 1, 3)
        pix_coords_t = self.pix_coords.expand(B*N, -1, -1).cpu()
        Rtnd = R + torch.matmul(t, n) / d
        # print(K[0, :3, :3], inv_K[0, :3, :3])
        H_s2t = torch.matmul(K[:, :3, :3], torch.matmul(Rtnd, inv_K[:, :3, :3]))
        #H_t2s = torch.inverse(H_s2t +1e-7)

        pix_coords = torch.matmul(H_s2t, pix_coords_t)
        
        padding_mask = (torch.matmul(inv_K[:, :3, :3], pix_coords_t) * torch.matmul(R, n[:, 0, :, None])).sum(1) > 0.
        z = pix_coords[:, 2:3, :]
        padding_mask = padding_mask * (z[:, 0] > 1e-7)
        padding_mask = padding_mask.reshape(B, N, 1, self.height, self.width)
        z[z < 1e-7] = 1e-7
        pix_coords = pix_coords[:, :2, :] / z
        pix_coords = pix_coords.view(B*N, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords, padding_mask, H_s2t


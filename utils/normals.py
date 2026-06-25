import torch
import torch.nn.functional as F

def get_3d_points_from_height(height, hori_centers):
    """
    height:       [B, H, W]  - elevation map (metric) y
    hori_centers: [H, W, 2]  - (x, z) horizontal coordinates for each pixel
    returns:      [B, H, W, 3]
    """
    x = hori_centers[None, :, :, 0].expand(height.shape[0], -1, -1)             
    z = hori_centers[None, :, :, 1].expand(height.shape[0], -1, -1)             
    y = height                                                                    
    print(x.device)
    print(y.device)
    return torch.stack([x, y, z], dim=-1)                

def compute_normals(points):
    """
    points: [B, H, W, 3]
    returns normals: [B, H, W, 3]
    """
    p0 = points[:, 1:-1, 1:-1, :]                               
    p1 = points[:, 1:-1, 2:,   :]          
    p2 = points[:, :-2,  1:-1, :]       
    p3 = points[:, 1:-1, :-2,  :]         
    p4 = points[:, 2:,   1:-1, :]         

    v1 = p1 - p0
    v2 = p2 - p0
    v3 = p3 - p0
    v4 = p4 - p0

                      
    n0 = torch.cross(v1, v2, dim=-1)
    n1 = torch.cross(v2, v3, dim=-1)
    n2 = torch.cross(v3, v4, dim=-1)
    n3 = torch.cross(v4, v1, dim=-1)

                    
    n = n0 + n1 + n2 + n3
    n = F.normalize(n, dim=-1, p=2)                    

    return n

def normal_loss(pred_height, gt_height, hori_centers, mask=None):
    """
    pred_height:  [B, H, W]   - predicted elevation
    gt_height:    [B, H, W]   - ground truth elevation
    hori_centers: [H, W, 2]   - (x, y) horizontal position of each pixel in meters
    mask:         [B, H, W]   - bool, valid pixels
    """

    print("Horicenter", hori_centers.shape)
    pred_points = get_3d_points_from_height(pred_height, hori_centers)
    gt_points   = get_3d_points_from_height(gt_height,   hori_centers)

    pred_normals = compute_normals(pred_points)                     
    gt_normals   = compute_normals(gt_points)
    print("prednormal",pred_normals.shape)
                            
    cos_sim = (pred_normals * gt_normals).sum(dim=-1)                 
    print("cossim", cos_sim.shape)
    loss = 1.0 - cos_sim

    if mask is not None:
        mask = mask[:, 1:-1, 1:-1]
                                           
        valid = mask.sum(dim=(1, 2)).clamp(min=1.0)

                                                         
        loss_per_sample = (loss * mask).sum(dim=(1, 2)) / valid
        

    else:
        loss_per_sample = loss.mean(dim=(1, 2))

                      
    return loss_per_sample.mean()
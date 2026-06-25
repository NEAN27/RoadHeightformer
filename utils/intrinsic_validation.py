"""
Validation utilities for camera intrinsics after preprocessing.
Verify correctness of intrinsic matrix adjustments during image resizing.
"""

import torch
import numpy as np


def validate_intrinsic_scaling(K_original, K_preprocessed, original_size, target_size, tolerance=1e-5):
    """
    Validate that intrinsics were scaled correctly when resizing image.
    
    For image resize from (W_old, H_old) to (W_new, H_new):
    - fx_new = fx_old * (W_new / W_old)
    - fy_new = fy_old * (H_new / H_old)
    - cx_new = cx_old * (W_new / W_old)
    - cy_new = cy_old * (H_new / H_old)
    """
    K_original = K_original.cpu().numpy() if isinstance(K_original, torch.Tensor) else K_original
    K_preprocessed = K_preprocessed.cpu().numpy() if isinstance(K_preprocessed, torch.Tensor) else K_preprocessed
    
    W_old, H_old = original_size
    W_new, H_new = target_size
    
    scale_x = W_new / W_old
    scale_y = H_new / H_old
    
    # Expected scaled intrinsics
    K_expected = K_original.copy().astype(np.float32)
    K_expected[0, 0] *= scale_x  # fx
    K_expected[1, 1] *= scale_y  # fy
    K_expected[0, 2] *= scale_x  # cx
    K_expected[1, 2] *= scale_y  # cy
    
    # Check differences
    diff = np.abs(K_preprocessed - K_expected)
    max_diff = np.max(diff)
    
    print("=" * 70)
    print("INTRINSIC SCALING VALIDATION")
    print("=" * 70)
    print(f"Original size: {W_old}×{H_old}")
    print(f"Target size:   {W_new}×{H_new}")
    print(f"Scale factors: x={scale_x:.6f}, y={scale_y:.6f}")
    print()
    print("Expected intrinsics:")
    print(K_expected)
    print("\nPreprocessed intrinsics:")
    print(K_preprocessed)
    print("\nDifference (absolute):")
    print(diff)
    print(f"\nMax difference: {max_diff:.2e}")
    print(f"Tolerance: {tolerance:.2e}")
    
    if max_diff < tolerance:
        print("✅ PASS: Intrinsics scaled correctly")
        return True
    else:
        print("❌ FAIL: Intrinsic scaling error exceeds tolerance")
        return False


def validate_intrinsic_range(K, img_size):
    """
    Validate that intrinsic values are within reasonable ranges.
    """
    W, H = img_size
    fx, fy = K[0, 0].item(), K[1, 1].item()
    cx, cy = K[0, 2].item(), K[1, 2].item()
    
    print("=" * 70)
    print("INTRINSIC RANGE VALIDATION")
    print("=" * 70)
    print(f"Image size: {W}×{H}")
    print(f"fx = {fx:.2f}, fy = {fy:.2f}")
    print(f"cx = {cx:.2f}, cy = {cy:.2f}")
    print()
    
    checks = []
    
    # Focal length should be positive and reasonable
    if fx > 0 and fy > 0:
        print("✅ Focal lengths are positive")
        checks.append(True)
    else:
        print("❌ Focal lengths must be positive")
        checks.append(False)
    
    # Principal point should be inside image
    if 0 < cx < W and 0 < cy < H:
        print(f"✅ Principal point ({cx:.1f}, {cy:.1f}) is inside image")
        checks.append(True)
    else:
        print(f"❌ Principal point ({cx:.1f}, {cy:.1f}) is outside image!")
        checks.append(False)
    
    # Focal length shouldn't be extreme
    aspect_ratio = fx / fy
    if 0.8 < aspect_ratio < 1.2:
        print(f"✅ Focal length aspect ratio {aspect_ratio:.3f} is reasonable")
        checks.append(True)
    else:
        print(f"⚠️  Focal length aspect ratio {aspect_ratio:.3f} seems unusual")
        checks.append(False)
    
    # Field of view should be reasonable (5° to 120°)
    fov_x = 2 * np.arctan(W / (2 * fx)) * 180 / np.pi
    fov_y = 2 * np.arctan(H / (2 * fy)) * 180 / np.pi
    print(f"Horizontal FoV: {fov_x:.1f}°, Vertical FoV: {fov_y:.1f}°")
    if 5 < fov_x < 120 and 5 < fov_y < 120:
        print("✅ Field of view is reasonable")
        checks.append(True)
    else:
        print("⚠️  Field of view seems unusual")
        checks.append(False)
    
    return all(checks)


def validate_reprojection_bounds(K, voxel_uv, image_size):
    """
    Validate that projected 3D points fall within image bounds.
    """
    W, H = image_size
    
    print("=" * 70)
    print("REPROJECTION BOUNDS VALIDATION")
    print("=" * 70)
    print(f"Image size: {W}×{H}")
    
    if isinstance(voxel_uv, torch.Tensor):
        voxel_uv = voxel_uv.cpu().numpy()
    
    u = voxel_uv[0, :]  # x coordinates
    v = voxel_uv[1, :]  # y coordinates
    
    # Check bounds
    valid_u = (u >= 0) & (u < W)
    valid_v = (v >= 0) & (v < H)
    valid_both = valid_u & valid_v
    
    total = len(u)
    valid_count = np.sum(valid_both)
    valid_pct = 100 * valid_count / total
    
    print(f"Total projected points: {total}")
    print(f"Points within bounds: {valid_count} ({valid_pct:.1f}%)")
    print(f"  U range: [{u.min():.1f}, {u.max():.1f}] (valid: [0, {W}))")
    print(f"  V range: [{v.min():.1f}, {v.max():.1f}] (valid: [0, {H}))")
    
    if valid_pct > 70:
        print(f"✅ PASS: {valid_pct:.1f}% of projections are valid")
        return True
    else:
        print(f"⚠️  Warning: Only {valid_pct:.1f}% of projections are valid")
        return False


def compare_downscaled_intrinsics(K, down_scale, target_feature_size):
    """
    Validate downscaled intrinsics for feature map projection.
    
    When features are at downscaled resolution, intrinsics must also be downscaled:
    K_feature = K / down_scale
    """
    K_downscaled = K / down_scale
    
    print("=" * 70)
    print("DOWNSCALED INTRINSICS VALIDATION")
    print("=" * 70)
    print(f"Downscale factor: {down_scale}")
    print(f"Target feature size: {target_feature_size}")
    print()
    
    print("Original intrinsics:")
    print(K)
    print("\nDownscaled intrinsics (÷ {}):"
        .format(down_scale))
    print(K_downscaled)
    print()
    
    fx_down = K_downscaled[0, 0].item()
    fy_down = K_downscaled[1, 1].item()
    cx_down = K_downscaled[0, 2].item()
    cy_down = K_downscaled[1, 2].item()
    
    W_feat, H_feat = target_feature_size
    
    # Check if principal point is reasonable
    if 0 < cx_down < W_feat and 0 < cy_down < H_feat:
        print(f"✅ Principal point ({cx_down:.1f}, {cy_down:.1f}) is valid for {W_feat}×{H_feat}")
        return True
    else:
        print(f"❌ Principal point ({cx_down:.1f}, {cy_down:.1f}) is INVALID for {W_feat}×{H_feat}")
        return False


def full_validation_report(K_orig, K_preproc, original_size, target_size, 
                          voxel_uv=None, down_scale=None, feature_size=None):
    """
    Run comprehensive validation of preprocessed intrinsics.
    """
    print("\n" + "🔍 " * 20)
    print("COMPREHENSIVE INTRINSIC VALIDATION REPORT")
    print("🔍 " * 20 + "\n")
    
    results = {}
    
    # 1. Scaling validation
    results['scaling'] = validate_intrinsic_scaling(K_orig, K_preproc, original_size, target_size)
    print()
    
    # 2. Range validation
    results['range'] = validate_intrinsic_range(K_preproc, target_size)
    print()
    
    # 3. Reprojection bounds (if provided)
    if voxel_uv is not None:
        results['reprojection'] = validate_reprojection_bounds(K_preproc, voxel_uv, target_size)
        print()
    
    # 4. Downscaled validation (if provided)
    if down_scale is not None and feature_size is not None:
        results['downscaled'] = compare_downscaled_intrinsics(K_preproc, down_scale, feature_size)
        print()
    
    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:20s}: {status}")
    
    all_pass = all(results.values())
    print()
    if all_pass:
        print("✅ ALL CHECKS PASSED - Intrinsics are likely correct!")
    else:
        print("⚠️  Some checks failed - Review preprocessing logic")
    
    return results


# Example usage functions
def example_validation_after_crop_image():
    """
    Example: Validate intrinsics after resize operation
    """
    # Original values
    K_orig = np.array([
        [1920.0, 0, 960.0],
        [0, 1920.0, 540.0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    original_size = (1920, 1080)
    target_size = (952, 532)
    
    # Simulated preprocessed intrinsics
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]
    K_preproc = K_orig.copy()
    K_preproc[0, 0] *= scale_x
    K_preproc[1, 1] *= scale_y
    K_preproc[0, 2] *= scale_x
    K_preproc[1, 2] *= scale_y
    
    print("Example: Intrinsic Validation After Image Resize")
    print("-" * 70)
    validate_intrinsic_scaling(K_orig, K_preproc, original_size, target_size)


if __name__ == "__main__":
    example_validation_after_crop_image()

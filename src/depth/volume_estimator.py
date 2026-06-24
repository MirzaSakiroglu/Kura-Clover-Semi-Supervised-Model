import numpy as np
import open3d as o3d
import cv2

class VolumeEstimator:
    """Computes biomass volume using metric depth and 2D semantic masks."""
    def __init__(self, voxel_size=0.01):
        # Default 1cm voxel resolution
        self.voxel_size = voxel_size

    def compute_volume(self, depth_map, mask, f_px, downsample_factor=4):
        """
        Computes canopy volume from depth map.
        
        Args:
            depth_map (np.ndarray): HxW depth map in meters
            mask (np.ndarray): HxW binary mask (1 for clover, 0 for background/soil)
            f_px (float): Focal length in pixels
            downsample_factor (int): Downsample to speed up point cloud processing
            
        Returns:
            volume_m3 (float): Estimated canopy volume in cubic meters
            avg_height (float): Average canopy height above ground plane in meters
        """
        h, w = depth_map.shape
        
        # Downsample for faster O3D processing
        dh, dw = h // downsample_factor, w // downsample_factor
        depth_down = cv2.resize(depth_map, (dw, dh), interpolation=cv2.INTER_NEAREST)
        mask_down = cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
        
        # Intrinsic parameters based on f_px and downsampled resolution
        fx = fy = f_px / downsample_factor
        cx = dw / 2.0
        cy = dh / 2.0
        
        # Vectorized 3D projection
        x_grid, y_grid = np.meshgrid(np.arange(dw), np.arange(dh))
        
        # Filter out invalid depth
        valid = depth_down > 0
        z = depth_down[valid]
        x = x_grid[valid]
        y = y_grid[valid]
        m = mask_down[valid]
        
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        
        points = np.column_stack((X, Y, z))
        is_clover = m
        
        if len(points) == 0:
            return 0.0, 0.0
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate ground plane using RANSAC on the non-clover points (soil)
        non_clover_idx = np.where(is_clover == 0)[0]
        if len(non_clover_idx) < 100:
            # Not enough ground points, fallback to using all points
            non_clover_idx = np.arange(len(points))
            
        ground_pcd = pcd.select_by_index(non_clover_idx)
        
        try:
            plane_model, inliers = ground_pcd.segment_plane(distance_threshold=0.05,
                                                          ransac_n=3,
                                                          num_iterations=1000)
            [a, b, c, d] = plane_model
        except RuntimeError:
            # RANSAC failed
            return 0.0, 0.0
            
        # Calculate heights of clover points above ground plane
        clover_idx = np.where(is_clover > 0)[0]
        if len(clover_idx) == 0:
            return 0.0, 0.0
            
        clover_points = points[clover_idx]
        
        # Distance to plane: |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
        denom = np.sqrt(a**2 + b**2 + c**2)
        heights = np.abs(np.dot(clover_points, [a, b, c]) + d) / denom
        
        # Filter out extreme outliers (e.g. noise points floating way too high)
        valid_heights = heights[heights < 1.0] # Assume kura clover doesn't exceed 1 meter
        if len(valid_heights) == 0:
            return 0.0, 0.0
            
        avg_height = np.mean(valid_heights)
        
        # Voxel grid volume estimation
        clover_pcd = pcd.select_by_index(clover_idx)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(clover_pcd, voxel_size=self.voxel_size)
        num_voxels = len(voxel_grid.get_voxels())
        volume_m3 = num_voxels * (self.voxel_size ** 3)
        
        return volume_m3, float(avg_height)

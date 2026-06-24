import cv2
import numpy as np

class MaskMetrics:
    """Computes phenotypic metrics from a 6-class segmentation mask."""
    
    def __init__(self, mask: np.ndarray, class_map: dict):
        self.mask = mask
        self.class_map = class_map
        
        # Extract class indices
        self.quadrat_idx = class_map.get('quadrat', 1)
        self.leaves_idx = class_map.get('clover_leaves', 2)
        self.stems_idx = class_map.get('clover_stems', 3)
        
        # Binarize masks
        self.quadrat_mask = (mask == self.quadrat_idx).astype(np.uint8)
        self.leaves_mask = (mask == self.leaves_idx).astype(np.uint8)
        self.stems_mask = (mask == self.stems_idx).astype(np.uint8)
        
        # Combined clover (leaves + stems)
        self.clover_mask = ((mask == self.leaves_idx) | (mask == self.stems_idx)).astype(np.uint8)
        
        # Cache basic areas
        self.quadrat_area = np.sum(self.quadrat_mask)
        self.leaves_area = np.sum(self.leaves_mask)
        self.stems_area = np.sum(self.stems_mask)
        self.clover_area = self.leaves_area + self.stems_area
        
    def compute_all(self):
        """Computes all structural and morphological metrics."""
        
        # Quadrat is required for normalizing areas realistically
        if self.quadrat_area == 0:
            return self._empty_metrics()
            
        # Get contours of the full clover plant
        contours, _ = cv2.findContours(self.clover_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._empty_metrics()
            
        # Global convex hull (simulating a single canopy)
        all_points = np.vstack(contours)
        hull = cv2.convexHull(all_points)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            return self._empty_metrics()
            
        # Perimeter sum for circularity
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        circularity = (4 * np.pi * self.clover_area) / (perimeter ** 2) if perimeter > 0 else 0
            
        # Connected components for rhizome spread estimation
        num_labels, _ = cv2.connectedComponents(self.clover_mask)
        num_components = num_labels - 1 # Exclude background
        
        metrics = {
            "clover_area_normalized": self.clover_area / self.quadrat_area,
            "convex_hull_area_normalized": hull_area / self.quadrat_area,
            "solidity": self.clover_area / hull_area,
            "circularity": circularity,
            "leaf_proportion": self.leaves_area / self.clover_area if self.clover_area > 0 else 0,
            "stem_proportion": self.stems_area / self.clover_area if self.clover_area > 0 else 0,
            "disconnected_components": num_components,
            "quadrat_px": int(self.quadrat_area),
            "clover_px": int(self.clover_area)
        }
        
        return metrics

    def _empty_metrics(self):
        return {
            "clover_area_normalized": 0.0,
            "convex_hull_area_normalized": 0.0,
            "solidity": 0.0,
            "circularity": 0.0,
            "leaf_proportion": 0.0,
            "stem_proportion": 0.0,
            "disconnected_components": 0,
            "quadrat_px": int(self.quadrat_area),
            "clover_px": int(self.clover_area)
        }

import torch
import numpy as np
import cv2

class DepthEstimator:
    """Wrapper for Apple Depth Pro zero-shot metric depth estimation."""
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print("Initializing Depth Pro estimator (ensure depth-pro is installed via pip).")
        try:
            import depth_pro
            # Load model and preprocessing transforms
            self.model, self.transform = depth_pro.create_model_and_transforms()
            self.model = self.model.to(self.device)
            self.model.eval()
            self.installed = True
        except ImportError:
            print("Warning: depth_pro not installed. Returning dummy depth maps for testing.")
            self.installed = False
            
    def estimate(self, image_path):
        """
        Estimates depth from a single image.
        
        Returns:
            depth (np.ndarray): Depth map in meters, shape (H, W).
            f_px (float): Estimated focal length in pixels.
        """
        if not self.installed:
            img = cv2.imread(image_path)
            h, w = img.shape[:2]
            # Return dummy flat ground plane with some noise
            dummy_depth = np.ones((h, w), dtype=np.float32) * 1.5 + np.random.normal(0, 0.05, (h, w))
            return dummy_depth, 3000.0
            
        import depth_pro
        image, _, f_px = depth_pro.load_rgb(image_path)
        image_tensor = self.transform(image).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(image_tensor)
            depth = prediction["depth"].cpu().numpy()
            
        # Ensure depth map matches original image size
        orig_img = cv2.imread(image_path)
        h, w = orig_img.shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            
        return depth, f_px

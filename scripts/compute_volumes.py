import os
import cv2
import glob
import argparse
import pandas as pd
from tqdm import tqdm
import json

from src.depth.depth_estimator import DepthEstimator
from src.depth.volume_estimator import VolumeEstimator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True, help='Directory of source RGB images')
    parser.add_argument('--masks-dir', type=str, required=True, help='Directory of inferred PNG masks')
    parser.add_argument('--output-csv', type=str, default='outputs/volume_estimates.csv')
    parser.add_argument('--class-map', type=str, default='metadata/segmentation_class_map.json')
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.class_map, 'r') as f:
        class_map = json.load(f)
        
    leaves_idx = class_map.get('clover_leaves', 2)
    stems_idx = class_map.get('clover_stems', 3)
    
    depth_estimator = DepthEstimator()
    volume_estimator = VolumeEstimator(voxel_size=0.01) # 1cm voxels
    
    img_paths = glob.glob(os.path.join(args.images_dir, '*.*'))
    results = []
    
    for img_path in tqdm(img_paths, desc="Computing Volumes"):
        base_name = os.path.basename(img_path)
        # Handle matching logic, assuming mask is named base_name.png
        mask_name = os.path.splitext(base_name)[0] + ".png"
        mask_path = os.path.join(args.masks_dir, mask_name)
        
        if not os.path.exists(mask_path):
            print(f"Skipping {base_name}, no mask found.")
            continue
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Binary clover mask
        clover_mask = ((mask == leaves_idx) | (mask == stems_idx)).astype(np.uint8)
        
        # Estimate depth
        depth_map, f_px = depth_estimator.estimate(img_path)
        
        # Estimate volume
        vol_m3, avg_h = volume_estimator.compute_volume(depth_map, clover_mask, f_px)
        
        results.append({
            'image_filename': base_name,
            'volume_m3': vol_m3,
            'avg_canopy_height_m': avg_h
        })
        
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Volume estimates saved to {args.output_csv}")

if __name__ == '__main__':
    main()

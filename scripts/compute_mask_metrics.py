import os
import cv2
import glob
import json
import argparse
import pandas as pd
from tqdm import tqdm

from src.mask_metrics import MaskMetrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--masks-dir', type=str, required=True, help='Directory containing inferred PNG masks')
    parser.add_argument('--output-csv', type=str, default='outputs/mask_metrics.csv')
    parser.add_argument('--class-map', type=str, default='metadata/segmentation_class_map.json')
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.class_map, 'r') as f:
        class_map = json.load(f)
        
    mask_paths = glob.glob(os.path.join(args.masks_dir, '*.png'))
    results = []
    
    for path in tqdm(mask_paths, desc="Computing Metrics"):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load {path}")
            continue
            
        metric_calc = MaskMetrics(mask, class_map)
        metrics = metric_calc.compute_all()
        
        # Add filename as ID to allow joining with accession metadata
        metrics['image_filename'] = os.path.basename(path).replace('_mask.png', '.png').replace('.png', '.jpg')
        results.append(metrics)
        
    df = pd.DataFrame(results)
    
    # Reorder columns to put image_filename first
    cols = ['image_filename'] + [c for c in df.columns if c != 'image_filename']
    df = df[cols]
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Metrics saved to {args.output_csv}")

if __name__ == '__main__':
    main()

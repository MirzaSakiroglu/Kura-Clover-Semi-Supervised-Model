import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing images/ and targets/')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--size', type=int, nargs=2, default=[1024, 1024], help='Target size (W, H)')
    return parser.parse_args()

def process_directory(args):
    img_dir = os.path.join(args.input_dir, 'images')
    tgt_dir = os.path.join(args.input_dir, 'targets')
    
    out_img_dir = os.path.join(args.output_dir, 'images')
    out_tgt_dir = os.path.join(args.output_dir, 'targets')
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_tgt_dir, exist_ok=True)
    
    img_paths = glob.glob(os.path.join(img_dir, '*.*'))
    
    for img_path in tqdm(img_paths, desc=f"Resizing {args.input_dir}"):
        img_name = os.path.basename(img_path)
        base_name, _ = os.path.splitext(img_name)
        tgt_name = base_name + '.png'
        tgt_path = os.path.join(tgt_dir, tgt_name)
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
            
        # Resize image using Lanczos4
        img_resized = cv2.resize(img, tuple(args.size), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(out_img_dir, img_name), img_resized)
        
        if os.path.exists(tgt_path):
            tgt = cv2.imread(tgt_path, cv2.IMREAD_UNCHANGED)
            # Resize mask using Nearest Neighbor to preserve class indices
            tgt_resized = cv2.resize(tgt, tuple(args.size), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(out_tgt_dir, tgt_name), tgt_resized)

if __name__ == '__main__':
    args = parse_args()
    process_directory(args)

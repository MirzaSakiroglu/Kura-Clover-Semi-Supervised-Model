"""
scripts/export_labelbox_masks.py
Downloads mask annotations from the new Labelbox NDJSON export and 
composites them into 6-class index-encoded PNG masks.

The NDJSON masks are individual per-annotation binary masks hosted on Labelbox.
This script downloads each mask, maps the class name to the 6-class index,
and writes the composited mask.

The output masks OVERWRITE the Zenodo targets for the 7 labeled training images,
giving them the full 6-class resolution (leaves vs stems vs person vs other_veg).

Requirements:
    pip install Pillow numpy requests

Usage:
    python scripts/export_labelbox_masks.py \
        --ndjson "metadata/Export  project - K1702 - 6_23_2026.ndjson" \
        --output-dir data/processed/train/labeled/targets \
        --class-map metadata/segmentation_class_map.json \
        --api-key YOUR_LABELBOX_API_KEY
"""

import os
import sys
import json
import argparse
import urllib.request
import time
import numpy as np
from PIL import Image
import io


# 6-class mapping
DEFAULT_CLASS_MAP = {
    "soil": 0,
    "quadrat": 1,
    "clover_leaves": 2,
    "clover_stems": 3,
    "person": 4,
    "other_veg": 5
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ndjson', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='data/processed/train/labeled/targets')
    parser.add_argument('--class-map', type=str, default='metadata/segmentation_class_map.json')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Labelbox API key (for downloading mask images). '
                             'Can also be set via LABELBOX_API_KEY env var.')
    parser.add_argument('--only-train', action='store_true', default=True,
                        help='Only export masks for images in train labeled dir')
    return parser.parse_args()


def download_mask(url, api_key=None, max_retries=3):
    """Download a mask image from a URL, return as numpy array."""
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            img = Image.open(io.BytesIO(data))
            return np.array(img)
        except Exception as e:
            print(f"  Attempt {attempt+1}/{max_retries} failed: {e}")
            time.sleep(2 ** attempt)
    
    print(f"  WARNING: Failed to download mask from {url[:80]}...")
    return None


def main():
    args = parse_args()
    
    # Load class map
    if os.path.exists(args.class_map):
        with open(args.class_map, 'r') as f:
            class_map = json.load(f)
    else:
        class_map = DEFAULT_CLASS_MAP
        
    # API key
    api_key = args.api_key or os.environ.get('LABELBOX_API_KEY', None)
    if not api_key:
        print("WARNING: No Labelbox API key provided. Mask download may fail for authenticated URLs.")
        print("Set --api-key or LABELBOX_API_KEY environment variable.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check which images are in the output dir already (to determine which to process)
    existing_images = set()
    img_dir = os.path.join(os.path.dirname(args.output_dir), 'images')
    if os.path.isdir(img_dir):
        existing_images = {os.path.splitext(f)[0] for f in os.listdir(img_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    
    # Parse NDJSON
    processed = 0
    skipped = 0
    
    with open(args.ndjson, 'r') as f:
        for line_num, line in enumerate(f):
            row = json.loads(line)
            external_id = row['data_row']['external_id']
            img_id = os.path.splitext(external_id)[0]
            
            # Skip if not in our target images
            if args.only_train and existing_images and img_id not in existing_images:
                skipped += 1
                continue
            
            height = row['media_attributes']['height']
            width = row['media_attributes']['width']
            
            print(f"Processing {external_id} ({width}x{height})...")
            
            # Create empty target
            target = np.zeros((height, width), dtype=np.uint8)
            
            # Get annotations from the project
            projects = row.get('projects', {})
            for proj_id, proj_data in projects.items():
                labels = proj_data.get('labels', [])
                for label in labels:
                    objects = label.get('annotations', {}).get('objects', [])
                    
                    for obj in objects:
                        if obj['annotation_kind'] != 'ImageSegmentationMask':
                            continue
                        
                        class_name = obj['name']
                        class_idx = class_map.get(class_name, None)
                        
                        if class_idx is None:
                            print(f"  WARNING: Unknown class '{class_name}', skipping")
                            continue
                        
                        mask_url = obj.get('mask', {}).get('url', '')
                        if not mask_url:
                            continue
                        
                        mask_arr = download_mask(mask_url, api_key)
                        if mask_arr is None:
                            continue
                        
                        # Handle RGBA/RGB/grayscale masks
                        if len(mask_arr.shape) == 3:
                            # Use any channel (they're typically all the same for binary masks)
                            binary = mask_arr[..., 0] > 127
                        else:
                            binary = mask_arr > 127
                        
                        # Later classes overwrite earlier ones (higher priority)
                        target[binary] = class_idx
            
            # Save the composited mask
            output_path = os.path.join(args.output_dir, f"{img_id}.png")
            mask_img = Image.fromarray(target, mode='L')
            mask_img.save(output_path)
            print(f"  Saved: {output_path} (unique values: {np.unique(target)})")
            processed += 1
    
    print(f"\nDone! Processed {processed} images, skipped {skipped}.")
    if processed == 0 and api_key is None:
        print("Hint: You may need to provide a Labelbox API key to download the masks.")
        print("The Zenodo 3-class targets are already in place and compatible.")


if __name__ == '__main__':
    main()

import os
import cv2
import glob
import torch
import numpy as np
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import OrderedDict

from src.models import create_smp_model

CLASS_NAMES = {
    0: 'soil',
    1: 'quadrat',
    2: 'clover_leaves',
    3: 'clover_stems',
    4: 'person',
    5: 'other_veg',
}

def get_class_colors():
    # BGR format for OpenCV
    return {
        0: [101, 119, 139], # soil
        1: [255, 255, 255], # quadrat
        2: [34, 139, 34],   # clover_leaves
        3: [144, 238, 144], # clover_stems
        4: [0, 0, 255],     # person
        5: [0, 165, 255],   # other_veg
    }

def apply_color_map(mask, color_map):
    res = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls_idx, color in color_map.items():
        res[mask == cls_idx] = color
    return res

def save_heatmaps(probs, base_name, output_dir):
    heatmap_dir = os.path.join(output_dir, 'heatmaps', base_name)
    os.makedirs(heatmap_dir, exist_ok=True)

    for cls_idx, cls_name in CLASS_NAMES.items():
        prob_map = probs[cls_idx]  # shape (H, W), values 0.0–1.0

        # Scale to 0–255 and apply JET colormap (blue=low, red=high confidence)
        prob_uint8 = (prob_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)

        out_path = os.path.join(heatmap_dir, f"{cls_idx}_{cls_name}.jpg")
        cv2.imwrite(out_path, heatmap)

    print(f"  Heatmaps saved to {heatmap_dir}/")

def sliding_window_inference(model, image, tile_size=(1024, 1024), step_size=(512, 512), device='cuda'):
    h, w = image.shape[:2]
    th, tw = tile_size
    sh, sw = step_size

    # Pad image to ensure full coverage
    pad_h = (th - h % th) % th if h % th != 0 else 0
    pad_w = (tw - w % tw) % tw if w % tw != 0 else 0
    image_padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    ph, pw = image_padded.shape[:2]

    probs = np.zeros((6, ph, pw), dtype=np.float32)
    counts = np.zeros((ph, pw), dtype=np.float32)

    for y in range(0, ph - th + 1, sh):
        for x in range(0, pw - tw + 1, sw):
            patch = image_padded[y:y+th, x:x+tw]
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            patch = (patch - mean) / std

            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(patch_tensor)
                patch_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            probs[:, y:y+th, x:x+tw] += patch_probs
            counts[y:y+th, x:x+tw] += 1

    # Normalize by counts and crop back to original size
    probs /= np.maximum(counts, 1)
    probs = probs[:, :h, :w]
    mask = np.argmax(probs, axis=0).astype(np.uint8)
    return mask, probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_semisup_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--use-ema', action='store_true', help='Use EMA weights if available')
    parser.add_argument('--heatmaps', action='store_true', help='Save per-class confidence heatmaps')
    args = parser.parse_args()

    from src.utils.config import TrainSemiSupervisedConfig
    yaml_conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(OmegaConf.structured(TrainSemiSupervisedConfig), yaml_conf)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Rebuild model
    model = create_smp_model(conf).to(device)

    # Load checkpoint
    print(f"Loading checkpoint {args.checkpoint}...")
    chkpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if args.use_ema and 'ema_shadow_params' in chkpt:
        state_dict = chkpt['ema_shadow_params']
        print("Loaded EMA weights.")
    elif args.use_ema and 'ema_state_dict' in chkpt:
        state_dict = chkpt['ema_state_dict']
        print("Loaded EMA weights.")
    else:
        state_dict = chkpt.get('model_state_dict', chkpt)
        print("Loaded standard weights.")

    # Handle DDP 'module.' prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    os.makedirs(os.path.join(args.output_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'overlays'), exist_ok=True)

    color_map = get_class_colors()

    img_paths = glob.glob(os.path.join(args.input_dir, '*.*'))
    for img_path in tqdm(img_paths, desc="Inference"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        mask, probs = sliding_window_inference(model, img, tile_size=(1024, 1024), step_size=(512, 512), device=device)

        # Save raw mask
        cv2.imwrite(os.path.join(args.output_dir, 'masks', f"{base_name}.png"), mask)

        # Save overlay
        colored_mask = apply_color_map(mask, color_map)
        overlay = cv2.addWeighted(img, 0.6, colored_mask, 0.4, 0)
        cv2.imwrite(os.path.join(args.output_dir, 'overlays', f"{base_name}_overlay.jpg"), overlay)

        # Save per-class heatmaps only if requested
        if args.heatmaps:
            save_heatmaps(probs, base_name, args.output_dir)

if __name__ == '__main__':
    main()

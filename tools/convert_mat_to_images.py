"""
Convert a folder of .mat files (with structure like the Colab example) into a filesystem dataset:
- saves images to new_dataset/bt_images
- saves masks to new_dataset/bt_mask (if present)
- writes labels.npy and borders.npy in new_dataset

Usage (example):
python tools/convert_mat_to_images.py --src "<mat_dir>" --out "new_dataset" --start 1 --end 3064 --ext .mat --size 512

This script is defensive: if files are missing it will skip and continue.
"""
import os
import argparse
import h5py
import numpy as np
from PIL import Image


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def read_cjdata(mat_path):
    # returns (img_array (H,W) uint8 or float), label (int), mask_array or None, border_coords (1D array) or None
    with h5py.File(mat_path, 'r') as f:
        if 'cjdata' not in f:
            raise RuntimeError('No cjdata group in {}'.format(mat_path))
        cj = f['cjdata']
        # image
        img = None
        if 'image' in cj:
            img = np.array(cj['image'][()])
        # label
        label = None
        if 'label' in cj:
            try:
                label = int(np.array(cj['label'])[0][0])
            except Exception:
                # fallback if label stored differently
                label = int(np.array(cj['label']).squeeze())
        # mask
        mask = None
        if 'tumorMask' in cj:
            mask = np.array(cj['tumorMask'][()])
        # border
        border = None
        if 'tumorBorder' in cj:
            border = np.array(cj['tumorBorder'][()]).squeeze()
    return img, label, mask, border


def save_image(arr, path, cmap='gray'):
    # arr expected HxW or HxWxC
    if arr is None:
        return
    if arr.ndim == 2:
        # convert to uint8 if not already
        if arr.dtype != np.uint8:
            # scale if in float [-1,1] or [0,1] or int16
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                a = arr
                # try common scalings
                if a.min() >= -1.1 and a.max() <= 1.1:
                    a = ((a + 1.0) * 127.5).clip(0,255)
                elif a.max() <= 1.1:
                    a = (a * 255.0).clip(0,255)
                else:
                    a = a.clip(0,255)
                arr8 = a.astype(np.uint8)
            else:
                arr8 = arr.astype(np.uint8)
        else:
            arr8 = arr
        im = Image.fromarray(arr8)
        im.save(path)
    elif arr.ndim == 3:
        arr8 = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
        im = Image.fromarray(arr8)
        im.save(path)
    else:
        raise RuntimeError('Unsupported array shape: {}'.format(arr.shape))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='Directory with .mat files')
    p.add_argument('--out', default='new_dataset', help='Output folder (will contain bt_images, bt_mask, labels.npy, borders.npy)')
    p.add_argument('--start', type=int, default=1)
    p.add_argument('--end', type=int, default=3064)
    p.add_argument('--ext', default='.mat')
    p.add_argument('--size', type=int, default=None, help='If given, will resize images to size x size')
    args = p.parse_args()

    src = args.src
    out = args.out
    images_dir = os.path.join(out, 'bt_images')
    masks_dir = os.path.join(out, 'bt_mask')
    ensure_dir(images_dir)
    ensure_dir(masks_dir)

    labels = []
    borders = []

    count = 0
    for i in range(args.start, args.end + 1):
        fname = str(i) + args.ext
        mat_path = os.path.join(src, fname)
        if not os.path.exists(mat_path):
            # skip missing files
            continue
        try:
            img, label, mask, border = read_cjdata(mat_path)
        except Exception as e:
            print('Failed to read', mat_path, 'err:', e)
            continue
        if img is None:
            print('No image for', mat_path)
            continue
        # optionally resize
        if args.size is not None:
            # use PIL for resizing
            pil = Image.fromarray(img.astype(np.uint8) if img.dtype != np.uint8 else img)
            pil = pil.resize((args.size, args.size))
            img_out = np.array(pil)
        else:
            img_out = img

        out_img_path = os.path.join(images_dir, f"{i}.png")
        save_image(img_out, out_img_path)

        if mask is not None:
            if args.size is not None:
                pilm = Image.fromarray(mask.astype(np.uint8)).resize((args.size, args.size))
                mask_out = np.array(pilm)
            else:
                mask_out = mask
            out_mask_path = os.path.join(masks_dir, f"{i}.png")
            save_image(mask_out, out_mask_path)
        else:
            out_mask_path = None

        labels.append(label if label is not None else -1)
        borders.append(border if border is not None else np.array([]))
        count += 1
        if count % 100 == 0:
            print('Processed', count)

    labels = np.array(labels, dtype=np.int64)
    # store
    np.save(os.path.join(out, 'labels.npy'), labels)
    # borders is an array of objects — save as npy with object dtype
    np.save(os.path.join(out, 'borders.npy'), np.array(borders, dtype=object))

    print(f'Done. {count} files processed. Images saved in {images_dir}, masks in {masks_dir}')


if __name__ == '__main__':
    main()

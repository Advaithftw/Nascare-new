import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MATFolderDataset(Dataset):
    """Loads images from a folder produced by convert_mat_to_images.py
    Expects folder structure:
      out/
        bt_images/  (images named 1.png ... N.png)
        bt_mask/    (optional masks)
        labels.npy
        borders.npy
    """
    def __init__(self, root, transform=None, use_mask=False):
        self.root = root
        self.images_dir = os.path.join(root, 'bt_images')
        self.masks_dir = os.path.join(root, 'bt_mask')
        self.labels = np.load(os.path.join(root, 'labels.npy'))
        # borders may be object array
        try:
            self.borders = np.load(os.path.join(root, 'borders.npy'), allow_pickle=True)
        except Exception:
            self.borders = None
        self.transform = transform
        self.use_mask = use_mask
        # collect image filenames that match numeric names
        files = [f for f in os.listdir(self.images_dir) if f.lower().endswith('.png') or f.lower().endswith('.jpg')]
        # sort by numeric stem
        def keyfn(x):
            try:
                return int(os.path.splitext(x)[0])
            except Exception:
                return x
        files.sort(key=keyfn)
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.images_dir, fname)
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # label index: filenames are 1-based usually; convert by stem
        try:
            stem = int(os.path.splitext(fname)[0])
            label = int(self.labels[stem-1])
        except Exception:
            label = int(self.labels[idx])
        ret = {'image': img, 'label': label, 'fname': fname}
        if self.use_mask:
            mpath = os.path.join(self.masks_dir, fname)
            if os.path.exists(mpath):
                m = Image.open(mpath).convert('L')
                if self.transform:
                    # do not use color transforms on mask; resize handled by PIL when saved
                    m = np.array(m)
                ret['mask'] = m
            else:
                ret['mask'] = None
        return ret

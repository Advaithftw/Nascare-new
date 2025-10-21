import sys
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np

# Ensure project root is on sys.path so `import backend` works
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import backend.model as backend_model

# Simple helper to load image files from Testing/ (one per class up to max_per_class)
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEST_DIR = os.path.join(BASE, 'Testing')
CLASS_NAMES = backend_model.CLASS_NAMES
IMAGE_SIZE = backend_model.IMAGE_TARGET_SIZE

preprocess = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    # If your model expects normalization, uncomment and match training normalization
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def collect_test_images(max_per_class=3):
    imgs = []
    labels = []
    for i, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith('.png') or f.lower().endswith('.jpg')]
        for fname in files[:max_per_class]:
            imgs.append(os.path.join(cls_dir, fname))
            labels.append(i)
    return imgs, labels


def load_model_from_path(path):
    # Reuse backend loader but with small wrapper to accept arbitrary path
    # We'll instantiate NASModel and then try to load the checkpoint like backend.model does
    arch = backend_model.BEST_MODEL_ARCHITECTURE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = backend_model.NASModel(arch, num_classes=backend_model.NUM_CLASSES)
    loaded = torch.load(path, map_location=device)
    if isinstance(loaded, dict):
        # common nested
        for key in ('model', 'state_dict', 'model_state_dict'):
            if key in loaded and isinstance(loaded[key], dict):
                loaded = loaded[key]
                break
        # attempt prefix strip
        def strip(sd):
            out = {}
            for k,v in sd.items():
                newk = k
                for p in ('model.', 'module.'):
                    if k.startswith(p):
                        newk = k[len(p):]
                        break
                out[newk]=v
            return out
        try:
            model.load_state_dict(loaded)
        except Exception:
            stripped = strip(loaded)
            model.load_state_dict(stripped, strict=False)
    elif isinstance(loaded, torch.nn.Module):
        model = loaded
    else:
        raise RuntimeError('Unknown model file')
    model.to(device).eval()
    return model


def evaluate_model(model, image_paths, labels):
    device = next(model.parameters()).device
    correct = 0
    total = 0
    per_image = []
    for p, lbl in zip(image_paths, labels):
        img = Image.open(p).convert('RGB')
        inp = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            probs = torch.nn.functional.softmax(out[0], dim=0)
            pred = int(torch.argmax(probs).item())
            per_image.append((os.path.basename(p), CLASS_NAMES[lbl], CLASS_NAMES[pred], float(probs[pred].item())))
            correct += (pred == lbl)
            total += 1
    acc = correct / total if total else 0
    return acc, per_image


if __name__ == '__main__':
    # gather models
    models_to_try = []
    # root best.pth
    root_best = os.path.join(BASE, 'best.pth')
    if os.path.exists(root_best):
        models_to_try.append(root_best)
    # models/ directory
    models_dir = os.path.join(BASE, 'backend', 'models')
    if os.path.isdir(models_dir):
        for f in os.listdir(models_dir):
            if f.lower().endswith('.pth') or f.lower().endswith('.pt'):
                models_to_try.append(os.path.join(models_dir, f))

    # Use a large max_per_class to include all available test images per class
    image_paths, labels = collect_test_images(max_per_class=1000)
    print('Test images:', image_paths)

    results = {}
    for mpath in models_to_try:
        print('\nEvaluating', mpath)
        try:
            mdl = load_model_from_path(mpath)
            acc, per_image = evaluate_model(mdl, image_paths, labels)
            results[os.path.basename(mpath)] = (acc, per_image)
            print('Accuracy:', acc)
            for info in per_image:
                print(info)
        except Exception as e:
            print('Failed to evaluate', mpath, 'error:', e)

    print('\nSummary:')
    for k,v in results.items():
        print(k, '->', v[0])

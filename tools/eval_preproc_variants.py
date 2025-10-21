import sys
import os
import torch
from PIL import Image
from torchvision import transforms

# ensure project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import backend.model as backend_model

BASE = ROOT
TEST_DIR = os.path.join(BASE, 'Testing')
CLASS_NAMES = backend_model.CLASS_NAMES

# collect test images (same as previous script)
def collect_test_images(max_per_class=5):
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


def make_preprocess(image_size=(224,224), normalize=False):
    steps = [transforms.Resize(image_size), transforms.ToTensor()]
    if normalize:
        steps.append(transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    return transforms.Compose(steps)


def load_model_from_path(path):
    arch = backend_model.BEST_MODEL_ARCHITECTURE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = backend_model.NASModel(arch, num_classes=backend_model.NUM_CLASSES)
    loaded = torch.load(path, map_location=device)
    if isinstance(loaded, dict):
        for key in ('model','state_dict','model_state_dict'):
            if key in loaded and isinstance(loaded[key], dict):
                loaded = loaded[key]
                break
        def strip(sd):
            out = {}
            for k,v in sd.items():
                newk = k
                for p in ('model.','module.'):
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


def evaluate_model(model, image_paths, labels, preprocess):
    device = next(model.parameters()).device
    correct = 0
    total = 0
    for p, lbl in zip(image_paths, labels):
        img = Image.open(p).convert('RGB')
        inp = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            probs = torch.nn.functional.softmax(out[0], dim=0)
            pred = int(torch.argmax(probs).item())
            correct += (pred == lbl)
            total += 1
    acc = correct/total if total else 0
    return acc


if __name__ == '__main__':
    # support a --full flag to evaluate on the whole Testing set
    full = '--full' in sys.argv
    max_per_class = 1000 if full else 3
    image_paths, labels = collect_test_images(max_per_class=max_per_class)
    print('Images used:', len(image_paths), '(full run)' if full else '')

    # collect models
    models_to_try = []
    root_best = os.path.join(BASE, 'best.pth')
    if os.path.exists(root_best):
        models_to_try.append(root_best)
    models_dir = os.path.join(BASE, 'backend', 'models')
    if os.path.isdir(models_dir):
        for f in os.listdir(models_dir):
            if f.lower().endswith('.pth') or f.lower().endswith('.pt'):
                models_to_try.append(os.path.join(models_dir, f))

    variants = [
        {'size':(224,224), 'normalize':True},
        {'size':(224,224), 'normalize':False},
        {'size':(128,128), 'normalize':True},
        {'size':(128,128), 'normalize':False},
    ]

    for v in variants:
        print('\nPreproc variant:', v)
        preprocess = make_preprocess(image_size=v['size'], normalize=v['normalize'])
        for mpath in models_to_try:
            try:
                mdl = load_model_from_path(mpath)
                acc = evaluate_model(mdl, image_paths, labels, preprocess)
                print(os.path.basename(mpath), '->', acc)
            except Exception as e:
                print('Failed', os.path.basename(mpath), 'err:', e)

# =============================================================================
#                Download test setova CIFAR10/100 Imagenet 10000
# =============================================================================

import os, tarfile, urllib.request, tempfile, shutil
from torchvision import datasets, transforms

ROOT = "./datasets"  # jedinstveni root
os.makedirs(ROOT, exist_ok=True)

# %%----------------------- CIFAR test delovi (po 10k) ------------------------

_basic_tfm = transforms.ToTensor()

cifar10_test  = datasets.CIFAR10(root=ROOT, train=False, download=True, transform=_basic_tfm)
cifar100_test = datasets.CIFAR100(root=ROOT, train=False, download=True, transform=_basic_tfm)

print("==================================================================")
print("CIFAR-10 test images:", len(cifar10_test))    # očekivano 10_000
print("CIFAR-100 test images:", len(cifar100_test))  # očekivano 10_000)
print("==================================================================")

# %%------- Imagenette val (≈3.9k) kao brza zamena za ImageNet val ------------

def download_imagenette_val_160(out_dir: str) -> str:
    """
    Preuzima samo 'val/' iz imagenette2-160.tgz i smešta u out_dir/imagenette_val/
    """
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    tmp = tempfile.mktemp(suffix=".tgz")
    print("[INFO] Downloading:", url)
    urllib.request.urlretrieve(url, tmp)

    with tarfile.open(tmp, "r:gz") as tar:
        members = tar.getmembers()
        if not members:
            raise RuntimeError("Arhiva je prazna.")
        top = members[0].name.split("/")[0]     # npr. 'imagenette2-160'
        val_prefix = f"{top}/val/"
        val_members = [m for m in members if m.name.startswith(val_prefix)]
        if not val_members:
            raise RuntimeError("Nisam našao 'val/' u arhivi.")

        target = os.path.join(out_dir, "imagenette_val")
        if os.path.exists(target):
            shutil.rmtree(target)
        os.makedirs(target, exist_ok=True)

        tmpdir = tempfile.mkdtemp(prefix="imagenette_")
        tar.extractall(path=tmpdir, members=val_members)
        src_val = os.path.join(tmpdir, val_prefix)
        for cls in os.listdir(src_val):
            shutil.move(os.path.join(src_val, cls), os.path.join(target, cls))
        shutil.rmtree(tmpdir)

    os.remove(tmp)
    print("==================================================================")
    print("[DONE] Imagenette val at:", target)
    print("==================================================================")
    return target

imagenette_val_root = download_imagenette_val_160(ROOT)

# (opciono) samo brzinski prebroj klasama/uzorcima:
imagenette_val = datasets.ImageFolder(imagenette_val_root, transform=_basic_tfm)
print("==================================================================")
print("Imagenette classes:", imagenette_val.classes)
print("Imagenette val images:", len(imagenette_val))
print("==================================================================")
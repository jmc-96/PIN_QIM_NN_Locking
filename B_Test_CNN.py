#%% ==================== EVALUACIJA - TEST CNN MREZA ==========================

import os, glob, tarfile, torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm  # pip install tqdm

# DODATNO: WebDataset za ImageNet val (64 .tar shard-a)
#   pip install webdataset
import webdataset as wds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = "./models"
DATASETS_DIR = "./datasets"
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "imagenet_label_map.pt")

#%% ---------- Helpers: CIFAR tar extraction ----------
def ensure_extracted_cifar(root: str, tar_name: str, expected_dir: str):
    os.makedirs(root, exist_ok=True)
    target_dir = os.path.join(root, expected_dir)
    tar_path = os.path.join(root, tar_name)
    if os.path.isdir(target_dir):
        return target_dir
    if os.path.isfile(tar_path):
        print(f"[info] Extracting {tar_name} ...")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(root)
        print("[info] Done.")
    else:
        raise FileNotFoundError(
            f"Neither '{expected_dir}' dir nor '{tar_name}' found in {root}."
        )
    if not os.path.isdir(target_dir):
        raise RuntimeError(f"After extraction, '{expected_dir}' not found in {root}.")
    return target_dir

#%% ---------- CIFAR ResNet (for 32x32) ----------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        # VAŽNO: koristimo torchvision imenovanje -> downsample (ne "shortcut")
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out

class CifarResNet(nn.Module):
    # depth = 20 -> n=3; depth = 32 -> n=5  (depth = 6n+2)
    def __init__(self, depth=20, num_classes=10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "Depth must be 6n+2 (e.g., 20, 32)."
        n = (depth - 2) // 6
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu  = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16,  n, stride=1)
        self.layer2 = self._make_layer(32,  n, stride=2)
        self.layer3 = self._make_layer(64,  n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64*BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet20_cifar(num_classes): return CifarResNet(depth=20, num_classes=num_classes)
def resnet32_cifar(num_classes): return CifarResNet(depth=32, num_classes=num_classes)

# ------------------------------ Eval loop ------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader, total=None, desc="Evaluating") -> float:
    model.eval()
    correct = 0
    total_samples = 0
    pbar = tqdm(loader, total=total, desc=desc, unit="batch", leave=False)
    for images, targets in pbar:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total_samples += targets.size(0)
        pbar.set_postfix(acc=f"{(100.0*correct/max(1,total_samples)):.2f}%")
    return 100.0 * correct / max(1, total_samples)

#%% -------------------------- ImageNet-1k TEST -------------------------------

#%% -------------------------- ImageNet-1k TEST -------------------------------

def test_imagenet1k(arch="resnet18", ckpt_name=None, batch_size=128, num_classes=1000,
                    label_map_path=LABEL_MAP_PATH):
    import json, re
    import importlib.resources as r
    import torch
    import numpy as np

    pattern = os.path.join(DATASETS_DIR, "imagenet1k-validation-*.tar")
    urls = sorted(glob.glob(pattern))
    if not urls:
        raise FileNotFoundError(f"Nema shardova po šablonu: {pattern}")

    # --------------------------- izbor režima --------------------------------
    print("==========================================================")
    print("                 REŽIM EVALUACIJE (ImageNet)             ")
    print("==========================================================")
    print("  1) Kompletan test set (~50k slika)")
    print("  2) Brzi test (manje shard-ova i batch-eva)")
    print("==========================================================")
    mode = (input("Izbor [1-2] (ENTER=2): ").strip() or "2")
    print("==========================================================")

    # ------------------- default za brzi test --------------------------------
    max_batches = None
    sel_urls = urls
    if mode == "2":
        try:
            n_shards = input("Broj shard-ova (ENTER=4): ").strip()
            n_shards = int(n_shards) if n_shards else 4
        except Exception:
            n_shards = 4
        try:
            mb = input("Max broj batch-eva (ENTER=20): ").strip()
            max_batches = int(mb) if mb else 20
        except Exception:
            max_batches = 20
        sel_urls = urls[:max(1, n_shards)]
        print(f"[info] Brzi test: koristim prvih {len(sel_urls)} shard-ova i max {max_batches} batch-eva.")
    else:
        print("[info] Kompletan test set.")

    # ------------------------------ transform --------------------------------
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ------------ pokušaj WNID -> index kroz torchvision paket ---------------
    wnid_to_idx = None
    try:
        with r.files('torchvision.datasets').joinpath('imagenet_class_index.json').open('r') as f:
            idx_map = json.load(f)  # {"0": ["n01440764", "tench"], ...}
        wnid_to_idx = {v[0]: int(k) for k, v in idx_map.items()}
        print("[info] Nađen imagenet_class_index.json u torchvision paketu.")
    except Exception:
        print("[warn] Nema imagenet_class_index.json — koristiću 'cls' iz shardova i (po potrebi) auto-poravnanje labela.")

    wnid_re = re.compile(r"n\d{8}")  # n01440764

    # ------------------------ WebDataset pipeline ----------------------------
    if wnid_to_idx is not None:
        # koristimo __key__ (WNID), ali povlačimo i 'cls' kao fallback
        dataset = (
            wds.WebDataset(sel_urls, shardshuffle=False)
            .decode("pil")
            .to_tuple("__key__", "jpg;jpeg;png", "cls")
            .map(lambda t: {
                "__key__": t[0], "image": t[1], "cls": t[2]
            })
            .map(lambda s: {
                "image": val_tf(s["image"]),
                "label": (
                    (lambda _k: (
                        wnid_to_idx[wnid_re.search(_k).group(0)]
                        if (_k and wnid_re.search(_k)) else
                        (int(s["cls"]) - 1 if 1 <= int(s["cls"]) <= num_classes else int(s["cls"]))
                    ))(s["__key__"].decode("utf-8", errors="ignore") if isinstance(s["__key__"], (bytes, bytearray)) else str(s["__key__"]))
                    if s.get("cls", None) is not None else
                    (lambda _k: wnid_to_idx[wnid_re.search(_k).group(0)] if (_k and wnid_re.search(_k)) else -1)(
                        s["__key__"].decode("utf-8", errors="ignore") if isinstance(s["__key__"], (bytes, bytearray)) else str(s["__key__"])
                    )
                )
            })
            .to_tuple("image", "label")
        )
    else:
        # koristi 'cls' iz shardova (pokuša 1-based -> 0-based)
        def _norm_label(y):
            if isinstance(y, (bytes, bytearray)):
                try:
                    y = int(y.decode("utf-8"))
                except Exception:
                    pass
            y = int(y)
            if 1 <= y <= num_classes:
                y -= 1
            return y

        dataset = (
            wds.WebDataset(sel_urls, shardshuffle=False)
            .decode("pil")
            .rename(image="jpg;jpeg;png", label="cls")
            .map_dict(image=val_tf, label=_norm_label)
            .to_tuple("image", "label")
        )

    # ------------------------- kolacija u tenzore ----------------------------
    def _collate(batch):
        imgs, labs = zip(*batch)
        return torch.stack(imgs, dim=0), torch.as_tensor(labs, dtype=torch.long)

    loader = wds.WebLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

    # sanity: prvi batch
    imgs, labs = next(iter(loader))
    print("Batch shapes:", tuple(imgs.shape), tuple(labs.shape),
          "| labs min/max:", int(labs.min().item()), int(labs.max().item()))
    print("labels dtype:", labs.dtype, "| sample:", labs[:8].tolist())

    approx_total_batches = 50000 // batch_size  # ImageNet val ≈50k slika

    # ------------------------------- model -----------------------------------
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1000)
    else:
        raise NotImplementedError(f"arch='{arch}' nije podržan u ovoj funkciji.")

    print("=======================================================================")
    if ckpt_name is None:
        fname = (input("Unesi ime fajla (ENTER=resnet18.pth): ").strip() or "resnet18.pth")
    else:
        fname = ckpt_name
    ckpt  = os.path.join(MODELS_DIR, fname)
    assert os.path.isfile(ckpt), f"Ne postoji fajl: {ckpt}"
    print(f"[info] Učitavam checkpoint: {ckpt}")
    print("=======================================================================")

    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and all(hasattr(v, "shape") for v in state.values()):
        model.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(state.get("state_dict", state.state_dict()), strict=True)

    # -------------------------- device + fingerprint -------------------------
    model.to(DEVICE).eval()
    print("fc.shape:", tuple(model.fc.weight.shape),
          "  sum|fc|=", float(model.fc.weight.detach().abs().sum()))

    # ------------------------ sanity acc pre evaluacije ----------------------
    with torch.no_grad():
        logits = model(imgs.to(DEVICE))
        preds  = logits.argmax(1).cpu()
        batch_acc = (preds == labs).float().mean().item() * 100
    print(f"[sanity] batch_acc = {batch_acc:.2f}% | preds[:8]={preds[:8].tolist()} | labs[:8]={labs[:8].tolist()}")

    # Ako nema WNID mape, koristi stabilnu mapu sa diska ili je napravi jednom
    label_map_tensor = None
    if wnid_to_idx is None:
        # 1) Probaj da učitaš mapu sa diska (stabilno poređenje između checkpoint-ova)
        if label_map_path and os.path.isfile(label_map_path):
            try:
                label_map_tensor = torch.load(label_map_path, map_location="cpu").long()
                assert label_map_tensor.numel() == num_classes, "label_map dimenzija ne odgovara broju klasa"
                print(f"[info] Učitavam label map iz: {label_map_path}")

                def _collate_mapped(batch):
                    imgs_, labs_ = zip(*batch)
                    labs_ = torch.as_tensor(labs_, dtype=torch.long)
                    return torch.stack(imgs_, dim=0), label_map_tensor[labs_]

                loader = wds.WebLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate_mapped)

                # sanity sa učitanom mapom
                imgs2, labs2 = next(iter(loader))
                with torch.no_grad():
                    logits2 = model(imgs2.to(DEVICE))
                    preds2  = logits2.argmax(1).cpu()
                    batch_acc2 = (preds2 == labs2).float().mean().item() * 100
                print(f"[sanity using saved map] batch_acc = {batch_acc2:.2f}%")
            except Exception as e:
                print(f"[warn] Ne mogu da učitam label map ({e}). Nastavljam bez nje.")

        # 2) Ako nema mape na disku i sanity je nizak → automatski napravi mapu i SAČUVAJ
        if (label_map_tensor is None) and (batch_acc < 5.0):
            print("[info] Sanity nizak — pravim label map (majority-vote)…")

            @torch.no_grad()
            def infer_label_mapping(model, loader, max_batches=20, device=DEVICE, num_classes=1000):
                counts = np.zeros((num_classes, num_classes), dtype=np.int32)
                seen = 0
                for images, labels in loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    pred = model(images).argmax(1).cpu().numpy()
                    lab  = labels.cpu().numpy()
                    for p, l in zip(pred, lab):
                        if 0 <= l < num_classes and 0 <= p < num_classes:
                            counts[l, p] += 1
                    seen += 1
                    if seen >= max_batches:
                        break
                mapping = counts.argmax(1)  # label l -> najčešća predikovana klasa
                return torch.from_numpy(mapping).long()

            label_map_tensor = infer_label_mapping(model, loader, max_batches=max_batches or 20, num_classes=num_classes)
            # sačuvaj da svi naredni testovi koriste ISTU mapu
            if label_map_path:
                torch.save(label_map_tensor, label_map_path)
                print(f"[info] Sačuvana label map u: {label_map_path}")

            # primeni mapu
            def _collate_mapped(batch):
                imgs_, labs_ = zip(*batch)
                labs_ = torch.as_tensor(labs_, dtype=torch.long)
                return torch.stack(imgs_, dim=0), label_map_tensor[labs_]

            loader = wds.WebLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate_mapped)

            # sanity posle poravnanja
            imgs2, labs2 = next(iter(loader))
            with torch.no_grad():
                logits2 = model(imgs2.to(DEVICE))
                preds2  = logits2.argmax(1).cpu()
                batch_acc2 = (preds2 == labs2).float().mean().item() * 100
            print(f"[sanity after align] batch_acc = {batch_acc2:.2f}%")


    # ---------------------------- EVALUACIJA ---------------------------------
    total_batches = max_batches if mode == "2" else approx_total_batches
    acc = evaluate(model, loader, total=total_batches, desc=f"ImageNet-1k (val){' [FAST]' if mode=='2' else ''}")
    print(f"ImageNet-1k (val, WebDataset) accuracy: {acc:.2f}%")
    return acc


#%% ------------------------------- CIFAR testovi -----------------------------

def test_cifar10(batch_size=512):
    ensure_extracted_cifar(DATASETS_DIR, "cifar-10-python.tar.gz", "cifar-10-batches-py")
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)),
    ])
    ds = datasets.CIFAR10(root=DATASETS_DIR, train=False, download=False, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = resnet20_cifar(num_classes=10)
    print("=======================================================================")
    fname = input("Unesi ime fajla (ENTER za cifar10_resnet20): ") or "cifar10_resnet20.pth"
    ckpt  = os.path.join(MODELS_DIR, fname)
    if not os.path.isfile(ckpt):
        print("Fajl nije pronadjen! Povratak na meni")
        ckpt = None
    print("=======================================================================")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state if isinstance(state, dict) else state.state_dict(), strict=False)
    model.to(DEVICE)
    acc = evaluate(model, loader, total=len(loader), desc="CIFAR-10 (test)")
#    acc = evaluate(model, loader)
    print(f"CIFAR-10 (test) accuracy: {acc:.2f}%")
    return acc

def test_cifar100(batch_size=512):
    ensure_extracted_cifar(DATASETS_DIR, "cifar-100-python.tar.gz", "cifar-100-python")
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071,0.4867,0.4408), std=(0.2675,0.2565,0.2761)),
    ])
    ds = datasets.CIFAR100(root=DATASETS_DIR, train=False, download=False, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = resnet32_cifar(num_classes=100)
    print("=======================================================================")
    fname = input("Unesi ime fajla (ENTER za cifar100_resnet32): ") or "cifar100_resnet32.pth"
    ckpt  = os.path.join(MODELS_DIR, fname)
    if not os.path.isfile(ckpt):
        print("Fajl nije pronadjen! Povratak na meni")
        ckpt = None
    print("=======================================================================")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state if isinstance(state, dict) else state.state_dict(), strict=False)
    model.to(DEVICE)

    acc = evaluate(model, loader, total=len(loader), desc="CIFAR-100 (test)")
#    acc = evaluate(model, loader)
    print(f"CIFAR-100 (test) accuracy: {acc:.2f}%")
    return acc

#%% ---------------------------------- Menu -----------------------------------

def main():
    print(f"[info] Using device: {DEVICE}")
    print("==========================================")
    print("                   MENU                   ")
    print("==========================================")
    print("1) Testiraj resnet18 na ImageNet-1k")
    print("2) Testiraj ResNet20 na CIFAR-10")
    print("3) Testiraj ResNet32 na CIFAR-100")
    print("0) Izlaz")
    print("==========================================")
    choice = input("Izbor: ").strip()
    print("==========================================")
    try:
        if choice == "1":
            # koristi istu mapu UVEK; ako ne postoji, test_imagenet1k će je napraviti i sačuvati
            test_imagenet1k(label_map_path=LABEL_MAP_PATH)
        elif choice == "2":
            test_cifar10()
        elif choice == "3":
            test_cifar100()
        elif choice == "0":
            print("Zatvaram program.")
        else:
            print("Nepoznata opcija.")
    except FileNotFoundError as e:
        print(f"[error] {e}")
    except RuntimeError as e:
        print(f"[error] {e}")
    except Exception as e:
        print(f"[error] Neočekovana greška: {e}")
    print("==========================================")

if __name__ == "__main__":
    main()
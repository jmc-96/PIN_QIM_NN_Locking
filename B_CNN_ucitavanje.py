# %% UNIVERZALNI MENU ZA UCITAVANJE IMAGENET I OBA CIFAR (CIFAR online)

import os, torch
from collections import OrderedDict

# %% SPISAK MODELA

IMAGENET_MODELS = [
    "resnet18",
    "resnet50",
    "mobilenet_v2",
    "efficientnet_b0",
    "vit_b_16",
]
CIFAR10_ARCHES = [
    "cifar10_resnet20",
    "cifar10_resnet32",
    "cifar10_vgg11_bn",
    "cifar10_mobilenetv2_x1_0",
    "cifar10_shufflenet_v2_x1_0",
    "cifar10_repvgg_a0",
]
CIFAR100_ARCHES = [
    "cifar100_resnet20",
    "cifar100_resnet32",
    "cifar100_vgg11_bn",
    "cifar100_mobilenetv2_x1_0",
    "cifar100_shufflenet_v2_x1_0",
    "cifar100_repvgg_a0",
]

# %% Zajednicki

def prefix_modula(sd: "OrderedDict[str, torch.Tensor]"):
    out = OrderedDict()
    for k, v in sd.items():
        out[k.replace("module.", "", 1)] = v
    return out

def arhitektura_imagenet(name: str):
    import torchvision.models as models
    name = name.lower()
    if   name == "resnet18":        return models.resnet18(weights=None)
    elif name == "resnet50":        return models.resnet50(weights=None)
    elif name == "mobilenet_v2":    return models.mobilenet_v2(weights=None)
    elif name == "efficientnet_b0": return models.efficientnet_b0(weights=None)
    elif name == "vit_b_16":        return models.vit_b_16(weights=None)
    else: raise ValueError(f"Nepoznata ImageNet arhitektura: {name}")

def arhitektura_cifar(arch: str): 
    # online Torch Hub, bez pretreniranja (arhitektura samo)
    return torch.hub.load("chenyaofo/pytorch-cifar-models", arch, pretrained=False)

# %% Ucitavanje modela

def ucitavanje_modela(models_root="."):
    # Prvi MENU
    print("==========================================================")
    print("                       IZABERI DATASET                    ")
    print("==========================================================")
    datasets = ["ImageNet-1K", "CIFAR-10 (online)", "CIFAR-100 (online)", "Odustani"]
    for i, n in enumerate(datasets):
        print(f"  {i}) {n}")
    print("==========================================================")
    try:
        ds_idx = int(input(f"Izbor [0-{len(datasets)-1}]: ").strip())
        assert 0 <= ds_idx < len(datasets)
        print("==========================================================")
    except Exception:
        print("Nevažeći izbor."); return None, None, None

    if ds_idx == 3:
        print("Prekid."); return None, None, None

    # Izbor arhitekture
    arches = IMAGENET_MODELS if ds_idx == 0 else (CIFAR10_ARCHES if ds_idx == 1 else CIFAR100_ARCHES)
    print("==========================================================")
    print("                    IZABERI ARHITEKTURU                   ")
    print("==========================================================")
    for i, n in enumerate(arches):
        print(f"  {i}) {n}")
    try:
        a_idx = int(input(f"Izbor [0-{len(arches)-1}]: ").strip())
        assert 0 <= a_idx < len(arches)
        print("==========================================================")
    except Exception:
        print("Nevažeći izbor."); return None, None, None

    arch = arches[a_idx]

    # ===== DODATO: opcioni unos imena fajla / putanje =====
    models_dir = "models"
    podrazumevano_ime = f"{arch}.pth"
    unos = input(f"Ime fajla (.pth) [Enter za {podrazumevano_ime}]: ").strip()

    if unos == "":
        ime = podrazumevano_ime
        putanja = os.path.join(models_dir, ime)
    else:
        # Ako je apsolutna putanja – koristi direktno; u suprotnom traži u models/
        if os.path.isabs(unos):
            putanja = unos
            ime = os.path.basename(unos)
        else:
            putanja = os.path.join(models_dir, unos)
            ime = os.path.basename(unos)
    # ===== kraj dodatka =====

    if not os.path.isfile(putanja):
        print("Taj model ne postoji, izaberite drugi iz spiska ili pokrenite ponovo download.")
        print("Očekivano ime fajla:", ime)
        print("Folder:", os.path.abspath(models_dir))
        print("===============================================================================")
        return None, None, None

    # Učitavamo .pth sadržaj
    obj = torch.load(putanja, map_location="cpu")
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        sd = obj
    elif hasattr(obj, "state_dict"):
        sd = obj.state_dict()
    else:
        print("Nepoznat format .pth (nije ni state_dict ni torch.nn.Module).")
        return None, None, None

    # kreiramo arhitekturu
    try:
        if ds_idx == 0:
            model = arhitektura_imagenet(arch)      # torchvision
        else:
            model = arhitektura_cifar(arch)         # torch.hub online
    except Exception as e:
        print("Greška pri kreiranju arhitekture:", repr(e))
        print("==========================================================")
        return None, None, None

    # Učitamo težine (strict), fallback bez 'module.' prefiksa
    try:
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        try:
            sd2 = prefix_modula(sd)
            model.load_state_dict(sd2, strict=True)
            sd = sd2
        except Exception as e2:
            print("Greška pri load_state_dict (strict=True):")
            print("  original:", repr(e))
            print("  bez 'module.' prefiksa:", repr(e2))
            print("======================================================================")
            return None, None, None

    model.eval()
    print("[OK] Učitano:", putanja, "| arhitektura:", arch)
    print("======================================================================")
    return sd, ime, model


# =============================================================================
#                                  DOWNLOAD CNN
# =============================================================================

# Create a universal menu-based downloader script for ImageNet, CIFAR-10, CIFAR-100
# Univerzalni skript sa menijem za preuzimanje pretreniranych modela i čuvanje u ORIGINALNOM
# PyTorch formatu: state_dict (.pth) + opciono TorchScript (.pt) + opciono Meta (.json).

# Podržano:
#  • ImageNet-1K (torchvision): resnet18, resnet50, mobilenet_v2, efficientnet_b0, vit_b_16
#  • CIFAR-10 (torch.hub chenyaofo/pytorch-cifar-models): nekoliko popularnih arhitektura
#  • CIFAR-100 (torch.hub chenyaofo/pytorch-cifar-models): nekoliko popularnih arhitektura

import os
import sys
import time
from typing import Dict, Callable, Tuple
import json
import torch
import torch.nn as nn

# %% ------------------------- ImageNet (torchvision) -------------------------

IMAGENET_MODELS = [
    "resnet18",
    "resnet50",
    "mobilenet_v2",
    "efficientnet_b0",
    "vit_b_16",
]

def get_imagenet_model(name: str) -> nn.Module:
    import torchvision.models as models
    name = name.lower()
    if name == "resnet18":
        try:  return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception: return models.resnet18(pretrained=True)
    if name == "resnet50":
        try:  return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except Exception: return models.resnet50(pretrained=True)
    if name == "mobilenet_v2":
        try:  return models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        except Exception: return models.mobilenet_v2(pretrained=True)
    if name == "efficientnet_b0":
        try:  return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        except Exception: return models.efficientnet_b0(pretrained=True)
    if name == "vit_b_16":
        try:  return models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        except Exception: return models.vit_b_16(pretrained=True)
    raise ValueError(f"Nepoznata ImageNet arhitektura: {name}")

# %% ------------------- CIFAR-10 / CIFAR-100 (torch.hub) ---------------------

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

def get_cifar_model(arch: str) -> nn.Module:
    # chenyaofo/pytorch-cifar-models (pretrained=True)
    return torch.hub.load("chenyaofo/pytorch-cifar-models", arch, pretrained=True)

# %% --------------------------------- Helpers --------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def count_bias_tensors(model: nn.Module) -> int:
    return sum(1 for n, p in model.named_parameters() if "bias" in n)

def save_state_dict(model: nn.Module, out_dir: str, basename: str) -> str:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{basename}.pth")
    torch.save(model.state_dict(), path)
    print(f"[OK] Sačuvan state_dict: {path}  ({os.path.getsize(path)/1e6:.2f} MB)")
    print("-------------------------------------------------------------------------------")
    return path

def save_torchscript(model: nn.Module, out_dir: str, basename: str, shape: Tuple[int, int, int]):
    ensure_dir(out_dir)
    example = torch.randn(1, *shape)
    try:
        with torch.no_grad():
            try:
                scripted = torch.jit.trace(model.eval().cpu(), example, strict=False)
            except Exception:
                scripted = torch.jit.script(model.eval().cpu())
        path = os.path.join(out_dir, f"{basename}.pt")
        scripted.save(path)
        print(f"[OK] Sačuvan TorchScript: {path}  ({os.path.getsize(path)/1e6:.2f} MB)")
        print("-------------------------------------------------------------------------------")
        return path
    except Exception as e:
        print(f"[WARN] TorchScript export nije uspeo: {e}")
        print("-------------------------------------------------------------------------------")
        return ""

def save_Meta(model, out_dir, basename, dataset, arch):
    meta = {
        "dataset": dataset,
        "arch": arch,
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "bias_tensors": sum(1 for n,_ in model.named_parameters() if "bias" in n),
        "pytorch_versions": {"torch": torch.__version__},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import torchvision
        meta["pytorch_versions"]["torchvision"] = torchvision.__version__
    except Exception:
        pass
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, f"{basename}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[OK] Sačuvan meta fajl:", meta_path)
    print("==========================================================================")
    return ""

def choose(prompt: str, options: list) -> int:
    while True:
        try:
            idx = int(input(prompt).strip())
            if 0 <= idx < len(options):
                return idx
        except Exception:
            pass
        print("❗ Nevažeći izbor, probaj ponovo.")
        print("==========================================================================")

# %% ----------------------------------- Menu ---------------------------------

MAIN_MENU = [
    "ImageNet-1K",
    "CIFAR-10",
    "CIFAR-100",
    "Izlaz",
]

def main():
    print("===============================================================================")
    print("="*60)
    print("  Univerzalni downloader modela (ImageNet / CIFAR-10 / CIFAR-100)")
    print("="*60)
    print("\nGlavni meni:")
    print("===============================================================================")
    for i, item in enumerate(MAIN_MENU):
        print(f"  {i}) {item}")
    print("===============================================================================")
    choice = choose("\nIzbor [0-3]: ", MAIN_MENU)

    if choice == 0:
        # ImageNet
        print("\nImageNet arhitekture:")
        print("-------------------------------------------------------------------------------")
        for i, name in enumerate(IMAGENET_MODELS):
            print(f"  {i}) {name}")
        idx = choose(f"Izbor [0-{len(IMAGENET_MODELS)-1}]: ", IMAGENET_MODELS)
        name = IMAGENET_MODELS[idx]
        out_dir = input("Output folder [models]: ").strip() or "models"
        want_ts = (input("Sačuvati i TorchScript? [y/N]: ").strip().lower() == "y")
        want_meta = (input("Sačuvati i Meta podatke? [y/N]: ").strip().lower() == "y")
        print(f"\n[*] Učitavam {name} (ImageNet, torchvision) ...")
        t0 = time.time()
        model = get_imagenet_model(name)
        model.eval()
        print(f"[i] Gotovo za {time.time()-t0:.2f}s | params={count_params(model):,} | bias_tensors={count_bias_tensors(model)}")
        print("===============================================================================")
        base = name.lower()
        save_state_dict(model, out_dir, base)
        if want_ts:
            save_torchscript(model, out_dir, base, (3, 224, 224))
        if want_meta:
            save_Meta(model, out_dir, base, "ImageNet-1K", name)
    elif choice == 1:
        # CIFAR-10
        print("\nCIFAR-10 arhitekture:")
        print("-------------------------------------------------------------------------------")
        for i, name in enumerate(CIFAR10_ARCHES):
            print(f"  {i}) {name}")
        idx = choose(f"Izbor [0-{len(CIFAR10_ARCHES)-1}]: ", CIFAR10_ARCHES)
        arch = CIFAR10_ARCHES[idx]
        out_dir = input("Output folder [models]: ").strip() or "models"
        want_ts = (input("Sačuvati i TorchScript? [y/N]: ").strip().lower() == "y")
        want_meta = (input("Sačuvati i Meta podatke? [y/N]: ").strip().lower() == "y")
        print(f"\n[*] Učitavam {arch} (torch.hub) ...")
        t0 = time.time()
        model = get_cifar_model(arch)
        model.eval()
        print(f"[i] Gotovo za {time.time()-t0:.2f}s | params={count_params(model):,} | bias_tensors={count_bias_tensors(model)}")
        print("===============================================================================")
        base = arch
        save_state_dict(model, out_dir, base)
        if want_ts:
            save_torchscript(model, out_dir, base, (3, 32, 32))
        if want_meta:
            save_Meta(model, out_dir, base, "CIFAR-10", arch)
    elif choice == 2:
        # CIFAR-100
        print("\nCIFAR-100 arhitekture:")
        print("-------------------------------------------------------------------------------")
        for i, name in enumerate(CIFAR100_ARCHES):
            print(f"  {i}) {name}")
        idx = choose(f"Izbor [0-{len(CIFAR100_ARCHES)-1}]: ", CIFAR100_ARCHES)
        arch = CIFAR100_ARCHES[idx]
        out_dir = input("Output folder [models]: ").strip() or "models"
        want_ts = (input("Sačuvati i TorchScript? [y/N]: ").strip().lower() == "y")
        want_meta = (input("Sačuvati i Meta podatke? [y/N]: ").strip().lower() == "y")
        print(f"\n[*] Učitavam {arch} (torch.hub) ...")
        t0 = time.time()
        model = get_cifar_model(arch)
        model.eval()
        print(f"[i] Gotovo za {time.time()-t0:.2f}s | params={count_params(model):,} | bias_tensors={count_bias_tensors(model)}")
        print("===============================================================================")
        base = arch
        save_state_dict(model, out_dir, base)
        if want_ts:
            save_torchscript(model, out_dir, base, (3, 32, 32))
        if want_meta:
            save_Meta(model, out_dir, base, "CIFAR-10", arch)           
    else:
        print("-------------------------------------------------------------------------------")
        print("Pozdrav!")
        print("===============================================================================")
        return

    print("\nDone. Fajlovi su u zadatom izlaznom folderu.")
    print("===============================================================================")

if __name__ == "__main__":
    main()
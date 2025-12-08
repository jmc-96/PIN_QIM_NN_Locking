# %% STRICT PROVERA PRE SNIMANJA (i snimanje samo ako prođe)
''' Ne menja model, vec samo snima strukturu i vrednosti koeficijenata'''

import os, torch
from collections import OrderedDict

IMAGENET = {"resnet18","resnet50","mobilenet_v2","efficientnet_b0","vit_b_16"}

def _infer_arch(ime: str) -> str:
    return os.path.splitext(os.path.basename(ime))[0]  # npr. "resnet50" ili "cifar100_resnet32"

def arhitektura_imagenet(arch: str):
    import torchvision.models as models
    builders = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "mobilenet_v2": models.mobilenet_v2,
        "efficientnet_b0": models.efficientnet_b0,
        "vit_b_16": models.vit_b_16,
    }
    return builders[arch](weights=None)

def arhitektura_cifar(arch: str):
    # zahteva internet (repo: chenyaofo/pytorch-cifar-models)
    return torch.hub.load("chenyaofo/pytorch-cifar-models", arch, pretrained=False)

def _strip_module(sd: "OrderedDict[str, torch.Tensor]"):
    return OrderedDict((k.replace("module.", "", 1), v) for k, v in sd.items())

def strict_validate(sd, ime: str):
    """
    Vraća (ok, model, sd_used).
    Pokušava strict=True load u tačnu arhitekturu (ImageNet iz torchvision, CIFAR online).
    Ako padne zbog 'module.' prefiksa, proba bez prefiksa.
    """
    arch = _infer_arch(ime)
    if arch in IMAGENET:
        model = arhitektura_imagenet(arch)
    elif arch.startswith("cifar10_") or arch.startswith("cifar100_"):
        model = arhitektura_cifar(arch)
    else:
        raise ValueError(f"Nepoznata arhitektura iz imena: {arch}")

    try:
        model.load_state_dict(sd, strict=True)
        return True, model.eval().cpu(), sd
    except Exception as e1:
        try:
            sd2 = _strip_module(sd)
            model.load_state_dict(sd2, strict=True)
            return True, model.eval().cpu(), sd2
        except Exception as e2:
            print("==========================================================")
            print("STRICT VALIDACIJA NIJE PROŠLA.")
            print("Greška (original):", repr(e1))
            print("Greška (bez 'module.' prefiksa):", repr(e2))
            print("==========================================================")
            return False, None, None

def snimi_sd_u_pth(sd, ime: str, ime_LW, out_dir="models"):
    """
    Radi strict validaciju; ako prođe, snima <out_dir>/<base>_L.pth i vraća putanju.
    Ako ne prođe, vraća None i ne snima ništa.
    """
    ok, model, sd_used = strict_validate(sd, ime)
    if not ok:
        return None
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(ime))[0]
    out_path = os.path.join(out_dir, f"{base}{ime_LW}.pth")
    torch.save(sd_used, out_path)
    print("=============================================================================")
    print("[OK] Strict validacija prošla. Sačuvan state_dict:", out_path)
    print("=============================================================================")
    return out_path

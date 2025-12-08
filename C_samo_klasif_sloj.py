import torch
import numpy as np
from copy import deepcopy

# ——— pronalazak ključa završnog klasifikatora u state_dict-u
def _find_fc_key(sd):
    # prvo probaj tipične nazive
    preferred = (
        "fc.weight", "module.fc.weight",          # torchvision resnet / DP
        "classifier.weight", "head.weight", "linear.weight"
    )
    for k in sd.keys():
        if any(p in k and k.endswith("weight") for p in preferred):
            return k
    # fallback: poslednji 2D parametar ili 1x1 conv kao "glava"
    two_d = [k for k in sd.keys() if sd[k].ndim == 2 and k.endswith("weight")]
    if two_d:
        return two_d[-1]
    four_d_1x1 = [k for k in sd.keys()
                  if sd[k].ndim == 4 and sd[k].shape[2:] == (1, 1) and k.endswith("weight")]
    if four_d_1x1:
        return four_d_1x1[-1]
    raise KeyError("Nisam našao fc.weight kandidat u state_dict-u.")

# ——— ekstrakcija u NumPy 2D + meta info (ključ, originalni shape, dtype)
def extract_fc_numpy(sd_or_model):
    sd = sd_or_model.state_dict() if hasattr(sd_or_model, "state_dict") else sd_or_model
    fc_key = _find_fc_key(sd)
    W = sd[fc_key].detach().cpu().clone()       # tensor na CPU
    orig_shape = tuple(W.shape)
    dtype = W.dtype

    # pretvori u 2D: [out, in_flat]
    if W.ndim == 2:
        W2d = W.numpy()
    elif W.ndim == 4 and W.shape[2:] == (1, 1):  # 1×1 conv "glava"
        W2d = W.view(W.shape[0], -1).numpy()
    else:
        raise ValueError(f"Neočekivan shape za fc težine: {orig_shape}")

    return fc_key, W2d, orig_shape, dtype

# ——— vraćanje iz NumPy 2D u tensor i upis u state_dict (kopija ili in-place za model)

def build_sd_re_with_locked_fc(sd_or_model, fc_key, W2d_locked: np.ndarray, orig_shape, ensure_cpu=True):
    """
    Vrati novi state_dict (sd_re) u koji je upisan W2d_locked (NumPy 2D) 
    vraćen na originalni shape i dtype, spreman za torch.save.
    """
    # Polazni sd (iz modela ili već postojeći sd)
    base_sd = sd_or_model.state_dict() if hasattr(sd_or_model, "state_dict") else sd_or_model

    # Napravi kopiju za snimanje (po difoltu na CPU radi portabilnosti)
    sd_re = {k: (v.detach().cpu().clone() if ensure_cpu else v.clone()) for k, v in base_sd.items()}

    # Rekonstruiši fc.weight iz NumPy u originalni shape i dtype
    t_fc = torch.from_numpy(W2d_locked).to(dtype=sd_re[fc_key].dtype)
    t_fc = t_fc.view(orig_shape).contiguous()
    if ensure_cpu:
        t_fc = t_fc.cpu()

    # Upis u sd_re
    sd_re[fc_key] = t_fc
    return sd_re

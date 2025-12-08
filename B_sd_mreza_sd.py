# ========================= RAZVRSTAVANJE & SCHEMA ============================
#=== ==========================================================================

from typing import Dict, Tuple, List, Any
import re
import torch
from collections import OrderedDict as COD
from typing import OrderedDict as TOD
import numpy as np

Token = Dict[str, Any]  # {"key": str, "shape": List[int], "dtype": str, "target": str}

_re_running = re.compile(
    r"(?:^|[._])running_(?:mean|var)(?:$|[._])|(?:^|[._])num_batches_tracked(?:$|[._])"
)
_re_bias   = re.compile(r"(?:^|[._])(bias|beta)(?:$|[._])")
_re_weight = re.compile(r"(?:^|[._])(weight|gamma|kernel|embedding|embeddings)(?:$|[._])")

def _tensor_meta(name: str, t: torch.Tensor, target: str) -> Token:
    return {
        "key": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "target": target,
    }

# %% ==========================================================================
#         RAZVRSTAVANJE sd NA WEIGHTS, BIASES, SCALARS i RUNNINGS
# =============================================================================

# 1)  --------- sd -> (sd_weights, sd_biases, sd_scalars, sd_runns) -----------  

def razvrstaj_sd(
    sd: Dict[str, torch.Tensor]
) -> Tuple[Tuple[List[np.ndarray], List[np.ndarray], List[Any], List[np.ndarray]], List[str]]:
    sd_weights, sd_biases, sd_scalars, sd_runns = {}, {}, {}, {}
    sd_schema_W, sd_schema_b, sd_schema_s, sd_schema_r = [], [], [], []
    global_order = list(sd.keys())

    for name, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim == 0 or (tensor.ndim == 1 and tensor.numel() == 1):
            sd_scalars[name] = tensor
            sd_schema_s.append(_tensor_meta(name, tensor, "scalars"))
        elif _re_running.search(name):
            sd_runns[name] = tensor
            sd_schema_r.append(_tensor_meta(name, tensor, "runns"))
        elif name.endswith(".bias") or _re_bias.search(name):
            sd_biases[name] = tensor
            sd_schema_b.append(_tensor_meta(name, tensor, "biases"))
        elif name.endswith(".weight") or _re_weight.search(name):
            sd_weights[name] = tensor
            sd_schema_W.append(_tensor_meta(name, tensor, "weights"))
        else:
            sd_weights[name] = tensor
            meta = _tensor_meta(name, tensor, "weights")
            meta["note"] = "fallback_weights"
            sd_schema_W.append(meta)

    # >>> OVO MORA VAN PETLJE <<<
    fc_pos = next(
    (i for i, entry in enumerate(sd_schema_W) if entry["key"] == "fc.weight"),
    None
)
    
    mreza = sd_2_mreza(
        sd_weights, sd_biases, sd_scalars, sd_runns,
        sd_schema_W, sd_schema_b, sd_schema_s, sd_schema_r
    )
    return mreza, global_order, fc_pos

# 2) --------------------- DICT -> mreza (NumPy/list) -------------------------

def sd_2_mreza(
    sd_W, sd_b, sd_s, sd_r,
    sd_schema_W, sd_schema_b, sd_schema_s, sd_schema_r
):
    """
    Iz dict-ova sa PyTorch tenzorima pravi tuple listi NumPy objekata:
      - list_W: [ (out,-1) 2D np.ndarray ... ]
      - list_b: [ (n,1) np.ndarray ... ]
      - list_s: [ skalar (np.generic ili python broj) ... ]
      - list_r: [ (n,1) np.ndarray ... ]
    Redosled je određen redosledom u odgovarajućoj šemi (schema listama).
    """
    list_W, list_b, list_s, list_r = [], [], [], []

    # Weights -> 2D
    for entry in sd_schema_W:
        k = entry["key"]
        t = sd_W[k]
        arr = t.detach().cpu().numpy().reshape(t.shape[0], -1)
        list_W.append(arr)

    # Biases -> (n,1)
    for entry in sd_schema_b:
        k = entry["key"]
        t = sd_b[k]
        arr = t.detach().cpu().numpy().reshape(-1, 1)
        list_b.append(arr)

    # Scalars -> skalar vrednosti
    for entry in sd_schema_s:
        k = entry["key"]
        t = sd_s[k]
        # čuvamo kao python broj (ili np scalar) — oba su ok
        val = t.detach().cpu().item()
        list_s.append(val)

    # Runnings -> (n,1)
    for entry in sd_schema_r:
        k = entry["key"]
        t = sd_r[k]
        arr = t.detach().cpu().numpy().reshape(-1, 1)
        list_r.append(arr)
        
    mreza = (list_W, list_b, list_s, list_r)
    return mreza
    
# %% ==========================================================================
#                            REKONSTRUKCIJA
# =============================================================================

# 1) ----------------- mreza (NumPy/list) -> DICT (PyTorch) -------------------

def _torch_dtype(dtype_str: str):
    # npr. "float32", "float16", "bfloat16", "int64", ...
    return getattr(torch, dtype_str)


def citaj_cheme(
    sd: Dict[str, torch.Tensor]
) -> Tuple[List[Token], List[Token], List[Token], List[Token], List[str]]:
    sd_schema_W, sd_schema_b, sd_schema_s, sd_schema_r = [], [], [], []
    global_order = list(sd.keys())

    for name, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim == 0 or (tensor.ndim == 1 and tensor.numel() == 1):
            sd_schema_s.append(_tensor_meta(name, tensor, "scalars"))
        elif _re_running.search(name):
            sd_schema_r.append(_tensor_meta(name, tensor, "runns"))
        elif name.endswith(".bias") or _re_bias.search(name):
            sd_schema_b.append(_tensor_meta(name, tensor, "biases"))
        elif name.endswith(".weight") or _re_weight.search(name):
            sd_schema_W.append(_tensor_meta(name, tensor, "weights"))
        else:
            meta = _tensor_meta(name, tensor, "weights")
            meta["note"] = "fallback_weights"
            sd_schema_W.append(meta)

    return sd_schema_W, sd_schema_b, sd_schema_s, sd_schema_r, global_order

def mreza_2_sd(mreza, sd, device="cpu", strict_shapes: bool = True):
    sd_schema_W, sd_schema_b, sd_schema_s, sd_schema_r, global_order = citaj_cheme(sd)
    list_W, list_b, list_s, list_r = mreza
    sd_W, sd_b, sd_s, sd_r = {}, {}, {}, {}

    # Weights: 2D -> originalni shape
    if strict_shapes and len(list_W) != len(sd_schema_W):
        raise ValueError("Dužina list_W ne odgovara sd_schema_W.")
    for entry, arr in zip(sd_schema_W, list_W):
        key = entry["key"]
        exp_shape = entry["shape"]
        dtype = _torch_dtype(entry["dtype"])
        t = torch.tensor(arr, dtype=dtype, device=device).reshape(exp_shape)
        sd_W[key] = t

    # Biases: (n,1) -> (n,)
    if strict_shapes and len(list_b) != len(sd_schema_b):
        raise ValueError("Dužina list_b ne odgovara sd_schema_b.")
    for entry, arr in zip(sd_schema_b, list_b):
        key = entry["key"]
        exp_shape = entry["shape"]
        dtype = _torch_dtype(entry["dtype"])
        t = torch.tensor(np.asarray(arr).squeeze(-1), dtype=dtype, device=device).reshape(exp_shape)
        sd_b[key] = t

    # Scalars: skalar -> () ili (1,) kako je u šemi
    if strict_shapes and len(list_s) != len(sd_schema_s):
        raise ValueError("Dužina list_s ne odgovara sd_schema_s.")
    for entry, val in zip(sd_schema_s, list_s):
        key = entry["key"]
        exp_shape = entry["shape"]  # [] ili [1]
        dtype = _torch_dtype(entry["dtype"])
        t = torch.tensor(val, dtype=dtype, device=device)
        if len(exp_shape) == 0:
            # shape == ()
            sd_s[key] = t
        else:
            # npr. (1,)
            sd_s[key] = t.reshape(exp_shape)

    # Runnings: (n,1) -> (n,)
    if strict_shapes and len(list_r) != len(sd_schema_r):
        raise ValueError("Dužina list_r ne odgovara sd_schema_r.")
    for entry, arr in zip(sd_schema_r, list_r):
        key = entry["key"]
        exp_shape = entry["shape"]
        dtype = _torch_dtype(entry["dtype"])
        t = torch.tensor(np.asarray(arr).squeeze(-1), dtype=dtype, device=device).reshape(exp_shape)
        sd_r[key] = t
    sd_re = rekonstruisi_sd(
        sd_W, sd_b, sd_s, sd_r,
        sd_schema_W, sd_schema_b, sd_schema_s, sd_schema_r,
        strict_shapes=strict_shapes,
        global_order=global_order)
    
    return sd_re

# 2) --------------- DICS (PyTorch tensors) -> sd (OrderedDict) ---------------

def rekonstruisi_sd(
    sd_weights: Dict[str, torch.Tensor],
    sd_biases: Dict[str, torch.Tensor],
    sd_scalars: Dict[str, torch.Tensor],
    sd_runns: Dict[str, torch.Tensor],
    sd_schema_W: List[Token],
    sd_schema_b: List[Token],
    sd_schema_s: List[Token],
    sd_schema_r: List[Token],
    strict_shapes: bool = True,
    global_order: List[str] = None,
) -> TOD[str, torch.Tensor]:
    """
    Rekonstruiše originalni state_dict po ključevima i očekivanim oblicima iz šema.
    VRAĆA OrderedDict i poštuje global_order ako je dat.
    """
    sd_tmp: Dict[str, torch.Tensor] = {}

    def _restore(block: Dict[str, torch.Tensor], schema: List[Token], label: str):
        for entry in schema:
            k = entry["key"]
            if k not in block:
                raise KeyError(f"[{label}] nedostaje ključ: {k}")
            t = block[k]
            if strict_shapes:
                exp_shape = tuple(entry["shape"])
                if tuple(t.shape) != exp_shape:
                    raise ValueError(f"[{label}] shape mismatch za {k}: {tuple(t.shape)} != {exp_shape}")
                exp_dtype = entry["dtype"]
                if str(t.dtype).replace("torch.", "") != exp_dtype:
                    raise TypeError(f"[{label}] dtype mismatch za {k}: {t.dtype} != torch.{exp_dtype}")
            sd_tmp[k] = t

    _restore(sd_weights, sd_schema_W, "weights")
    _restore(sd_biases,  sd_schema_b, "biases")
    _restore(sd_scalars, sd_schema_s, "scalars")
    _restore(sd_runns,   sd_schema_r, "runns")

    if global_order is not None:
        return COD((k, sd_tmp[k]) for k in global_order if k in sd_tmp)

    seq = [
        *[e["key"] for e in sd_schema_W],
        *[e["key"] for e in sd_schema_b],
        *[e["key"] for e in sd_schema_s],
        *[e["key"] for e in sd_schema_r],
    ]
    return COD((k, sd_tmp[k]) for k in seq if k in sd_tmp)

# %% ==========================================================================
#                                  API
#==============================================================================

def sd_u_mrezu(sd):
    mreza, _, fc_pos = razvrstaj_sd(sd)
    return mreza, fc_pos

def mreza_u_sd(mreza, sd_ref, device="cpu", strict_shapes=True):
    return mreza_2_sd(mreza, sd_ref, device=device, strict_shapes=strict_shapes)

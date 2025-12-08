# ========================== KONVERZIJA MATRICA ===============================

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
#%%                      DTYPE / bridge helpers
# -----------------------------------------------------------------------------

_TORCH_DTYPE_MAP = {
    "float32": torch.float32, "float": torch.float32,
    "float64": torch.float64, "double": torch.float64,
    "float16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64, "long": torch.int64,
    "int32": torch.int32, "int": torch.int32,
    "int16": torch.int16, "short": torch.int16,
    "int8": torch.int8, "uint8": torch.uint8,
    "bool": torch.bool,
}

def _np_supports_bfloat16() -> bool:
    try:
        _ = np.dtype("bfloat16")
        return True
    except Exception:
        return False

def _torch_dtype_from_str(s: Optional[str]) -> Optional[torch.dtype]:
    if not s:
        return None
    return _TORCH_DTYPE_MAP.get(s)

@dataclass(frozen=True)
class ArrayMeta:
    """Meta-informacije za rekonverziju."""
    shape: Tuple[int, ...]
    dtype: str
    rawbf16: bool

class DTypeBridge:
    """Jedinstveno mesto za Torch<->NumPy prebacivanja sa BF16 podrškom."""
    _NP_HAS_BF16 = _np_supports_bfloat16()

    @staticmethod
    def torch_to_numpy(t: torch.Tensor) -> Tuple[np.ndarray, ArrayMeta]:
        dt_str = str(t.dtype).replace("torch.", "")
        if t.dtype == torch.bfloat16 and not DTypeBridge._NP_HAS_BF16:
            # čuvamo bitpattern kao uint16
            arr_u16 = t.detach().cpu().contiguous().view(torch.uint16).numpy()
            return arr_u16, ArrayMeta(shape=tuple(t.shape), dtype="bfloat16", rawbf16=True)
        # standardni slučaj (NumPy podržava dtype)
        arr = t.detach().cpu().contiguous().numpy()
        return arr, ArrayMeta(shape=tuple(t.shape), dtype=dt_str, rawbf16=False)

    @staticmethod
    def numpy_to_torch(
        arr: np.ndarray,
        meta: ArrayMeta,
        *,
        device: torch.device,
        like_param: Optional[torch.nn.Parameter] = None
    ) -> torch.Tensor:
        target_dtype = _torch_dtype_from_str(meta.dtype) or (like_param.dtype if like_param is not None else None)
        if meta.dtype == "bfloat16" and meta.rawbf16:
            # arr je np.uint16 bitpattern za BF16
            t_u16 = torch.from_numpy(arr).to(device=device, dtype=torch.uint16)
            return t_u16.view(torch.bfloat16)

        t = torch.from_numpy(arr).to(device=device)
        if target_dtype is not None and t.dtype != target_dtype:
            t = t.to(dtype=target_dtype)
        return t

# -----------------------------------------------------------------------------
#%%                             Meta zapisi
# -----------------------------------------------------------------------------

@dataclass
class ParamInfo:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    device: str
    param: torch.Tensor

# -----------------------------------------------------------------------------
#%%                   Detekcija embedding/bias parametara
# -----------------------------------------------------------------------------

def list_embedings(model: nn.Module) -> List[ParamInfo]:
    """Jedinstvena lista nn.Embedding težina (po id)."""
    seen = set()
    out: List[ParamInfo] = []
    for mod_name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding) and hasattr(mod, "weight"):
            p = mod.weight
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            out.append(ParamInfo(
                name=f"{mod_name}.weight",
                shape=tuple(p.shape),
                dtype=str(p.dtype).replace("torch.", ""),
                device=str(p.device),
                param=p,
            ))

    # Fallback heuristika (retko potrebno, ali korisno)
    for full_name, p in model.named_parameters():
        if id(p) in seen:
            continue
        if p.ndim == 2 and "embedding" in full_name.lower():
            seen.add(id(p))
            out.append(ParamInfo(
                name=full_name,
                shape=tuple(p.shape),
                dtype=str(p.dtype).replace("torch.", ""),
                device=str(p.device),
                param=p,
            ))

    out.sort(key=lambda d: d.name)
    return out

def find_biases(model: nn.Module) -> List[ParamInfo]:
    """Pronađi sve bias parametre (tipično 1D) sa deduplikacijom po id."""
    seen = set()
    out: List[ParamInfo] = []

    for full_name, p in model.named_parameters():
        if id(p) in seen:
            continue
        if full_name.endswith(".bias") and p.ndim == 1:
            seen.add(id(p))
            out.append(ParamInfo(full_name, tuple(p.shape), str(p.dtype).replace("torch.", ""), str(p.device), p))

    # fallback ako su preimenovani, ali i dalje 1D i sadrže "bias"
    for full_name, p in model.named_parameters():
        if id(p) in seen:
            continue
        if p.ndim == 1 and "bias" in full_name.lower():
            seen.add(id(p))
            out.append(ParamInfo(full_name, tuple(p.shape), str(p.dtype).replace("torch.", ""), str(p.device), p))

    out.sort(key=lambda d: d.name)
    return out

def _redosled_embeddings(model: nn.Module) -> Tuple[str, ...]:
    return tuple(m.name for m in list_embedings(model))

def _redosled_biases(model: nn.Module) -> Tuple[str, ...]:
    return tuple(m.name for m in find_biases(model))

# -----------------------------------------------------------------------------
#%%                            Meni / izbor
# -----------------------------------------------------------------------------

def pravi_menu(model: nn.Module) -> Tuple[str, List[ParamInfo]]:
    embeds = list_embedings(model)
    if not embeds:
        return "Nije pronađen nijedan nn.Embedding sloj.", embeds
    lines = [f"{i}. {e.name}  shape={e.shape}  dtype={e.dtype}  device={e.device}"
             for i, e in enumerate(embeds, start=1)]
    lines.append(f"{len(embeds)+1}. ALL  (tuple svih embedding matrica + tuple shape-ova)")
    return "\n".join(lines), embeds

def print_menu(menu_text: str) -> None:
    print(menu_text)

# -----------------------------------------------------------------------------
#%%         Konverzija u NumPy 2D + meta (drži dtype i BF16 flag)
# -----------------------------------------------------------------------------

def _to_2d_keepmeta(t: torch.Tensor) -> Tuple[np.ndarray, ArrayMeta]:
    arr, meta = DTypeBridge.torch_to_numpy(t)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
        meta = ArrayMeta(shape=(arr.shape[0], arr.shape[1]), dtype=meta.dtype, rawbf16=meta.rawbf16)
    return arr, meta

def _bias_to_2d_keepmeta(t: torch.Tensor) -> Tuple[np.ndarray, ArrayMeta]:
    arr, meta = DTypeBridge.torch_to_numpy(t)
    arr = arr.reshape(-1, 1)
    meta = ArrayMeta(shape=tuple(t.shape), dtype=meta.dtype, rawbf16=meta.rawbf16)
    return arr, meta

# -----------------------------------------------------------------------------
#%%             Selekcija embedding vektora za zaključavanje
# -----------------------------------------------------------------------------

def izaberi_embeding(embeds_meta: List[ParamInfo], choice: int) -> Dict[str, Any]:
    N = len(embeds_meta)
    if N == 0:
        raise ValueError("Nema dostupnih embedding slojeva u modelu.")
    if choice == N + 1:
        matrices, shapes, names, dtypes, rawbf16 = [], [], [], [], []
        for e in embeds_meta:
            np2d, m = _to_2d_keepmeta(e.param)
            matrices.append(np2d); shapes.append(m.shape); names.append(e.name)
            dtypes.append(m.dtype); rawbf16.append(m.rawbf16)
        return {
            "mode": "all",
            "matrices": tuple(matrices),
            "shapes": tuple(shapes),
            "names": tuple(names),
            "dtypes": tuple(dtypes),
            "rawbf16": tuple(rawbf16),
        }
    if 1 <= choice <= N:
        e = embeds_meta[choice - 1]
        np2d, m = _to_2d_keepmeta(e.param)
        return {
            "mode": "single",
            "name": e.name,
            "matrix": np2d,
            "shape": m.shape,
            "dtype": m.dtype,
            "rawbf16": m.rawbf16,
        }
    raise ValueError(f"Nevažeći izbor: {choice}. Dozvoljeno je 1..{N+1}.")

# -----------------------------------------------------------------------------
#%%                         Bias lista u 2D
# -----------------------------------------------------------------------------

def list_biases(model: nn.Module) -> List[Dict[str, Any]]:
    biases_meta = find_biases(model)
    out: List[Dict[str, Any]] = []
    for b in biases_meta:
        np2d, m = _bias_to_2d_keepmeta(b.param)
        out.append({
            "name": b.name,
            "matrix": np2d,
            "shape": m.shape,
            "dtype": m.dtype,
            "rawbf16": m.rawbf16,
        })
    return out

# -----------------------------------------------------------------------------
#%%       Formiranje mreže (embed + bias) i META za rekonstrukciju
# -----------------------------------------------------------------------------

def mreza_iz_selekcije(selected_embed_result: Dict[str, Any], model: nn.Module):
    if selected_embed_result["mode"] == "single":
        embed_list = [selected_embed_result["matrix"]]
        sel_names  = (selected_embed_result["name"],)
        emb_dtypes = (selected_embed_result.get("dtype"),)
        emb_rawbf  = (selected_embed_result.get("rawbf16", False),)
        mode = "single"
    elif selected_embed_result["mode"] == "all":
        embed_list = list(selected_embed_result["matrices"])
        sel_names  = tuple(selected_embed_result["names"])
        emb_dtypes = tuple(selected_embed_result.get("dtypes", (None,) * len(embed_list)))
        emb_rawbf  = tuple(selected_embed_result.get("rawbf16", (False,) * len(embed_list)))
        mode = "all"
    else:
        raise ValueError("Nepoznat 'mode' u selected_embed_result.")

    biases = list_biases(model)
    bias_list   = [b["matrix"] for b in biases]
    bias_dtypes = tuple(b["dtype"] for b in biases)
    bias_rawbf  = tuple(b.get("rawbf16", False) for b in biases)

    target_order = _redosled_embeddings(model)
    bias_order   = _redosled_biases(model)
    name2idx     = {n: i for i, n in enumerate(target_order)}
    sel_indices  = tuple(name2idx[n] for n in sel_names if n in name2idx)

    meta = {
        "mode": mode,
        "selected_names": sel_names,
        "selected_indices": sel_indices,
        "embedding_target_order": target_order,
        "bias_target_order": bias_order,
        "embedding_dtypes": emb_dtypes,
        "embedding_rawbf16": emb_rawbf,
        "bias_dtypes": bias_dtypes,
        "bias_rawbf16": bias_rawbf,
    }

    mreza = (embed_list, bias_list)
    return mreza, meta

# -----------------------------------------------------------------------------
#%%                       Interaktivni wrapper
# -----------------------------------------------------------------------------

def pravi_mrezu(arg1, model: Optional[nn.Module] = None, interactive: bool = True):
    if isinstance(arg1, nn.Module) and model is None:
        model = arg1
        menu_text, embeds_meta = pravi_menu(model)
        if interactive:
            print_menu(menu_text)
            N = len(embeds_meta)
            if N == 0:
                choice = 1
            else:
                while True:
                    s = input(f"Izbor (1..{N}) ili ENTER za ALL: ").strip()
                    if s == "":
                        choice = N + 1; break
                    if s.isdigit():
                        c = int(s)
                        if 1 <= c <= N + 1:
                            choice = c; break
                    print("Nevažeći unos, pokušaj ponovo.")
        else:
            choice = (len(embeds_meta) + 1) if embeds_meta else 1

        selected = izaberi_embeding(embeds_meta, choice)
        return mreza_iz_selekcije(selected, model)

    if not isinstance(model, nn.Module):
        raise TypeError("pravi_mrezu(stari potpis): očekujem nn.Module kao drugi argument.")
    return mreza_iz_selekcije(arg1, model)

# -----------------------------------------------------------------------------
#%%                     Provera & upis nazad u model
# -----------------------------------------------------------------------------

def _embedding_params_order(model: nn.Module):
    metas = list_embedings(model)
    return [(m.name, m.param) for m in metas]

def _bias_params_order(model: nn.Module):
    metas = find_biases(model)
    return [(m.name, m.param) for m in metas]

def proveri_mrezu_vs_model(
    mreza,
    model: nn.Module,
    meta: Optional[Dict[str, Any]] = None,
    embed_index: Optional[int] = None
) -> Dict[str, Any]:
    if not (isinstance(mreza, tuple) and len(mreza) == 2):
        raise TypeError("proveri_mrezu_vs_model: očekujem mreza=(embed_list, bias_list).")

    emb_mats, bias_mats = mreza
    emb_targets = _embedding_params_order(model)
    bias_targets = _bias_params_order(model)

    report = {"embeddings": [], "biases": [], "summary": {}}

    # Embeddings
    if len(emb_mats) == 0:
        pass
    elif len(emb_mats) == len(emb_targets):
        for (name, p), mat in zip(emb_targets, emb_mats):
            ok = tuple(p.shape) == tuple(mat.shape)
            report["embeddings"].append({"name": name, "expected": tuple(p.shape), "got": tuple(mat.shape), "ok": ok})
    elif len(emb_mats) == 1:
        if embed_index is None and meta and meta.get("mode") == "single":
            sel_name = meta["selected_names"][0]
            name2idx = {n: i for i, (n, _) in enumerate(emb_targets)}
            if sel_name not in name2idx:
                raise KeyError(f"Embedding '{sel_name}' ne postoji u trenutnom modelu.")
            embed_index = name2idx[sel_name] + 1
        if embed_index is None:
            raise ValueError("Jedan embedding u mrezi; prosledi `meta` ili `embed_index` (1..N).")
        idx = int(embed_index) - 1
        if not (0 <= idx < len(emb_targets)):
            raise IndexError(f"embed_index van opsega: 1..{len(emb_targets)}")
        name, p = emb_targets[idx]
        mat = emb_mats[0]
        ok = tuple(p.shape) == tuple(mat.shape)
        report["embeddings"].append({"name": name, "expected": tuple(p.shape), "got": tuple(mat.shape), "ok": ok})
    else:
        raise ValueError(
            f"Nejasan broj embedding matrica u mreza[0]: {len(emb_mats)} (očekujem 0, 1 ili {len(emb_targets)})."
        )

    # Biases
    if len(bias_mats) != len(bias_targets):
        report["summary"]["bias_mismatch_count"] = (len(bias_mats), len(bias_targets))
        raise ValueError(
            f"Broj bias matrica ({len(bias_mats)}) ne odgovara broju bias parametara u modelu ({len(bias_targets)})."
        )
    for (name, p), mat in zip(bias_targets, bias_mats):
        vecN = mat.reshape(-1).shape[0]
        ok = (p.numel() == vecN)
        report["biases"].append({"name": name, "expected": (p.numel(),), "got": (vecN,), "ok": ok})

    emb_ok = all(x["ok"] for x in report["embeddings"]) if report["embeddings"] else True
    bias_ok = all(x["ok"] for x in report["biases"]) if report["biases"] else True
    report["summary"].update({"embedings_ok": emb_ok, "biases_ok": bias_ok})
    return report

# -----------------------------------------------------------------------------
#%%                     Numpy matrica upis nazad u model
# -----------------------------------------------------------------------------

def mreza_2_model(mreza, model: nn.Module, meta: Optional[Dict[str, Any]] = None,
    embed_index: Optional[int] = None, strict: bool = True) -> Dict[str, Any]:
    
    if not (isinstance(mreza, tuple) and len(mreza) == 2):
        raise TypeError("mreza_2_model: očekujem mreza=(embed_list, bias_list).")

    emb_mats, bias_mats = mreza
    emb_targets = _embedding_params_order(model)
    bias_targets = _bias_params_order(model)

    written = {"embeddings": 0, "biases": 0}

    def _embeddings_keep_model_dtype(param: torch.Tensor, np_mat: np.ndarray, name: str):
        if strict and tuple(param.shape) != tuple(np_mat.shape):
            raise ValueError(f"Shape mismatch '{name}': očekujem {tuple(param.shape)}, dobio {tuple(np_mat.shape)}")
        # UVEK koristi dtype ciljnog parametra (original model dtype)
        t = torch.from_numpy(np_mat).to(device=param.device, dtype=param.dtype)
        param.copy_(t)

    def _bias_keep_numpy_dtype(param: torch.Tensor, np_vec: np.ndarray, name: str):
        vec = np_vec.reshape(-1)
        if strict and param.numel() != vec.size:
            raise ValueError(f"Bias size mismatch '{name}': očekujem {param.numel()}, dobio {vec.size}")
        # Zadrži numpy dtype (npr. float64 za watermark)
        t = torch.from_numpy(vec).to(device=param.device)
        if param.dtype != t.dtype:
            # Upcast parametra na dtype izvora (ne radimo downcast koji bi kvario watermark)
            param.data = param.data.to(dtype=t.dtype)
        param.copy_(t)

    with torch.no_grad():

        # ---------------------------- Embeddings -----------------------------

        if len(emb_mats) == 0:
            pass

        elif embed_index is not None:
            idx = int(embed_index) - 1
            if not (0 <= idx < len(emb_targets)):
                raise IndexError(f"embed_index van opsega: 1..{len(emb_targets)}")
            (tgt_name, tgt_p) = emb_targets[idx]
            _embeddings_keep_model_dtype(tgt_p, emb_mats[0], tgt_name)
            written["embeddings"] += 1

        elif meta and meta.get("mode") == "single":
            sel_name = meta["selected_names"][0]
            tgt = {n: p for n, p in emb_targets}
            if sel_name not in tgt:
                raise KeyError(f"Embedding '{sel_name}' ne postoji u trenutnom modelu.")
            tgt_p = tgt[sel_name]
            _embeddings_keep_model_dtype(tgt_p, emb_mats[0], sel_name)
            written["embeddings"] += 1

        elif meta and meta.get("mode") == "all":
            name_to_mat = {n: m for n, m in zip(meta["selected_names"], emb_mats)}
            for tgt_name, tgt_p in emb_targets:
                if tgt_name not in name_to_mat:
                    raise KeyError(f"Nedostaje matrica za '{tgt_name}' u meta['selected_names'].")
                _embeddings_keep_model_dtype(tgt_p, name_to_mat[tgt_name], tgt_name)
                written["embeddings"] += 1

        else:
            if len(emb_mats) != len(emb_targets):
                raise ValueError(
                    f"Nejasan broj embedding matrica u mreza[0]: {len(emb_mats)} (očekujem 0, 1 ili {len(emb_targets)})."
                )
            for (tgt_name, tgt_p), mat in zip(emb_targets, emb_mats):
                _embeddings_keep_model_dtype(tgt_p, mat, tgt_name)
                written["embeddings"] += 1

        # ----------------------------- Biases --------------------------------

        # Ignorišemo meta dtype/rawbf16 za bias-e: čuvamo dtype iz mreza (NumPy)
        
        for (name, p), mat in zip(bias_targets, bias_mats):
            _bias_keep_numpy_dtype(p, mat, name)
            written["biases"] += 1                      

    return {"written": written, "total_emb_targets": len(emb_targets), "total_bias_targets": len(bias_targets)}

# =============================================================================
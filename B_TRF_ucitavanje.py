# ------------------------ UCITAVANJE + ARHITEKTURA  --------------------------
#         SST-2 fine-tuned varijante sa klasifikacionom glavom
# 1) DistilBERT  2) TinyBERT  3) MiniLM  4) ALBERT  5) MobileBERT  6) BERT-Tiny
# -----------------------------------------------------------------------------

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# Tihi mod logova
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Varijable koje želiš u Exploreru
Tokenizer = None
Model = None
arch_info: Dict[str, Any] = {}
device_str = "cpu"
ime = None

# -----------------------------------------------------------------------------
#%%                                 helpers
# -----------------------------------------------------------------------------

def _device():
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return d, ("cuda" if d.type == "cuda" else "cpu")

def _export_to_ipython():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.user_ns.update({
                "Tokenizer": Tokenizer,
                "Model": Model,
                "arch_info": arch_info,
                "device_str": device_str,
                "ime": ime,
                "save_current": save_current,  # <— lako snimanje
            })
            print("----------------------------------------------------------------------")
            print("Varijable izvezene u IPython: Tokenizer, Model, arch_info, device_str, save_current().")
    except Exception:
        pass

# Heuristika za bias ključeve
def _is_bias_key(name: str, extra_bias_patterns: Optional[List[str]] = None) -> bool:
    patterns = [r"\bbias\b", r"\.bias$"]
    if extra_bias_patterns:
        patterns.extend(extra_bias_patterns)
    return any(re.search(p, name, flags=re.IGNORECASE) for p in patterns)

# ----- Učitavanje state_dict uz očuvanje dtype-a bias-a iz checkpoint-a ------
@torch.no_grad()
def load_state_dict_preserve_bias_dtype(
    model: nn.Module,
    state: Dict[str, torch.Tensor],
    *,
    keep_bias_from_state: bool = True,
    bias_patterns: Optional[List[str]] = None,
    map_to_device: Optional[torch.device] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    state_params = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    model_params = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())

    missing, unexpected = [], []

    for name, p in model_params.items():
        src = state_params.get(name, None)
        if src is None:
            missing.append(name)
            continue
        target_device = map_to_device if map_to_device is not None else p.device
        if keep_bias_from_state and _is_bias_key(name, bias_patterns):
            if p.dtype != src.dtype:
                p.data = p.data.to(dtype=src.dtype)  # upcast
            p.copy_(src.to(device=target_device))
        else:
            p.copy_(src.to(device=target_device, dtype=p.dtype))

    for name, b in model_buffers.items():
        src = state_params.get(name, None)
        if src is None:
            continue
        target_device = map_to_device if map_to_device is not None else b.device
        b.copy_(src.to(device=target_device, dtype=b.dtype))

    model_all = set(list(model_params.keys()) + list(model_buffers.keys()))
    for k in state_params.keys():
        if k not in model_all:
            unexpected.append(k)

    if strict and (missing or unexpected):
        raise RuntimeError(
            "load_state_dict_preserve_bias_dtype(strict=True) neuspešno.\n"
            f"Missing keys: {missing}\nUnexpected keys: {unexpected}"
        )
    return {"missing": missing, "unexpected": unexpected}

# WL loader (ostavljen ako želiš da ga koristiš i sa klasifikacionim modelom)
def _maybe_load_wl_checkpoint_from_dir(model: nn.Module, wl_base_dir: str, dev: torch.device):
    """
    Proba sledećim redom:
      1) <wl_base_dir>/model.safetensors
      2) <wl_base_dir>/WL/model.safetensors
    """
    _PRESERVE_WL_BIAS_DTYPE = True
    _WL_SUBDIR_NAME = "WL"
    _WL_FILENAME = "model.safetensors"

    if not _PRESERVE_WL_BIAS_DTYPE:
        return False

    base = Path(wl_base_dir)
    tried = []
    candidates = [
        base / _WL_FILENAME,
        base / _WL_SUBDIR_NAME / _WL_FILENAME
    ]

    wl_path = None
    for c in candidates:
        tried.append(str(c))
        if c.is_file():
            wl_path = c
            break

    if wl_path is None:
        print("[INFO] WL checkpoint nije pronađen (preskačem).")
        return False

    try:
        from safetensors.torch import load_file as safetensors_load_file
    except ImportError:
        print("[WARN] 'safetensors' nije instaliran. Preskačem WL učitavanje.")
        return False

    print(f"[INFO] Učitavam WL checkpoint iz: {wl_path}")
    state = safetensors_load_file(str(wl_path))
    report = load_state_dict_preserve_bias_dtype(
        model,
        state,
        keep_bias_from_state=True,
        bias_patterns=None,
        map_to_device=dev,
        strict=False,
    )

    any_bias64 = any(_is_bias_key(n) and (p.dtype == torch.float64) for n, p in model.named_parameters())
    print("[OK] WL učitan; bias dtype očuvan (float64)." if any_bias64 else
          "[INFO] WL učitan; float64 bias nije detektovan.")

    if report["missing"] or report["unexpected"]:
        print("[INFO] missing:", report["missing"])
        print("[INFO] unexpected:", report["unexpected"])
    return True

def _normpath(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    try:
        return str(Path(p).resolve())
    except Exception:
        return None

# -----------------------------------------------------------------------------
#%%                                 BIASES / EMBEDDINGS / SUMMARIZE
# -----------------------------------------------------------------------------

def _collect_bias_vectors(model: nn.Module):
    biases = []
    for name, p in model.named_parameters(recurse=True):
        if not name.endswith(".bias"):
            continue
        if p is None:
            continue
        module_type = None
        try:
            parent = model
            parts = name.split(".")[:-1]
            for pt in parts:
                parent = getattr(parent, pt)
            module_type = type(parent).__name__
        except Exception:
            module_type = None

        biases.append({
            "name": name,
            "shape": tuple(p.shape),
            "ndim": p.ndim,
            "dtype": str(p.dtype).replace("torch.", ""),
            "device": str(p.device),
            "numel": p.numel(),
            "module_type": module_type
        })
    return biases

def _collect_embedding_modules(model: nn.Module):
    embeds = []
    ptr2name = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            w = getattr(module, "weight", None)
            shape = tuple(w.shape) if (w is not None and hasattr(w, "shape")) else None
            num_embeddings = getattr(module, "num_embeddings", None)
            embedding_dim = getattr(module, "embedding_dim", None)
            padding_idx = getattr(module, "padding_idx", None)
            dtype = getattr(w, "dtype", None)
            device = str(getattr(w, "device", ""))

            tied_to = None
            try:
                if w is not None:
                    ptr = int(w.data_ptr())
                    if ptr in ptr2name:
                        tied_to = ptr2name[ptr]
                    else:
                        ptr2name[ptr] = name
            except Exception:
                pass

            embeds.append({
                "name": name,
                "type": module.__class__.__name__,
                "weight_shape": shape,
                "num_embeddings": num_embeddings,
                "embedding_dim": embedding_dim,
                "padding_idx": padding_idx,
                "dtype": str(dtype).replace("torch.", "") if dtype is not None else None,
                "device": device,
                "tied_to": tied_to,
            })
    return embeds

def _has_module(model: nn.Module, path: str) -> bool:
    cur = model
    for part in path.split("."):
        if not hasattr(cur, part):
            return False
        cur = getattr(cur, part)
    return True

def _first_existing(model: nn.Module, candidates):
    for c in candidates:
        if _has_module(model, c):
            return c
    return None

def summarize_architecture(model: nn.Module, tokenizer) -> Dict[str, Any]:
    cfg = model.config
    model_type = getattr(cfg, "model_type", type(model).__name__)

    hidden_size = getattr(cfg, "hidden_size", None)
    num_layers = getattr(cfg, "num_hidden_layers", None)
    num_heads = getattr(cfg, "num_attention_heads", None)
    intermediate_size = getattr(cfg, "intermediate_size", None)
    head_dim = int(hidden_size // num_heads) if hidden_size and num_heads else None

    vocab_size = getattr(cfg, "vocab_size", None)
    max_pos = getattr(cfg, "max_position_embeddings", None)
    type_vocab = getattr(cfg, "type_vocab_size", None)

    attn_dropout = getattr(cfg, "attention_probs_dropout_prob", getattr(cfg, "attention_dropout", None))
    hidden_dropout = getattr(cfg, "hidden_dropout_prob", getattr(cfg, "dropout", None))
    hidden_act = getattr(cfg, "hidden_act", None)

    is_decoder = getattr(cfg, "is_decoder", False)
    add_cross_attention = getattr(cfg, "add_cross_attention", False)

    pooler_path = _first_existing(model, ["bert.pooler", "albert.pooler", "distilbert.pre_classifier", "mobilebert.pooler"])
    embeddings = _collect_embedding_modules(model)
    biases = _collect_bias_vectors(model)

    encoder_path = _first_existing(model, ["bert.encoder", "albert.encoder", "distilbert.transformer", "mobilebert.encoder"])
    decoder_path = _first_existing(model, ["bert.decoder", "decoder"])

    tok_info = {
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "do_lower_case": getattr(tokenizer, "do_lower_case", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "special_tokens_map": list(getattr(tokenizer, "special_tokens_map", {}).keys()),
    }

    info = {
        "framework": "PyTorch",
        "model_class": type(model).__name__,
        "model_type": model_type,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_pos,
        "type_vocab_size": type_vocab,
        "attention_dropout": attn_dropout,
        "hidden_dropout": hidden_dropout,
        "activation": hidden_act,
        "is_decoder": is_decoder,
        "add_cross_attention": add_cross_attention,
        "has_pooler": pooler_path is not None,
        "pooler_module_path": pooler_path,
        "has_classifier": hasattr(model, "classifier"),
        "classifier_module_path": "classifier" if hasattr(model, "classifier") else None,
        "encoder_module_path": encoder_path,
        "decoder_module_path": decoder_path,
        "embedding_modules": embeddings,
        "bias_vectors": biases,
        "tokenizer": tok_info,
    }

    if pooler_path and "pooler" in pooler_path:
        info["pooler_activation"] = "tanh"

    # Dodatna proverka: mora da bude klasa za klasifikaciju sa 2 labela (SST-2)
    archs = (getattr(cfg, "architectures", None) or [])
    info["architectures"] = archs
    info["num_labels"] = getattr(cfg, "num_labels", None)
    return info

# -----------------------------------------------------------------------------
#%%                                  UCITAVANJE 
# -----------------------------------------------------------------------------

def _validate_classifier_config(cfg: AutoConfig):
    archs = (getattr(cfg, "architectures", None) or [])
    if not any(str(a).endswith("ForSequenceClassification") for a in archs):
        print("[WARN] Checkpoint nije *ForSequenceClassification (možda bazni enkoder).")
    if getattr(cfg, "num_labels", None) != 2:
        print(f"[WARN] Očekivano num_labels=2 za SST-2, ali je {getattr(cfg, 'num_labels', None)}.")

def load_one(hf_id: str, src_for_model: Optional[str], wl_dir: Optional[str] = None):
    """
    Učitava *klasifikacioni* model (sa glavom).
    Ako postoji local dir (sa config.json), koristi njega; inače povlači sa HF (hf_id).
    Opcioni WL merge ostavljen ako ti treba.
    """
    use_local_for_model = bool(src_for_model) and Path(src_for_model).exists() and any(Path(src_for_model).glob("*.json"))
    src = src_for_model if use_local_for_model else hf_id
    print(f"\nUčitavam model iz: {src}")

    tok = AutoTokenizer.from_pretrained(src, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(src, use_safetensors=True).eval()

    # Validacija da je zaista klasifikacioni checkpoint
    _validate_classifier_config(mdl.config)

    dev, dev_str = _device()
    mdl.to(dev)

    # WL checkpoint samo ako je eksplicitno prosleđen drugi folder
    if wl_dir:
        _maybe_load_wl_checkpoint_from_dir(mdl, wl_dir, dev)

    print(f"Učitano na uređaj: {dev_str}")
    return tok, mdl, dev_str

def save_current(dest_dir: str):
    """
    Snima trenutni Tokenizer + Model u 'dest_dir' tako da ostanu *potpuno upotrebljivi*.
    """
    global Tokenizer, Model
    if Tokenizer is None or Model is None:
        print("[ERR] Nema učitanog modela/tokenizera.")
        return
    out = Path(dest_dir)
    out.mkdir(parents=True, exist_ok=True)
    Tokenizer.save_pretrained(out)
    Model.save_pretrained(out)
    print(f"[OK] Sačuvano u: {out.resolve()}")

def print_menu(models_dict):
    print("\n========= TRANSFORMER LOADER (SST-2 fine-tuned) ==========")
    for k, v in models_dict.items():
        print(f"{k}) {v['name']}")
    print("Q) Quit")

# -----------------------------------------------------------------------------
#%%                                   MAIN
# -----------------------------------------------------------------------------

def main():
    global Tokenizer, Model, arch_info, device_str, ime

    # SST-2 (sa glavom) — lokalni folderi + HF fallback
    MODELS = {
        "1": {"name": "DistilBERT_sst2", "hf_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english", "local_dir": "models/DistilBERT_sst2"},
        "2": {"name": "TinyBERT_sst2",   "hf_id": "philschmid/tiny-bert-sst2-distilled",                         "local_dir": "models/TinyBERT_sst2"},
        "3": {"name": "MiniLM_sst2",     "hf_id": "philschmid/MiniLM-L6-H384-uncased-sst2",                      "local_dir": "models/MiniLM_sst2"},
        "4": {"name": "ALBERT_sst2",     "hf_id": "textattack/albert-base-v2-SST-2",                             "local_dir": "models/ALBERT_sst2"},
        "5": {"name": "MobileBERT_sst2", "hf_id": "Alireza1044/mobilebert_sst2",                                 "local_dir": "models/MobileBERT_sst2"},
        "6": {"name": "BERT-Tiny_sst2",  "hf_id": "M-FAC/bert-tiny-finetuned-sst2",                              "local_dir": "models/BERT-Tiny_sst2"},
    }

    print_menu(MODELS)
    print("---------------------------------------")
    choice = input("\nIzaberi model (1-6 ili Q): ").strip().lower()
    print("---------------------------------------")

    if choice == "q":
        print("Zatvaram.")
        return

    if choice in MODELS:
        v = MODELS[choice]
        ime = v["name"]
        print("----------------------------------------------------------------------")
        custom = input(
            f"Podrazumevani lokalni folder je '{v['local_dir']}'.\n"
            f"- Enter = učitaj iz tog foldera (ako postoji) ili automatski sa HF.\n"
            f"- Ili unesi drugi lokalni folder.\n: "
        ).strip()
        print("----------------------------------------------------------------------")

        # Odakle se učitava model (lokalno ili HF)
        src_for_model = custom if custom else v["local_dir"]

        # NORMALIZUJ putanje pre odluke o WL učitavanju
        default_dir_norm = _normpath(v["local_dir"])
        custom_dir_norm  = _normpath(custom)

        # WL se učitava SAMO ako je korisnik uneo DRUGI folder (po rezolviranoj putanji)
        wl_dir = None
        if custom_dir_norm and default_dir_norm and (custom_dir_norm != default_dir_norm):
            wl_dir = custom  # WL se traži u TOM (custom) folderu
        else:
            wl_dir = None    # nikad ne učitavaj WL iz podrazumevanog foldera

        print(f"[INFO] default_dir={default_dir_norm}")
        print(f"[INFO] chosen_dir ={custom_dir_norm or '(none)'}")
        print(f"[INFO] WL folder  ={wl_dir or '(disabled)'}")

        try:
            Tokenizer, Model, device_str = load_one(v["hf_id"], src_for_model, wl_dir=wl_dir)
            arch_info = summarize_architecture(Model, Tokenizer)

            print("\n----------------------- Embedding moduli ---------------------------")
            if arch_info["embedding_modules"]:
                for e in arch_info["embedding_modules"]:
                    shp = e.get("weight_shape", None)
                    shp_str = f"{shp}" if shp is not None else "N/A"
                    tie = f"  [tied → {e['tied_to']}]" if e.get("tied_to") else ""
                    print(f" - {e['name']}: shape={shp_str} dtype={e.get('dtype')}){tie}")
            else:
                print("Nije pronađen nijedan nn.Embedding modul.")
            print("----------------------------------------------------------------------")

            print("\n-------------------------- Bias vektori -----------------------------")
            biases = arch_info.get("bias_vectors", [])
            if biases:
                for b in biases:
                    print(f" - {b['name']} [{b.get('module_type','?')}]: shape={b['shape']} dtype={b['dtype']})")
            else:
                print("Nije pronađen nijedan bias parametar.")
            print("----------------------------------------------------------------------")

            print("[TIP] Sačuvaj trenutno učitani model/ tokenizer pozivom: save_current('models/Backup_sst2')")
            _export_to_ipython()

        except Exception as e:
            print(f" Greška: {e}!")
    else:
        print("Nepoznata opcija.")

    print("----------------------------------------------------------------------")

if __name__ == "__main__":
    main()

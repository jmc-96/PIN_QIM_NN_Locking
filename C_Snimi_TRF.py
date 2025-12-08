# ---------------------- SNIMANJE MODELA U models/<ime>/WL --------------------
from pathlib import Path
from typing import Optional, List
import shutil
import torch.nn as nn

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # NEW

AUX_WHITELIST_DEFAULT: List[str] = [
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "added_tokens.json",
    "preprocessor_config.json",
    "generation_config.json",
]

def _safe_copy(src: Path, dst: Path, overwrite_ok: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists() and not overwrite_ok:
        raise FileExistsError(f"Već postoji: {dst}")
    shutil.copy2(src, dst)
    return True

def _copy_aux_files(
    base: Path, wl_dir: Path, overwrite_ok: bool, whitelist: Optional[List[str]]
) -> list:
    names = whitelist if whitelist is not None else AUX_WHITELIST_DEFAULT
    copied = []
    for name in names:
        src = base / name
        dst = wl_dir / name
        if _safe_copy(src, dst, overwrite_ok):
            copied.append(dst)
    return copied

def snimi_u_WL(
    model: nn.Module,
    base_dir: str,
    copy_aux: bool = True,
    overwrite_ok: bool = True,
    move_to_cpu_before_save: bool = True,
    aux_whitelist: Optional[List[str]] = None,
    # NOVO:
    tokenizer=None,                  # prosledi Tokenizer ako ga imaš; biće snimljen direktno u WL
    subdir_name: str = "WL",         # bez interaktivnog inputa; i dalje možeš promeniti ime
    validate_after_save: bool = True # posle snimanja provera da li je WL učitljiv
) -> Path:
    """
    Snimi stanje modela u:  <base_dir>/<subdir_name>/model.safetensors
    + (opciono) kopiraj/sačuvaj prateće fajlove (config/tokenizer).

    - Ne menjaš model (nema fine-tuninga, nema reinit).
    - Ako proslediš 'tokenizer', biće snimljen direktno (preferirano).
      U suprotnom će se pokušati kopiranje tokenizer fajlova iz base_dir.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"snimi_u_WL: očekujem nn.Module, dobio {type(model).__name__}")

    base = Path(base_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Base dir ne postoji: {base}")

    wl_dir = base / subdir_name
    wl_dir.mkdir(parents=True, exist_ok=True)

    out_path = wl_dir / "model.safetensors"
    if out_path.exists() and not overwrite_ok:
        raise FileExistsError(f"Fajl već postoji: {out_path}")

    try:
        from safetensors.torch import save_file as safetensors_save_file
    except ImportError as e:
        raise RuntimeError("Nedostaje paket 'safetensors' (pip install safetensors).") from e

    # 1) Sačuvaj težine 1:1 (bez promena)
    state = model.state_dict()
    if move_to_cpu_before_save:
        state = {k: v.detach().cpu() for k, v in state.items()}
    else:
        state = {k: v.detach() for k, v in state.items()}
    safetensors_save_file(state, str(out_path))

    # 2) Sačuvaj/kopiraj tokenizer + config fajlove tako da WL bude self-contained
    copied = []
    if tokenizer is not None:
        # prioritet: direktno snimanje tokenizer-a u WL
        tokenizer.save_pretrained(wl_dir)
    elif copy_aux:
        # fallback: kopiraj iz parent foldera
        copied = _copy_aux_files(base, wl_dir, overwrite_ok=overwrite_ok, whitelist=aux_whitelist)

    # Ako nema tokenizer fajlova ni posle ova dva koraka, ostavljamo kako jeste;
    # korisnik može naknadno pozvati tokenizer.save_pretrained(WL).

    print("===================================================================")
    print(f"[OK] Sačuvano: {out_path}")
    if tokenizer is not None:
        print("[OK] Tokenizer snimljen direktno u WL.")
    elif copied:
        print("[OK] Kopirano u", subdir_name, ":", ", ".join(p.name for p in copied))
    else:
        print("[INFO] Nema dodatnih fajlova za tokenizer/config (copy_aux=False ili ne postoje u base_dir).")

    # 3) (Opcionalno) Validacija učitljivosti WL foldera – bez ikakvih izmena modela
    if validate_after_save:
        try:
            _tok = AutoTokenizer.from_pretrained(wl_dir)
            _mdl = AutoModelForSequenceClassification.from_pretrained(wl_dir)
            del _tok, _mdl
            print("[OK] WL folder je učitljiv sa AutoTokenizer/AutoModelForSequenceClassification.")
        except Exception as e:
            print(f"[WARN] WL validacija nije prošla: {e}")

    print("===================================================================")
    return wl_dir

__all__ = ["snimi_u_WL"]

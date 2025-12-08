# =============================================================================
#                            TEST PIN ROBUSTNESS
# =============================================================================

import os, json, argparse, math
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModel

# ----------------------------- LOADERI ---------------------------------------

def load_state_dict(path):
    """
    Podržava:
      - HF direktorijum (AutoModel.from_pretrained(path))
      - single .safetensors fajl (safetensors.safe_open)
      - .pt / .bin state_dict fajl (torch.load)
    """
    from pathlib import Path
    p = Path(path)

    # 1) Ako je direktorijum: probaj AutoModel
    if p.is_dir():
        try:
            mdl = AutoModel.from_pretrained(str(p))
            return mdl.state_dict()
        except Exception as e:
            raise RuntimeError(f"HF load failed za dir {p}: {e}")

    # 2) Ako je fajl .safetensors
    if p.is_file() and p.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except Exception:
            raise RuntimeError("Za .safetensors je potreban paket 'safetensors' (pip install safetensors).")
        sd = {}
        with safe_open(str(p), framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        return sd

    # 3) Ako je fajl .pt / .bin sa state_dict-om
    if p.is_file() and p.suffix in (".pt", ".bin"):
        sd = torch.load(str(p), map_location="cpu")
        if isinstance(sd, dict):
            if "state_dict" in sd and isinstance(sd["state_dict"], dict):
                return sd["state_dict"]
            return sd
        raise RuntimeError(f"Datoteka {p} nije dict state_dict.")

    raise RuntimeError(f"Nepodržan format putanje: {p}")

# ------------------------ KLJUČEVI ZA BIAS TENZORE ---------------------------

def _normalize_prefix(full_name: str) -> str:
    """
    Normalizuj ime parametra uklanjanjem vodećih prefiksa ('bert.', 'base_model.', 'model.')
    i zadrži rep od 'encoder.layer' nadole. .bias se uvek skida.
    Primer:
      'bert.encoder.layer.0.output.dense.bias' -> 'encoder.layer.0.output.dense'
      'base_model.encoder.layer.1.attention.output.dense.bias' -> 'encoder.layer.1.attention.output.dense'
      'classifier.bias' -> 'classifier'
    """
    name = full_name
    if name.endswith(".bias"):
        name = name[:-5]

    for lead in ("bert.", "base_model.", "model."):
        if name.startswith(lead):
            name = name[len(lead):]

    if "encoder.layer." in name:
        idx = name.index("encoder.layer.")
        return name[idx:]
    return name

def _build_bias_maps_with_norm(state_dict: dict):
    """
    Vrati:
      raw_map:  {raw_prefix: 1D bias tensor (ORIGINALNI dtype)}
      norm_map: {normalized_prefix: 1D bias tensor (isti objekat/referenca)}
    gde je 'prefix' ime parametra BEZ '.bias'.
    """
    raw_map, norm_map = {}, {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor): 
            continue
        if not k.endswith(".bias"): 
            continue
        if v.dim() != 1: 
            continue
        pref_raw = k[:-5]  # bez '.bias'
        # VAŽNO: BEZ .float() — čuvamo originalni dtype (fp16/bf16/fp32...)
        t = v.detach().cpu().view(-1)
        raw_map[pref_raw] = t
        pref_norm = _normalize_prefix(k)
        norm_map[pref_norm] = t
    return raw_map, norm_map

# ------------------------------ MASKA ----------------------------------------

def load_mask(mask_json):
    with open(mask_json, "r") as f:
        j = json.load(f)
    layers = j.get("layers", {})
    norm_layers = {}
    for key, idxs in layers.items():
        if key.endswith(".bias"):
            key = key[:-5]
        norm_layers[key] = sorted(set(int(i) for i in idxs))
    return norm_layers

# ---------------------------- HISTOGRAM (safe) --------------------------------

def make_histogram_torch(d_tensor: torch.Tensor, title, outpng, bins=50, log=False):
    """
    Crta histogram iz KOPIJE podataka:
    vrednosti se prevode u float64 samo za crtanje (ne menja originalni tensor).
    """
    arr = d_tensor.to(torch.float64).cpu().numpy()
    plt.figure()
    plt.hist(arr, bins=bins, log=log)
    plt.title(title)
    plt.xlabel("|Δbias|")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpng)
    plt.close()

# ------------------------------- MAIN ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True, help="Path to base (watermarked) checkpoint dir or state_dict file")
    ap.add_argument("--ft_ckpt", required=True, help="Path to finetuned checkpoint dir or state_dict file")
    ap.add_argument("--wm_mask_json", required=True, help="JSON with watermark indices per layer prefix")
    ap.add_argument("--threshold", type=float, default=0.1, help="Δ threshold for exceedance")
    ap.add_argument("--percentile", type=float, default=99.0, help="Percentile for delta* (e.g., 99 means 99%% within)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--verbose", action="store_true", help="Print progress/info")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load checkpoints
    sd_base = load_state_dict(args.base_ckpt)
    sd_ft   = load_state_dict(args.ft_ckpt)

    # Napravi RAW i NORMALIZOVANE mape za bias tenzore
    base_bias_raw, base_bias_norm = _build_bias_maps_with_norm(sd_base)
    ft_bias_raw,   ft_bias_norm   = _build_bias_maps_with_norm(sd_ft)

    # Pokušaj presjeka: RAW->RAW
    common = sorted(set(base_bias_raw.keys()) & set(ft_bias_raw.keys()))
    used_map_base, used_map_ft = base_bias_raw, ft_bias_raw

    # Ako nema, probaj NORMALIZOVANO
    if not common:
        common = sorted(set(base_bias_norm.keys()) & set(ft_bias_norm.keys()))
        if common:
            used_map_base, used_map_ft = base_bias_norm, ft_bias_norm

    # Fallback: poravnanje po DUŽINI bias vektora (heuristika)
    pairs_by_len = []
    if not common:
        def by_len(m):
            d = {}
            for name, ten in m.items():
                d.setdefault(int(ten.numel()), []).append((name, ten))
            return d
        bL = by_len(base_bias_norm if base_bias_norm else base_bias_raw)
        fL = by_len(ft_bias_norm   if ft_bias_norm   else ft_bias_raw)

        for L, base_list in bL.items():
            ft_list = fL.get(L, [])
            if not ft_list:
                continue
            base_list_sorted = sorted(base_list, key=lambda x: x[0])
            ft_list_sorted   = sorted(ft_list, key=lambda x: x[0])
            for (bname, bten), (fname, ften) in zip(base_list_sorted, ft_list_sorted):
                pairs_by_len.append((bname, fname))

        if pairs_by_len:
            # Kreiraj sintetičke ključeve koji uparuju parove
            common = [f"__PAIRLEN__::{i}" for i in range(len(pairs_by_len))]
            tmp_base, tmp_ft = {}, {}
            for i, (bname, fname) in enumerate(pairs_by_len):
                key = common[i]
                tb = base_bias_norm.get(bname) or base_bias_raw.get(bname)
                tf = ft_bias_norm.get(fname)   or ft_bias_raw.get(fname)
                tmp_base[key] = tb
                tmp_ft[key]   = tf
            used_map_base, used_map_ft = tmp_base, tmp_ft

    if not common:
        raise RuntimeError("No common bias tensors found between checkpoints.")

    # Load watermark mask
    wm_layers = load_mask(args.wm_mask_json)

    # Priprema
    layer_rows = []
    overall_all_abs_list = []
    overall_T_abs_list   = []

    thr_val = args.threshold  # float prag; u poređenju ćemo napraviti tensor iste preciznosti

    if args.verbose:
        print(f"[INFO] Matched bias entries: {len(common)}")
        # nekoliko primera ključeva
        sample_base = list(used_map_base.keys())[:5]
        sample_ft   = list(used_map_ft.keys())[:5]
        print("[INFO] Sample base keys:", sample_base)
        print("[INFO] Sample ft   keys:", sample_ft)

    # Glavna petlja po slojevima
    for pref in common:
        b0 = used_map_base[pref]  # torch 1D, original dtype
        b1 = used_map_ft[pref]    # torch 1D, original dtype
        if b0.numel() != b1.numel():
            continue

        # |Δbias|
        d = (b1 - b0).abs()
        overall_all_abs_list.append(d)

        # T indeksi za ovaj sloj
        T_idxs = wm_layers.get(pref, [])
        if not T_idxs:
            # probaj relaxed match: suffix/prefix
            for mk in wm_layers.keys():
                if pref.endswith(mk) or mk.endswith(pref):
                    T_idxs = wm_layers[mk]
                    break
        # filtriraj u granicama
        T_idxs = sorted(set(i for i in T_idxs if 0 <= i < d.numel()))

        if T_idxs:
            idx = torch.as_tensor(T_idxs, device=d.device)
            d_T = d.index_select(0, idx)
        else:
            d_T = None

        # METRIKE (ALL)
        mean_all = d.mean().item()
        max_all  = d.max().item()
        thr = torch.as_tensor(thr_val, dtype=d.dtype, device=d.device)
        frac_all = (d > thr).sum().item() / max(1, d.numel())

        # METRIKE (T)
        if d_T is not None and d_T.numel() > 0:
            mean_T = d_T.mean().item()
            max_T  = d_T.max().item()
            frac_T = (d_T > thr).sum().item() / max(1, d_T.numel())
            q = torch.tensor([args.percentile/100.0], dtype=d_T.dtype, device=d_T.device)
            delta_star_T = torch.quantile(d_T, q).item()
        else:
            mean_T = float("nan")
            max_T  = float("nan")
            frac_T = float("nan")
            delta_star_T = float("nan")

        layer_rows.append({
            "layer_prefix": pref,
            "bias_len": int(d.numel()),
            "T_count": int(len(T_idxs)),
            "mean_abs_delta_all": mean_all,
            "max_abs_delta_all": max_all,
            "frac_exceed_all": frac_all,
            "mean_abs_delta_T": mean_T,
            "max_abs_delta_T": max_T,
            "frac_exceed_T": frac_T,
            "delta_star_T_pct": float(args.percentile),
            "delta_star_T": float(delta_star_T)
        })

        # Histogrami po sloju (kopija u float64 za crtanje)
        lp_sanit = pref.replace("/", "_").replace(".", "_").replace(":", "_")
        hist_all = os.path.join(args.outdir, f"hist_absDelta_ALL_{lp_sanit}.png")
        make_histogram_torch(d, f"|Δbias| ALL — {pref}", hist_all, bins=50, log=False)
        if d_T is not None and d_T.numel() > 0:
            hist_T = os.path.join(args.outdir, f"hist_absDelta_T_{lp_sanit}.png")
            make_histogram_torch(d_T, f"|Δbias| T — {pref}", hist_T, bins=50, log=False)

        # Za ukupno T
        if d_T is not None and d_T.numel() > 0:
            overall_T_abs_list.append(d_T)

    # Overall metri ke (ALL)
    ov = {}
    if overall_all_abs_list:
        overall_all_abs = torch.cat(overall_all_abs_list, dim=0)
        thr_all = torch.as_tensor(thr_val, dtype=overall_all_abs.dtype, device=overall_all_abs.device)
        ov["overall_mean_abs_delta_all"] = float(overall_all_abs.mean().item())
        ov["overall_max_abs_delta_all"]  = float(overall_all_abs.max().item())
        ov["overall_frac_exceed_all"]    = float((overall_all_abs > thr_all).sum().item() / max(1, overall_all_abs.numel()))
        # Histogram overall ALL
        make_histogram_torch(overall_all_abs, "|Δbias| ALL — OVERALL", os.path.join(args.outdir, "hist_absDelta_ALL_overall.png"))
    else:
        ov["overall_mean_abs_delta_all"] = float("nan")
        ov["overall_max_abs_delta_all"]  = float("nan")
        ov["overall_frac_exceed_all"]    = float("nan")

    # Overall metrike (T)
    if overall_T_abs_list:
        overall_T_abs = torch.cat(overall_T_abs_list, dim=0)
        thr_T = torch.as_tensor(thr_val, dtype=overall_T_abs.dtype, device=overall_T_abs.device)
        ov["overall_mean_abs_delta_T"] = float(overall_T_abs.mean().item())
        ov["overall_max_abs_delta_T"]  = float(overall_T_abs.max().item())
        ov["overall_frac_exceed_T"]    = float((overall_T_abs > thr_T).sum().item() / max(1, overall_T_abs.numel()))
        ov["overall_delta_star_T_pct"] = float(args.percentile)
        qT = torch.tensor([args.percentile/100.0], dtype=overall_T_abs.dtype, device=overall_T_abs.device)
        ov["overall_delta_star_T"]     = float(torch.quantile(overall_T_abs, qT).item())
        # Histogram overall T
        make_histogram_torch(overall_T_abs, "|Δbias| T — OVERALL", os.path.join(args.outdir, "hist_absDelta_T_overall.png"))
    else:
        ov["overall_mean_abs_delta_T"] = float("nan")
        ov["overall_max_abs_delta_T"]  = float("nan")
        ov["overall_frac_exceed_T"]    = float("nan")
        ov["overall_delta_star_T_pct"] = float(args.percentile)
        ov["overall_delta_star_T"]     = float("nan")

    # Save per-layer CSV
    df = pd.DataFrame(layer_rows)
    df.to_csv(os.path.join(args.outdir, "layerwise_bias_delta.csv"), index=False)

    # Save overall CSV
    ov_df = pd.DataFrame([ov])
    ov_df.to_csv(os.path.join(args.outdir, "overall_bias_delta.csv"), index=False)

    if args.verbose:
        print("\n=== SUMMARY ===")
        print("overall_frac_exceed_T (>|Δ|>{}):".format(args.threshold), ov["overall_frac_exceed_T"])
        print("overall_delta_star_T@{}:".format(args.percentile), ov["overall_delta_star_T"])
        print("Outputs in:", args.outdir)

if __name__ == "__main__":
    main()

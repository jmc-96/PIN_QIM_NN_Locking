# =============================================================================
#%%                            EKSTRAKCIJA QIM
# =============================================================================
import numpy as np
import C_QIM

def eKstrakcija_QIM(y, bit_duzina, Δ):
    '''
      - y: 1D nosilac (flatenovan)
      - bit_duzina: #bitska duzina (bitova po PIN simbolu)
      - Δ: QIM kvantizacioni korak (isti kao u ugradnji)    
    '''
    y = C_QIM.izdvoji_zigosan(y, bit_duzina)
    u_hat = C_QIM.extract_qim(y, Δ=Δ)
    return np.asarray(u_hat, dtype=int)

# =============================================================================
#%%                          EKSTRAKCIJA SparseQIM
# =============================================================================
import C_SparseQIM as sQIM

def eKtrakcija_sQIM(y, bit_duzina, pin_duzina, Δ):
    """
    Sparse QIM ekstrakcija (fiksni seed=42 u C_SparseQIM).
      - y: 1D nosilac (flatenovan)
      - bit_duzina: #bitska duzina (bitova po PIN simbolu)
      - Δ: QIM kvantizacioni korak (isti kao u ugradnji)
      - T: ako None → izračuna se kao len(y)//L (kao u embed-u)
      - pin_duzina: broj PIN simbola → L = pin_duzina * bit_duzina
    """
    y = np.asarray(y, float).ravel()
#    L = int(pin_duzina) * int(bit_duzina)

    # Poziv na C_SparseQIM (seed=42 je unutra fiksan)
    if hasattr(sQIM, "sQIM_extract"):
        u_hat = sQIM.sQIM_extract(y, pin_duzina, bit_duzina, Δ=Δ)
    else:
        raise ImportError("C_SparseQIM mora imati 'sQIM_extract(y, L, Δ, T=None)'.")
    return np.asarray(u_hat, dtype=int)

# ----------------------------------------------------------------------------- 
# %%                        DETEKCIJA WATERMARKA
# -----------------------------------------------------------------------------

def dtype_32_or_64(mreza):
    """
    Vraca (y, info):
      - y: 1D numpy vektor za ekstrakciju ili None ako nije watermarked
      - info: dict sa kljucevima:
          source: "W" | "b" | None
          case:   1 | 2 | 3 | 4
          index:  int ili None (za case 1)
          message: objašnjenje
    """
    list_W, list_b = mreza

    # ---------------------------- Provera list_W -----------------------------
    
    w_dtypes = [getattr(a, "dtype", None) for a in list_W]
    w_is64 = [dt == np.float64 for dt in w_dtypes]
    cnt_w64 = sum(w_is64)

    if cnt_w64 == 1:
        idx = w_is64.index(True)
        y = list_W[idx].ravel(order='C')
        info = {
            "source": "W",
            "case": 1,
            "index": idx,
            "message": "Watermarkovan 0-ti embeding weights layer."
        }
        return y, info

    if len(list_W) > 0 and cnt_w64 == len(list_W):
        y = np.concatenate([a.ravel(order='C') for a in list_W], axis=0)
        info = {
            "source": "W",
            "case": 2,
            "index": None,
            "message": "Kompletan skup embeddings weight koeficijenata je watermarkovan."
        }
        return y, info

    # ---------------------------- Provera list_b -----------------------------
    
    b_dtypes = [getattr(a, "dtype", None) for a in list_b]
    if len(b_dtypes) == 0:
        return None, {
            "source": None,
            "case": 4,
            "index": None,
            "message": "Model nije watermarked."
        }

    all_b64 = all(dt == np.float64 for dt in b_dtypes)
    all_b32 = all(dt == np.float32 for dt in b_dtypes)

    if all_b64:
        y = np.concatenate([a.ravel(order='C') for a in list_b], axis=0)
        return y, {
            "source": "b",
            "case": 3,
            "index": None,
            "message": "Bias layer je watermarkovan."
        }

    if all_b32:
        return None, {
            "source": "b",
            "case": 4,
            "index": None,
            "message": "Model nije watermarked."
        }

    # Mešovito (ne bi smelo da se desi) -> tretiramo kao ne-watermarkovano
    return None, {
        "source": "b",
        "case": 4,
        "index": None,
        "message": "Model nije watermarked."
    }

# ----------------------------------------------------------------------------- 
# %%               DETEKCIJA DECIMALNOG IZ BINARNOG BROJA
# -----------------------------------------------------------------------------

def binarniUdecimalni(mreza, oznaka, bit_duzina):
    # Odredjujemo da li su float 32 ili 64
    y, info = dtype_32_or_64(mreza)
    print("==========================================================")
    print("[INFO]", info["message"])
    print("==========================================================")

    if y is None:
        print("decimalni = []")
        return []
    
    print("==========================================================")
    print("\n                Izbor ekstrakcije ")
    print("----------------------------------------------------------")
    print("                     1) QIM")
    print("                     2) Sparse QIM")
    print("==========================================================")
    izbor = (input("Izbor [1/2, default 1]: ").strip() or "1")
    # ----------------------- zajednicki unos Δ -------------------------------
    delta_in = input("Unesite Δ (npr. 0.1) [Enter za 0.1]: ").strip()
    try:
        delta = float(delta_in) if delta_in else 0.1
    except Exception:
        print("[WARN] Nevažeći unos za Δ, koristim 0.1")
        delta = 0.1

    if izbor == "2":
        # fiksni seed=42 je u C_SparseQIM, zato ovde više ne pitamo za seed
        pin_in = input("PIN dužina (broj simbola): ").strip()
        if not pin_in:
            raise ValueError("Za Sparse QIM navedi PIN dužinu (broj simbola).")
        pin_duzina = int(pin_in)
        binarni = eKtrakcija_sQIM(y, bit_duzina, pin_duzina, Δ=delta)
    else:
        binarni = eKstrakcija_QIM(y, bit_duzina, Δ=delta)

    # ------------------------- BINARNI U DECIMALNI ---------------------------
    binarni = np.array(binarni, dtype=int)
    assert len(binarni) % bit_duzina == 0, "Duzina mora biti deljiva sa bitskom duzinom"

    reshaped = binarni.reshape(-1, bit_duzina)
    powers = 2 ** np.arange(bit_duzina - 1, -1, -1)
    decimalni = reshaped.dot(powers)

    print("==========================================================")
    print("Extractovani PIN: ", decimalni)
    print("==========================================================")
    return decimalni.reshape(1, -1)

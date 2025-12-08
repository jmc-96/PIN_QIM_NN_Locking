# =============================================================================
#%%                             BITSKA DUZINA
# =============================================================================

# Obezbeđuje bit_duzinu i oznaku ("b" ili "W") za ostale funkcije.
# Parsira iz imena fajla (PIN_n i Watermarked_[b|W]_PIN) i računa bit_duzinu
# iz prvih pin_duzina elemenata ucitanog ključa.

import re
import numpy as np
import C_Ucitavanje as cu

def bit_duzina_iz_kljuca(ime_key: str | None = None, kljuc=None):
    """
    Vrati: (bit_duzina:int, oznaka:str) gde je oznaka 'b' ili 'W'.

    Ako kljuc nije prosleđen, učitaće se pomoću C_Ucitavanje.uciTavanje(ime_key).
    Ako ime_key nije prosleđen, pitaće se kroz input().
    """
    if ime_key is None:
        ime_key = input("Unesite ime fajla za ucitavanje kljuca: ").strip()

    if kljuc is None:
        kljuc = cu.uciTavanje(ime_key)

    # PIN dužina iz imena fajla
    m_pin = re.search(r"PIN_(\d+)", ime_key)
    if not m_pin:
        raise ValueError("Nije pronađen uzorak 'PIN_<broj>' u imenu fajla.")
    pin_duzina = int(m_pin.group(1))

    # Oznaka b/W iz imena fajla (npr. ...Watermarked_b_PIN_10.txt)
    m_ozn = re.search(r"Watermarked_(\w)_PIN", ime_key)
    if not m_ozn:
        raise ValueError("Nije pronađena oznaka 'Watermarked_[b|W]_PIN' u imenu fajla.")
    ozn_raw = m_ozn.group(1)
    oznaka = "b" if ozn_raw.lower() == "b" else "W"

    # Bitska dužina iz vrednosti prvih pin_duzina elemenata ključa
    arr = np.asarray(kljuc).flatten()
    if arr.size < pin_duzina:
        raise ValueError(f"Ključ ima {arr.size} elemenata, a očekivano je ≥ {pin_duzina}.")
    pin_vals = arr[:pin_duzina].astype(int)
    max_val = int(np.max(pin_vals)) if pin_vals.size else 0
    bit_duzina = max(1, max_val.bit_length())

    print("----------------------------------------------------------")
    print("PIN dužina:", pin_duzina)
    print("Bitska dužina PIN-a:", bit_duzina)
    print("Watermarkovano na:", "biases" if oznaka == "b" else "weights")
    print("----------------------------------------------------------")

    return bit_duzina, pin_duzina, oznaka
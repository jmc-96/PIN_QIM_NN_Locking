# ============================================================================= 
#                                Sparse QIM
# =============================================================================

import numpy as np

# ----------------------------------------------------------------------------- 
#%%                                  HELPERS
# -----------------------------------------------------------------------------

def _plan_alpha(T, seed):
    '''Jedan nasumičan jedinični smer dužine T (||alpha||=1).'''
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(T)
    n = np.linalg.norm(a)
    if n < 1e-12:
        a = np.zeros(T); a[0] = 1.0; n = 1.0
    return a / n

def _embed_bit_block(block, bit, Δ, alpha):
    '''Ugradi 0/1 u 'block' projekcijom na alpha i QIM kosetima C0=kΔ, C1=kΔ+Δ/2.'''
    proj = float(block @ alpha)
    if bit == 0:
        q = np.round(proj / Δ) * Δ                 # kΔ
    else:
        q = np.round((proj - Δ/2) / Δ) * Δ + Δ/2  # kΔ + Δ/2
    return block + (q - proj) * alpha

def _extract_bit_block(block, Δ, alpha):
    '''Izvuci 0/1 iz bloka (jedna detekcija).'''
    proj = float(block @ alpha)
    q0 = np.round(proj / Δ) * Δ                       # kΔ
    q1 = np.round((proj - Δ/2) / Δ) * Δ + Δ/2 # kΔ + Δ/2
    d0 = abs(proj - q0)
    d1 = abs(proj - q1)
    # tie-break ka 0 da se izbegne bias ka 1
    return 1 if (d1 < d0) else 0

def bits_to_ints(bits, bit_duzina):
    '''Pretvori niz bitova (0/1) u niz celih brojeva po blokovima od 'bit_duzina'.'''
    bits = np.asarray(bits)
    L = bits.size
    if L % bit_duzina != 0:
        raise ValueError(f"Ukupan broj bitova ({L}) nije deljiv sa bit_duzina={bit_duzina}.")
    nums = []
    for i in range(0, L, bit_duzina):
        v = 0
        for b in bits[i:i+bit_duzina]:
            if np.isnan(b):
                v = np.nan
                break
            v = (v << 1) | int(b)
        nums.append(v)
    return np.array(nums, dtype=float)

# ----------------------------------------------------------------------------- 
#%%                                  EMBEDDING
# -----------------------------------------------------------------------------

def sQIM_embed(x, u_bits, Δ):
    '''
    Embedding po mojim pravilima (sekvencijalno bit-po-bit u uzastopne T-blokove):
      1) Ulaz: x (1D), u_bits (0/1), Δ (Δ)
      2) T se izračuna kao T = len(x) // L, gde je L=len(u_bits)
      3) Upiši L bitova redom u prva L*T uzorka (blokovi dimenzije T). Ostatak x ostaje.
      4) Vrati (y, bit_duzina). Ovde bit_duzina nije poznata funkciji (jer velicina simbola nije data),
         pa vraćam None.
    '''
    x = np.asarray(x, float).ravel().copy()
    u_bits = np.asarray(u_bits).ravel()
    if u_bits.size == 0:
        raise ValueError("u_bits ne sme biti prazan.")
    if not np.all((u_bits == 0) | (u_bits == 1)):
        raise ValueError("u_bits mora biti 0/1.")
    L = int(u_bits.size)
    N = int(x.size)
    T = N // L
    if T <= 0:
        raise ValueError(f"Nedovoljno podataka: N={N}, L={L} -> T={T} (<=0).")
    # Radimo samo nad prvim L*T uzoraka
    blocks = x[:L*T].reshape(L, T)
    y_blocks = np.empty_like(blocks)

    # Za determinističnost, koristimo različit seed po bitu (npr. 1000+i)
    for i, (bit, block) in enumerate(zip(u_bits, blocks)):
        alpha = _plan_alpha(T, seed=1000 + i)
        y_blocks[i] = _embed_bit_block(block, int(bit), float(Δ), alpha)

    y = y_blocks.reshape(-1)
    if L*T < N:
        y = np.concatenate([y, x[L*T:]])

    # bit_duzina nije poznata ovoj funkciji (jer prima već "spakovane" bitove).
    bit_duzina = None  # <- ako želiš, dodaj argument pa vrati njegovu vrednost
    return y

# ----------------------------------------------------------------------------- 
# %%                                 EXTRACTION
# -----------------------------------------------------------------------------

def sQIM_extract(y, pin_duzina, bit_duzina, Δ):
    """
    Ekstrakcija po tvojim pravilima:
      1) Ulaz: y, pin_duzina, bit_duzina, Δ
      2) T se izračuna kao T = len(y) // (pin_duzina*bit_duzina)  (isti način kao embed)
      3) Ako dtype(y) == float32 -> nema watermarka, vrati PINex=[], u_hat_bits=[]
      4) Ako dtype(y) == float64 -> izvlaci se od pocetka, pravi se T nizova procena (po jedna sa seed=1000+i, shiftovana po "i"),
         pa se radi većinsko glasanje po bit poziciji; nečitljivo -> NaN
      5) Porede se svih T procena; na svakoj poziciji uzme se najčešći bit (0/1). Ako sve propadnu -> NaN.
      6) Na mestima gde su sve vrednosti NaN, ostaje NaN
      7) Ako postoji makar jedan NaN u finalu: prijavi “ekstrakcija nije moguća” uz broj oštećenih pozicija.
      8) Inače konvertuj u decimalni PIN po segmentima od bit_duzina.
    """
    y = np.asarray(y).ravel()
    L = int(pin_duzina) * int(bit_duzina)
    N = int(y.size)
    if y.dtype == np.float32:
        # Nema watermarka
        return [], np.array([], dtype=float)

    if y.dtype != np.float64 and y.dtype != float:
        # I dalje možemo pokušati, ali napomenimo da očekujemo float64
        y = y.astype(np.float64, copy=False)

    if L <= 0:
        raise ValueError("L=pin_duzina*bit_duzina mora biti > 0.")
    T = N // L
    if T <= 0:
        raise ValueError(f"Nedovoljno podataka: N={N}, L={L} -> T={T} (<=0).")

    # Radi se samo nad prvih L*T uzoraka
    blocks = y[:L*T].reshape(L, T)

    # Napravimo T nezavisnih procena (različiti seed-ovi daju različite smerove)
    votes = np.full((T, L), np.nan, dtype=float)
    for t in range(T):
        # isti alfa kao u embedu za odgovarajući bit i=0..L-1: seed=1000+i
        # (ovo odgovara “od pocetka” bez kruženja; svaka procena koristi svoj smer)
        for i in range(L):
            try:
                alpha = _plan_alpha(T, seed=1000 + i)  # isti kao u embed-u
                bit_est = _extract_bit_block(blocks[i], float(Δ), alpha)
                votes[t, i] = bit_est
            except Exception:
                votes[t, i] = np.nan  # gde ne može da se pročita -> NaN

    # Većinsko glasanje po koloni (po bit poziciji)
    u_hat = np.full(L, np.nan, dtype=float)
    for i in range(L):
        col = votes[:, i]
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            u_hat[i] = np.nan
        else:
            # većinski bit; u slučaju 50-50, opredeli se za 0
            c0 = np.sum(valid == 0)
            c1 = np.sum(valid == 1)
            u_hat[i] = 1.0 if (c1 > c0) else 0.0

    # Provera oštećenja
    num_nan = int(np.sum(np.isnan(u_hat)))
    if num_nan > 0:
        total = L
        print(f"Ekstrakcija NIJE moguća: oštećeno {num_nan}/{total} bit pozicija.")
        return [], u_hat  # vrati samo bitove (sa NaN) bez PIN-a

#    # Svi bitovi pročitani -> formiraj decimalni PIN
 #   pin_vals = bits_to_ints(u_hat, int(bit_duzina))  # float (ali bez NaN), može cast u int
  #  PINex = pin_vals.astype(int).tolist()
    return u_hat

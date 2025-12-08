# =============================================================================
#                       OTKLJUCAVANJE POMOCU KLJUCA
# =============================================================================

import numpy as np
import os

# %% ----------------------- UCITAVANJE KLJUCA --------------------------------

def ucitaj_kljuc():
    print("==========================================================")
    folder = input("Unesi folder za ucitavanje kljuca (default: data/key): ").strip()
    if folder == "":
        folder = "data/key"
    print("----------------------------------------------------------")
    fname = input("Unesi ime fajla kljuca (default: key.txt):\n > ").strip() or "key.txt"
    print("==========================================================")
    path = os.path.join(folder, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fajl '{path}' ne postoji.")

    kljuc = np.loadtxt(path, dtype=int)
    return kljuc


# %% --------------------------- PRETUMBACIJA ---------------------------------

def perMutacija(vektor: np.ndarray, kljuc: np.ndarray, copy=True):
    """
    Otključavanje permutacijom: primeni inverznu permutaciju redova/elemenata zadatu ključem.
    Radi i za 1D i za 2D (redovi) niz.
    """
    v = vektor.copy() if copy else vektor

    # Normalizuj ključ na oblik (2, k) i int tip
    K = np.asarray(kljuc)
    if K.ndim != 2 or not (K.shape[0] == 2 or K.shape[1] == 2):
        raise ValueError("kljuc mora biti oblika (2, k) ili (k, 2)")
    if K.shape[1] == 2:
        K = K.T
    K = K.astype(np.int64, copy=False)
    a_idx, b_idx = K[0], K[1]

    if v.ndim == 1:
        n = v.shape[0]
        p = np.arange(n)
        for a, b in zip(a_idx, b_idx):
            if not (0 <= a < n and 0 <= b < n):
                raise IndexError("Indeksi u ključu su van opsega za 1D vektor.")
            p[a], p[b] = p[b], p[a]
        # inverzna permutacija
        inv = np.empty_like(p)
        inv[p] = np.arange(n)
        return v[inv]

    elif v.ndim == 2:
        r = v.shape[0]
        p = np.arange(r)
        for a, b in zip(a_idx, b_idx):
            if not (0 <= a < r and 0 <= b < r):
                raise IndexError(f"Indeksi u ključu su van opsega za broj redova={r}.")
            p[a], p[b] = p[b], p[a]
        # inverzna permutacija redova
        inv = np.empty_like(p)
        inv[p] = np.arange(r)
        return v[inv, :]

    else:
        raise ValueError("vektor mora biti 1D ili 2D numpy niz")

# %% -------------------------- ZAMENA MREZE ----------------------------------

def zameni_mrezu(mreza, host, nova_lista):
    mapa = {"list_W": 0, "list_b": 1}
    i = mapa[host]             # indeks u tuple
    mr = list(mreza)           # konvertuj u listu
    mr[i] = nova_lista         # zameni element
    return tuple(mr)           # vrati tuple

# %% -------------------------- OTKLJUCAVANJE ---------------------------------

def otKljucavanje(mreza, em_pos, out_dir="data/key"):
    
    # UCITAVANJE KLJUCA
    kljuc = ucitaj_kljuc()
    # UCITAVANJE MREZE 
    list_W, list_b = mreza    
    print("==========================================================")
    print("       U toku je proces otkljucavanja modela.")
    print("==========================================================")
    # OTKLJUCAVANJE PRETUMBACIJA PREMA KLJUCU
    vek_em = list_W[em_pos]
    shape_em = vek_em.shape
    # PERMUTACIJA UNLOCK
    vek_em_unlocked = perMutacija(vek_em, kljuc, copy=True)   
    # Provera
    assert vek_em_unlocked.shape == shape_em, "Ne poklapa se broj elemenata/dimenzija posle permutacije."
    # uporedi samo vek_em_locked i vek_em
    print("==========================================================")
    razlike = ~(vek_em_unlocked == vek_em)
    print("Za duzina kljuca ", kljuc.shape[1]," i dimenziju embeding")
    print("vektora ", shape_em[1]," ocekivani broj pozicija je: ",2*shape_em[1]*kljuc.shape[1])
    print("Broj izmerenih promenjenih pozicija je:", np.sum(razlike))
    print("==========================================================")
    nastavak = '_unLocked'
        
    # FORMIRANJE NOVE UNLOCKED MREZE
    list_W_unlocked = list_W.copy()
    list_W_unlocked[em_pos] = np.ascontiguousarray(vek_em_unlocked)
    host = "list_W"
    mreza = zameni_mrezu(mreza, host, list_W_unlocked)
           
    return mreza, kljuc, nastavak
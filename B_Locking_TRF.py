# =============================================================================
#                        ZAKLJUCAVANJE PRETUMBACIJOM
# =============================================================================

import numpy as np

# %% --------------------------- PERMUTACIJA ----------------------------------

def perMutacija(vektor: np.ndarray, kljuc: np.ndarray, copy=True):
    v = vektor.copy() if copy else vektor

    # Normalizuj ključ na oblik (2, k) i int tip
    kljuc = np.asarray(kljuc)
    if kljuc.ndim != 2 or not (kljuc.shape[0] == 2 or kljuc.shape[1] == 2):
        raise ValueError("kljuc mora biti oblika (2, k) ili (k, 2)")
    if kljuc.shape[1] == 2:
        kljuc = kljuc.T  # sada (2, k)
    kljuc = kljuc.astype(np.int64, copy=False)

    if v.ndim == 1:
        # 1D permutacija elemenata
        n = v.shape[0]
        a_idx, b_idx = kljuc[0], kljuc[1]
        if a_idx.size == 0:
            raise ValueError("Ključ nema parova (dužina je 0).")
        if np.any(a_idx == b_idx):
            raise ValueError("Ključ sadrži parove sa istim indeksom (a==b).")
        if np.any((a_idx < 0) | (a_idx >= n) | (b_idx < 0) | (b_idx >= n)):
            raise IndexError("Indeksi u ključu su van opsega za 1D vektor.")
        for a, b in zip(a_idx, b_idx):
            v[a], v[b] = v[b], v[a]
    elif v.ndim == 2:
        # 2D permutacija REDOVA po ključu
        rows = v.shape[0]
        a_idx, b_idx = kljuc[0], kljuc[1]
        if a_idx.size == 0:
            raise ValueError("Ključ nema parova (dužina je 0).")
        if np.any(a_idx == b_idx):
            raise ValueError("Ključ sadrži parove sa istim indeksom (a==b).")
        if np.any((a_idx < 0) | (a_idx >= rows) | (b_idx < 0) | (b_idx >= rows)):
            raise IndexError(
                f"Indeksi u ključu su van opsega za broj redova={rows}."
            )
        for a, b in zip(a_idx, b_idx):
            tmp = v[[a, b]].copy()
            v[[a, b]] = tmp[::-1]
            
        import J_Test_permutacija as jtp
        print("----------------------------------------------------------------")
        ok = jtp.proveri_permutaciju_redova(vektor, v, kljuc, atol=1e-8)
        print("----------------------------------------------------------------")
    else:
        raise ValueError("vektor mora biti 1D ili 2D numpy niz")

    return v
  
# %% -----------------------  PIN WATERMARKING --------------------------------

def PINzigosanje(mreza, kljuc):
    import B_Watermarking_TRF as BW
    mreza_w, ime_w = BW.vodeniZig(mreza, kljuc)
    return mreza_w, ime_w

# %% -------------------------- ZAMENA MREZE ----------------------------------

def zameni_mrezu(mreza, host, nova_lista):
    mapa = {"list_W": 0, "list_b": 1}
    i = mapa[host]             # indeks u tuple
    mr = list(mreza)           # konvertuj u listu
    mr[i] = nova_lista         # zameni element
    return tuple(mr)           # vrati tuple

# %% -------------------------- ZAKLJUCAVANJE ---------------------------------

def zaKljucavanje(mreza, ime, out_dir="data/key"):
    
    # UCITAVANJE MREZE
    import os  
    # FORMIRANJE KLJUCA
    import B_Kljuc as BK
    
    # ------------------------- Prvi MENU ------------------------ 
    print("==========================================================")
    print("                 IZABERI LOCKING MEHANIZAM                ")
    print("==========================================================")
    datasets = ["Double-Lock", "Watermark", "Lock", "Odustani"]
    for i, n in enumerate(datasets):
        print(f"  {i}) {n}")
    print("==========================================================")
    try:
        ds_idx = int(input(f"Izbor [0-{len(datasets)-1}]: ").strip())
        assert 0 <= ds_idx < len(datasets)
        print("==========================================================")
    except Exception:
        print("Nevažeći izbor."); return None, None, None      
    # DOUBLE-LOCK mehanizam
    if ds_idx == 0:
        print("==========================================================")
        print("       Odabrali ste Double-lock mehanizam zastite.")
        print("==========================================================")
        # ---------------------------------------------------------------------
        #                    ZAKLJUCAVANJE PRETUMBACIJA
        # ---------------------------------------------------------------------
        list_W, list_b = mreza    
        Ws = [np.asarray(W) for W in list_W]
        col_dims = [W.shape[1] if W.ndim >= 2 else 1 for W in Ws]
        # Ako je ucitan samo jedan embedings ili se razlikuju dimenzije svih
        if (len(Ws) == 1) or (len(set(col_dims)) != 1):
            # jedan sloj ili neusklađene druge dimenzije → uzimamo prvi
            W0 = Ws[0]
            vek_em   = W0
            shape_em = W0.shape
        else:
            # usklađene druge dimenzije → spajamo u jednu 2D matricu
            ncol = col_dims[0]
            Ws2 = [W if W.ndim == 2 else W.reshape(-1, ncol) for W in Ws]  # 1D → (D,1) ako treba
            vek_em   = np.concatenate(Ws2, axis=0)
            shape_em = {"parts": tuple(W.shape for W in Ws2)}
        # jedinstveno formiranje vektora
        vektor_kljuc = vek_em[:, 0].ravel()
        # ------------------------ FORMIRANJE KLJUCA --------------------------
        kljuc, duzina, tip = BK.kljuc(vektor_kljuc)
        # ------------------------- PIN WATERMARKING --------------------------
        mreza_w, ime_w = PINzigosanje(mreza, kljuc)
        # -------------------------- PERMUTACIJA ------------------------------
        vek_em_locked = perMutacija(vek_em, kljuc, copy=True)
        # Provera
        assert vek_em_locked.shape == vek_em.shape, "Ne poklapa se broj elemenata/dimenzija posle permutacije."
        # uporedi samo vek_em_locked i vek_em
        print("==========================================================")
        razlike = ~(vek_em_locked == vek_em)
        print("   Broj promenjenih pozicija:", np.sum(razlike))
        print("==========================================================")
        nastavak = '_Locked' + str(duzina)
    # -------------------------------------------------------------------------
    #                               WATERMARK SAMO
    # -------------------------------------------------------------------------
    elif ds_idx == 1:
        print("==========================================================")
        print("       Odabrali ste samo da ugradite vodeni zig.")
        print("==========================================================")
        list_W, list_b = mreza 
        vek_em   = list_W[0]
        vektor_kljuc = vek_em[:, 0].ravel()
        vek_em_locked = vek_em
        kljuc, duzina, tip = BK.kljuc(vektor_kljuc)
        # PIN WATERMARKING
        mreza_w, ime_w = PINzigosanje(mreza, kljuc)
        nastavak = ''
    # -------------------------------------------------------------------------
    #                                 SAMO LOCK
    # -------------------------------------------------------------------------
    elif ds_idx == 2:
        print("==========================================================")
        print("       Odabrali ste zakljucavanje bez vodenog ziga.")
        print("==========================================================")
        # ZAKLJUCAVANJE PRETUMBACIJA
        # Zaključavanje embeding sloja
        list_W, list_b = mreza    
        Ws = [np.asarray(W) for W in list_W]
        col_dims = [W.shape[1] if W.ndim >= 2 else 1 for W in Ws]
        
        if (len(Ws) == 1) or (len(set(col_dims)) != 1):
            # jedan sloj ili neusklađene druge dimenzije → uzimamo prvi
            W0 = Ws[0]
            vek_em   = W0
            shape_em = W0.shape
        else:
            # usklađene druge dimenzije → spajamo u jednu 2D matricu
            ncol = col_dims[0]
            Ws2 = [W if W.ndim == 2 else W.reshape(-1, ncol) for W in Ws]  # 1D → (D,1) ako treba
            vek_em   = np.concatenate(Ws2, axis=0)
            shape_em = {"parts": tuple(W.shape for W in Ws2)}
        # jedinstveno formiranje vektora
        vektor_kljuc = vek_em[:, 0].ravel()
        # ------------------------ FORMIRANJE KLJUCA --------------------------
        kljuc, duzina, tip = BK.kljuc(vektor_kljuc)
        # -------------------------- PERMUTACIJA ------------------------------
        vek_em_locked = perMutacija(vek_em, kljuc, copy=True)
        # Provera
        assert vek_em_locked.shape == vek_em.shape, "Ne poklapa se broj elemenata/dimenzija posle permutacije."
        # uporedi samo vek_em_locked i vek_em
        print("==========================================================")
        razlike = ~(vek_em_locked == vek_em)
        print("   Broj promenjenih pozicija:", np.sum(razlike))
        print("==========================================================")
        ime_w = ""
        nastavak = '_Locked' + str(duzina)
    # -------------------------------------------------------------------------
    #                                     EXIT
    # -------------------------------------------------------------------------
    elif ds_idx == 3:
        print(">> Prekid rada.")
        print("==========================================================")
        return None, None, None
    # -------------------------------------------------------------------------
    #                          FORMIRANJE NOVE LOCKED MREZE
    # -------------------------------------------------------------------------
    
    shapes = (shape_em["parts"] if isinstance(shape_em, dict) else (shape_em,))
    sizes  = [int(np.prod(s)) for s in shapes]
    splits = np.cumsum(sizes)[:-1]
    parts  = np.split(vek_em_locked.ravel(), splits)
    list_W_restored = [np.ascontiguousarray(p.reshape(s)) for p, s in zip(parts, shapes)]
    
    host = "list_W"
    mreza = zameni_mrezu(mreza_w, host, list_W_restored)
        # Snimamo kljuc
    ime_LW = nastavak + ime_w
    os.makedirs(out_dir, exist_ok=True)
    baza = os.path.splitext(ime)[0]
    out_path = os.path.join(out_dir, f"{baza}{ime_LW}.txt")    
    # Snimamo kljuc kao txt file
    np.savetxt(out_path, kljuc, fmt="%d", delimiter="\t")  # - integer     
        
    return mreza, ime_LW, kljuc
# %% ZAKLJUCAVANJE PRETUMBACIJOM

# %% WEIGHT KOEFICIJENTI ZA LOCKOVANJE

import numpy as np

def flat_w(mreza):
    # Koristimo list_W za zakljucavanje
    n = len(mreza)
    if n == 2:
        list_W, list_b = mreza
    elif n == 3:
        list_W, list_b, list_T = mreza
    elif n == 4:
        list_W, list_b, list_s, list_r = mreza[:4]
    else:
        raise ValueError(f"{n} - Nevalidan format mreze. Očekivana dužina 2 za Transformere, 4 za CNN ili 3 Probno.")
    if not isinstance(list_W, (list, tuple)) or not isinstance(list_b, (list, tuple)):
        raise TypeError("list_W i list_b moraju biti liste/tuple.")

    shapes = [a.shape for a in list_W]
    vektor = np.concatenate([a.ravel(order='C') for a in list_W], axis=0)
    print("==================================================================")
    print("Vektor weight za zakljucavanje je duzine: ", len(vektor))
    print("==================================================================")
    host = "list_W"
    return vektor, shapes, host


# %% PRETUMBACIJA

def perMutacija(vektor: np.ndarray, kljuc: np.ndarray, copy=True):
    v = vektor.copy() if copy else vektor

    # podrži oba formata ključa
    if kljuc.shape[0] == 2:
        # shape (2, k) -> red 0: bot, red 1: top
        for a, b in zip(kljuc[0], kljuc[1]):
            v[a], v[b] = v[b], v[a]
    elif kljuc.shape[1] == 2:
        # shape (k, 2) -> kolone: (bot, top)
        for a, b in kljuc:
            v[a], v[b] = v[b], v[a]
    else:
        raise ValueError("kljuc mora biti oblika (2, k) ili (k, 2)")

    return v
    
# %% PIN WATERMARKING

def PINzigosanje(mreza, kljuc):
    import B_Watermarking_CNN as BW
    mreza, ime_w = BW.vodeniZig(mreza, kljuc)
    return mreza, ime_w

# %% ZAMENA MREZE

def zameni_mrezu(mreza, host, nova_lista):
    mapa = {"list_W": 0, "list_b": 1, "list_s": 2, "list_r": 3}
    i = mapa[host]             # indeks u tuple
    mr = list(mreza)           # konvertuj u listu
    mr[i] = nova_lista         # zameni element
    return tuple(mr)           # vrati tuple

# %% ZAKLJUCAVANJE

def zaKljucavanje(mreza, ime, fc_pos, out_dir="data/key"):
    
    # UCITAVANJE MREZE
    import os
    vektor, shapes, host = flat_w(mreza)
    sizes = [int(np.prod(shp)) for shp in shapes]   
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
        # ZAKLJUCAVANJE PRETUMBACIJA
        print("----------------------------------------------------------")
        print("   IZABERI OBLAST ZAKLJUCAVANJA")
        print("----------------------------------------------------------")
        opcije = [
            "Sve weight koeficijente",
            "Samo klasifikacioni sloj"]
        for i, n in enumerate(opcije, 1):
            print(f"  {i}) {n}")
        try:
            oblast_idx = int(input(f"Izbor [1-{len(opcije)}]: ").strip())
            assert 1 <= oblast_idx <= len(opcije)
        except Exception:
            print("Nevažeći izbor.")
            return None, None, None
        print("----------------------------------------------------------")        
        # 1) Zaključavanje svih weight koeficijenata
        if oblast_idx == 1:
            kljuc, duzina, tip = BK.kljuc(vektor)
            # PIN WATERMARKING
            mreza, ime_w = PINzigosanje(mreza, kljuc)
            # PERMUTACIJA
            vektor_locked = perMutacija(vektor, kljuc)
        # 2) Zaključavanje samo klasifikacionog sloja
        elif oblast_idx == 2:
            # Uzimamo iz list_W  fc - vektor klasifikacionog sloja.
            list_W, list_b, list_s, list_r = mreza    
            vek_fc = list_W[fc_pos]
            shape_fc = vek_fc.shape
            vek_fc = vek_fc.flatten()
            kljuc, duzina, tip = BK.kljuc(vek_fc)
            # PIN WATERMARKING
            mreza, ime_w = PINzigosanje(mreza, kljuc)
            # PERMUTACIJA
            vek_fc_locked = perMutacija(vek_fc, kljuc)
            # Vracanje 
            assert vek_fc.size == np.prod(shape_fc), "Ne poklapa se broj elemenata za reshape."
            vek_fc_locked = vek_fc_locked.reshape(shape_fc)
            list_W[fc_pos] = vek_fc_locked                
            vektor_locked = np.concatenate([a.ravel(order='C') for a in list_W], axis=0)     
        # Nastavak
        nastavak = '_Locked' + str(duzina)
        print("==========================================================")
        razlike = ~(vektor_locked == vektor)
        print("   Broj promenjenih pozicija:", np.sum(razlike))
        print("Pozicije gde se razlikuju:", np.where(razlike)[0])
        print("==========================================================")        
    # WATERMARK SAMO
    elif ds_idx == 1:
        print("==========================================================")
        print("       Odabrali ste samo da ugradite vodeni zig.")
        print("==========================================================")
        kljuc, duzina, tip = BK.kljuc(vektor)
        # PIN WATERMARKING
        mreza, ime_w = PINzigosanje(mreza, kljuc)
        vektor_locked = vektor
        nastavak = ''
    # SMO LOCK
    elif ds_idx == 2:
        print("==========================================================")
        print("       Odabrali ste zakljucavanje bez vodenog ziga.")
        print("==========================================================")
        # ZAKLJUCAVANJE PRETUMBACIJA
        print("----------------------------------------------------------")
        print("   IZABERI OBLAST ZAKLJUCAVANJA")
        print("----------------------------------------------------------")
        opcije = [
            "Sve weight koeficijente",
            "Samo klasifikacioni sloj"]
        for i, n in enumerate(opcije, 1):
            print(f"  {i}) {n}")
        try:
            oblast_idx = int(input(f"Izbor [1-{len(opcije)}]: ").strip())
            assert 1 <= oblast_idx <= len(opcije)
        except Exception:
            print("Nevažeći izbor.")
            return None, None, None
        print("----------------------------------------------------------")
        
        # 1) Zaključavanje svih weight koeficijenata
        if oblast_idx == 1:
            kljuc, duzina, tip = BK.kljuc(vektor)
            vektor_locked = perMutacija(vektor, kljuc)
        # 2) Zaključavanje samo klasifikacionog sloja
        elif oblast_idx == 2:
            # Uzimamo iz list_W  fc - vektor klasifikacionog sloja.
            list_W, list_b, list_s, list_r = mreza    
            vek_fc = list_W[fc_pos]
            shape_fc = vek_fc.shape
            vek_fc = vek_fc.flatten()
            kljuc, duzina, tip = BK.kljuc(vek_fc)
            vek_fc_locked = perMutacija(vek_fc, kljuc)
            # Vracanje 
            assert vek_fc.size == np.prod(shape_fc), "Ne poklapa se broj elemenata za reshape."
            vek_fc_locked = vek_fc_locked.reshape(shape_fc)
            list_W[fc_pos] = vek_fc_locked                
            vektor_locked = np.concatenate([a.ravel(order='C') for a in list_W], axis=0)
        ime_w = ""
        nastavak = '_Locked' + str(duzina)
    # EXIT
    elif ds_idx == 3:
        print(">> Prekid rada.")
        print("==========================================================")
        return None, None, None
    
    # FORMIRANJE NOVE LOCKED MREZE  
    '''vracamo flatenovan vatermarkovan vektor y u oblik liste'''
    parts = np.split(vektor_locked, np.cumsum(sizes)[:-1])
    list_W_restored = [p.reshape(shp) for p, shp in zip(parts, shapes)]
    mreza = zameni_mrezu(mreza, host, list_W_restored)
    
    # Snimamo kljuc
    ime_LW = nastavak + ime_w
    os.makedirs(out_dir, exist_ok=True)
    baza = os.path.splitext(ime)[0]
    out_path = os.path.join(out_dir, f"{baza}{ime_LW}.txt")
    
    # Snimamo kljuc kao txt file
#    np.savetxt("kljuc.txt", kljuc, fmt="%.6f", delimiter="\t") - 6 decimala
    np.savetxt(out_path, kljuc, fmt="%d", delimiter="\t")  # - integer     
        
    return mreza, ime_LW, kljuc
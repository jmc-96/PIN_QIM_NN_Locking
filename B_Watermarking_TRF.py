# =============================================================================
#                                 WATERMARKING
# =============================================================================

import numpy as np

# %% ------------------------------- OPCIJE -----------------------------------

def opcija_w(mreza):
    # Koristimo list_W
    list_W, list_b = mreza
    shapes = [a.shape for a in list_W]
    vektor = np.concatenate([a.ravel(order='C') for a in list_W], axis=0)
    host = "list_W"
    print("==================================================================")
    print("Izabrali ste weights vektor za nosioca vodenog ziga:", len(vektor))
    print("==================================================================")
    return vektor, shapes, host

def opcija_b(mreza):
    list_W, list_b = mreza
    shapes = [a.shape for a in list_b]
    vektor = np.concatenate([a.ravel(order='C') for a in list_b], axis=0)
    host = "list_b"
    print("==================================================================")
    print("Izabrali ste biases vektor za nosioca vodenog ziga:", len(vektor))
    print("==================================================================")
    return vektor, shapes, host

# %% ----------------------------- HOST VEKTOR --------------------------------

def hostVektor(mreza):
    x, x_shapes, host = opcija_b(mreza)
    return x, x_shapes, host

# %% ---------------------------- BINARNI PIN U -------------------------------
    
def binarniPIN(kljuc, x, pin_duzina): 
    import C_Binarni_PIN as CBP
    
    # formiramo PIN od kljuca
    if pin_duzina <= len(kljuc.flatten()):
        pin = kljuc.flatten()[:pin_duzina]
    else:
        raise ValueError("GRESKA! Prevelika duzina PIN-a!")
    max_vrednost = int(max(pin))
    bit_duzina = max(1, max_vrednost.bit_length())
    print("==========================================================")
    print("Za elemente kljuca izabranih za PIN bitska duzina je: ", bit_duzina)
    max_pin = len(x) // bit_duzina
    
    # proveravamo da li duzina kljuca nije veca od duzine host vektora
    if max_pin >= len(pin):
        u = CBP.celiUbinarni(pin)
    else:
        pin = pin[:max_pin]
        u = CBP.celiUbinarni(pin)
        print("==========================================================")
        print("Binarna duzina vaseg PIN-a premasuje duzinu host vektora, te je")
        print("Vas PIN automatski skracen na maksimalnu dozvoljenu vrenost: ",max_pin)
        print("i sada je definisan kao niz: \n", pin)
        print("==========================================================")
    
    return u, pin

# %% ZAMENI VEKTOR U MREZI

def zameni_mrezu(mreza, host, nova_lista):
    mapa = {"list_W": 0, "list_b": 1}
    i = mapa[host]             # indeks u tuple
    mr = list(mreza)           # konvertuj u listu
    mr[i] = nova_lista         # zameni element
    return tuple(mr)           # vrati tuple

# %% ------------------------  WATERMARKING  ----------------------------------

def vodeniZig(mreza, kljuc):
    
    # Formiramo prvo host vektor
    x, x_shapes, host = hostVektor(mreza)
    print("==========================================================")
    pin_duzina = int(input("Unesite zeljenu celobrojnu duzinu PIN-a: "))
    
    # Formiramo PIN i vodeni zig "u" 
    u_bits, PIN = binarniPIN(kljuc, x, pin_duzina)
    print("Vas PIN, duzine ", len(PIN), "je definisan nizom: \n", PIN)
    print("==========================================================")
    
    # ------------------------- MENI ZA UGRADNJU ------------------------------
    print("==========================================================")
    print("                Izbor ugradnje watermarka")
    print("----------------------------------------------------------")
    print("1) QIM")
    print("2) Sparse QIM")
    print("----------------------------------------------------------")
    izbor = input("Izbor [1/2, default 1]: ").strip() or "1"
    print("----------------------------------------------------------")
    
    # ------------- Zajednički Δ (sa podrazumevanom vrednošću) ----------------
    
    delta_in = input("Unesite Δ (npr. 0.1) [Enter za 0.1]: ").strip()
    try:
        delta = float(delta_in) if delta_in else 0.1
    except Exception:
        print("[WARN] Nevažeći unos za Δ, koristim 0.1")
        delta = 0.1
    print("----------------------------------------------------------")
    
    # -------------------- POKRETANJE QIM ili Sparse QIM ----------------------
    
    if izbor == "1":
        # UGRADNJA ZIGA QIM
        import C_QIM
        y = C_QIM.embed_qim(x, u_bits, Δ=delta)
    else:
        # UGRADNJA ZIGA Sparse QIM
        import C_SparseQIM as CSQIM
        y = CSQIM.sQIM_embed(x, u_bits, Δ=delta)

    # --------------------- FORMIRANJE WATERMARKED MREZE ----------------------
    
    list_W, list_b = mreza

    '''vracamo flatenovan vatermarkovan vektor y u oblik liste'''
    sizes = [r * c for (r, c) in x_shapes]
    parts = np.split(y, np.cumsum(sizes)[:-1])
    list_restored = [p.reshape(shp) for p, shp in zip(parts, x_shapes)]

    mreza = zameni_mrezu(mreza, host, list_restored) 

    ime = '_Watermarked_b' + '_PIN_' + str(pin_duzina)

    print("==========================================================")
    print("Vodeni zig je ugradjen")
    print("==========================================================")
    
        
    return mreza, ime
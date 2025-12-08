# =============================================================================
#                               CNN WATERMARKING
# =============================================================================

import numpy as np

# %% OPCIJE

def opcija_w(mreza):
    # Koristimo list_W
    list_W, list_b, list_s, list_r = mreza
    shapes = [a.shape for a in list_W]
    vektor = np.concatenate([a.ravel(order='C') for a in list_W], axis=0)
    host = "list_W"
    print("==================================================================")
    print("Izabrali ste weights vektor za nosioca vodenog ziga:", len(vektor))
    print("==================================================================")
    return vektor, shapes, host

def opcija_b(mreza):
    list_W, list_b, list_s, list_r = mreza
    shapes = [a.shape for a in list_b]
    vektor = np.concatenate([a.ravel(order='C') for a in list_b], axis=0)
    host = "list_b"
    print("==================================================================")
    print("Izabrali ste biases vektor za nosioca vodenog ziga:", len(vektor))
    print("==================================================================")
    return vektor, shapes, host

# %% ---------------------------- HOST VEKTOR ---------------------------------

def hostVektor(mreza):
    menu = {
        1: opcija_b,
        2: opcija_w
    }
    izbor = int(input("Izaberite host vektor za PIN watermark: 1 - Biases ili 2 - Weights: "))
    if izbor in menu:
        x, x_shapes, host = menu[izbor](mreza)
    else:
        print("Pogresan izbor! Dozvoljeni unos je 1 ili 2")
    
    return x, x_shapes, host

# %% BINARNI PIN U
    
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

# %% ------------------------ ZAMENI VEKTOR U MREZI ---------------------------

def zameni_mrezu(mreza, host, nova_lista):
    mapa = {"list_W": 0, "list_b": 1, "list_s": 2, "list_r": 3}
    i = mapa[host]             # indeks u tuple
    mr = list(mreza)           # konvertuj u listu
    mr[i] = nova_lista         # zameni element
    return tuple(mr)           # vrati tuple

# %% WATERMARKING

def vodeniZig(mreza, kljuc):
    
    # Formiramo prvo host vektor
    x, x_shapes, host = hostVektor(mreza)
    print("==========================================================")
    pin_duzina = int(input("Unesite zeljenu celobrojnu duzinu PIN-a: "))
    
    # Formiramo PIN i vodeni zig "u" 
    u, PIN = binarniPIN(kljuc, x, pin_duzina)
    print("Vas PIN, duzine ", len(PIN), "je definisan nizom: \n", PIN)
    print("==========================================================")
    
    # UGRADNJA ZIGA QIM
    import C_QIM
    y = C_QIM.embed_qim(x, u, Δ=0.1)
    
    # FORMIRANJE WATERMARKED MREZE
    list_W, list_b, list_s, list_r = mreza
    
    '''vracamo flatenovan vatermarkovan vektor y u oblik liste'''
    sizes = [r * c for (r, c) in x_shapes]
    parts = np.split(y, np.cumsum(sizes)[:-1])
    list_restored = [p.reshape(shp) for p, shp in zip(parts, x_shapes)]
    
    mreza = zameni_mrezu(mreza, host, list_restored) 
    
    ime = '_Watermarked_b'+ '_PIN_' + str(pin_duzina)
    
    print("==========================================================")
    print("Vodeni zig je ugradjen")
    print("==========================================================")
    
    # UGRADNJA ZIGA Sparse QIM
        
    return mreza, ime
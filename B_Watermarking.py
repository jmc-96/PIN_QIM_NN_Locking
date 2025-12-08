# =============================================================================
#                               NN WATERMARKING
# =============================================================================

import B_Reshape as BR

# %% OPCIJE

def opcija_w(mreza):
    vektor, shapes = BR.flattening_W(mreza)
    print("Izabrali ste biases vektor za nosioca vodenog ziga")
    return vektor, shapes

def opcija_b(mreza):
    vektor, shapes = BR.flattening_B(mreza)
    print("Izabrali ste weights vektor za nosioca vodenog ziga")
    return vektor, shapes

# %% ---------------------------- HOST VEKTOR ---------------------------------

def hostVektor(mreza):
    menu = {
        1: opcija_b,
        2: opcija_w
    }
    izbor = int(input("Izaberite host vektor: 1 - Biases ili 2 - Weights: "))
    if izbor in menu:
        x, x_shapes = menu[izbor](mreza)
    else:
        print("Pogresan izbor! Dozvoljeni unos je 1 ili 2")
    
    return x, x_shapes

# %% --------------------------- BINARNI PIN U --------------------------------
    
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

# %% ------------------------------ WATERMARKING ------------------------------

def vodeniZig(mreza, kljuc):
    
    # Formiramo prvo host vektor
    x, x_shapes = hostVektor(mreza)
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
    biases = BR.unflattening_B(y, x_shapes)
    weights = mreza[0]
    mreza = tuple([weights, biases])
    ime = '_Watermarked_B'+ '_PIN_' + str(pin_duzina)
    
    print("==========================================================")
    print("Vodeni zig je ugradjen")
    print("==========================================================")
    
    # UGRADNJA ZIGA Sparse QIM
        
    return mreza, ime
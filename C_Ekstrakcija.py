# %% EKSTRAKCIJA QIM

import C_QIM

def eKstrakcija():
    y, bit_duzina = kljucIpin()
    y = C_QIM.izdvoji_zigosan(y, bit_duzina)
    u_hat = C_QIM.extract_qim(y, Δ=0.1)
      
    #u_hat = C_QIM.extract_QIM(y[:wm_len])
    
    return u_hat, bit_duzina

# %% DETEKCIJA DECIMALNOG IZ BINARNOG BROJA

def binarniUdecimalni():
    binarni, bit_duzina = eKstrakcija()
    import numpy as np
    binarni = np.array(binarni).astype(int)
    assert len(binarni) % bit_duzina == 0, "Duzina mora biti deljiva sa bitskom duzinom"
    reshaped = binarni.reshape(-1, bit_duzina)
    powers = 2 ** np.arange(bit_duzina - 1, -1, -1)
    decimalni = reshaped.dot(powers)
    
    return decimalni

# %% UCITAVANJE MREZE

def kljucIpin():
    import B_NN_ucitavanje as UNN   
    mreza, ime = UNN.ucitavanje_NN()
    sizes = UNN.readSizes(mreza)
    if ime.endswith(".dat"):
        ime = ime.removesuffix(".dat")
    else:      
        ime = ime
        
    print("==========================================================")
    print('Ucitana je NN: ', ime)
    print('Broj layera ucitane NN: ', len(sizes))
    print('Raspored neurona po layerima za ucitanu NN: \n',sizes)
    print("==========================================================")
    
    # %% UCITAVANJE KLJUCA, PINA, ILI MREZE
    
    import C_Ucitavanje as cu
    print("==========================================================")
    ime = input("Unesite ime fajla za ucitavanje kljuca: ")
    kljuc = cu.uciTavanje (ime)
    
    # %% TRAZENJE BITSKE DUZINE
    # Iz imena vidimo da je PIN duzine 4. Izdvajamo prva 4 clana niza kljuca
    PIN = kljuc.flatten()[:4]
    bit_duzina = max(1, int(max(PIN)).bit_length())
    
    # %% IZDVAJAMO IZ BIASES VEKTORA UPISANI PIN
    import B_Reshape as BR
    vektor, shapes = BR.flattening_B(mreza)
    
    return vektor, bit_duzina


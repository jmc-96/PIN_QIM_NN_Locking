# =============================================================================
# %%                  Konverzija kljuca u binarni niz
# =============================================================================

def celiUbinarni(pin):

    # Konvertuje listu celih brojeva u binarni niz
    if pin.ndim != 1:
        pin = pin.flatten()
    
    # PROVERAVAMO KOLIKI NAM JE POTREBAN PROSTOR ZA BINARNI WATERMARK
    max_vrednost = int(max(pin))
    bit_duzina = max(1, max_vrednost.bit_length())

    import numpy as np
    binarna_matrica = []
    for k in pin:
        binarni_str = bin(k)[2:].zfill(bit_duzina)
        binarni_vektor = [int(bit) for bit in binarni_str]
        binarna_matrica.append(binarni_vektor)
        
    pin_binarni = [bit for row in binarna_matrica for bit in row]
    pin_binarni = np.concatenate([pin_binarni])
      
    return pin_binarni
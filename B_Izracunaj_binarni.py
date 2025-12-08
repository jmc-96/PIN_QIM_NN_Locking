# Definisemo [pretvaranje decimalnih brojeva u binarne]

import numpy as np

# Input na primer cifra 7
Inp =([0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
Inp = np.array(Inp)

# funkcija

def izracunaj_binarni (ulaz):
    # Tegovi
    W = [[0,0,0,0,0,0,0,0,1,1],
         [0,0,0,0,1,1,1,1,0,0],
         [0,0,1,1,0,0,1,1,0,0],
         [0,1,0,1,0,1,0,1,0,1]]
    W = np.array(W)
    # WT = W.transpose() # Transponovana matrica ako zatreba
    # Bias
    B = [0.01, 0.03, 0.03, 0.04]
    B = np.array(B)
    # Izracunavanje
    en = W[:,0].size
    if W[0,:].size == ulaz.size:
        priz = W * ulaz # Mnozimo svaki red ulaznim vektorom
        zbir = np.zeros(en)
        deka = 0
        for j in range(en):
            zbir[j] = sum(priz[j,:]) - B[j]
            deka = deka + zbir[j].round()*2**(en-1-j)
        zbir = np.flip(zbir) # Invertovani redosled zbog neurona
        print('\n','Aktivacije izlaznih ',en,' neurona su:', zbir)
        deka = int(deka)
        print('\n','Binarna vrednost dobijenog broja je:', bin(deka))
    else:
        print('\n','Pozor! Neispravan broj ulaznih aktivacija','\n')
        deka = ['N/A']
    return deka
 
odgovor = izracunaj_binarni (Inp)
print('\n','Procitani broj je:', odgovor)
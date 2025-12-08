# DUPLIKATI

import numpy as np

def nadji_duplikate(kljuc):
    # vrati: dict {vrednost_indeksa: np.ndarray(pozicije_gde_se_pojavljuje)}
    vals, inv, counts = np.unique(kljuc, return_inverse=True, return_counts=True)
    dup_mask = counts > 1
    dupli = {int(vals[i]): np.where(inv == i)[0] for i in np.where(dup_mask)[0]}
    if dupli:
        print("==========================================================")
        print ("UPOZORENJE! Testirani vektor sadrzi ponovljene clanove")
        print("==========================================================")
        for idx, pos in dupli.items():
            print(f"Indeks {idx} se pojavljuje na pozicijama {pos.tolist()}")
    else:
        print("==========================================================")
        print ("Nema duplikata u testiranom vektoru")
        print("==========================================================")
    return dupli
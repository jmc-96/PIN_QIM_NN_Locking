# %% UCITAVANJE

def uciTavanje (ime):
    print("==========================================================")
    folder = input("Unesite ime foldera u kome je kljuc (default: data/key): ").strip()
    if folder == "":
        folder = "data/key"
    print("==========================================================")
    if ime.endswith(".dat"):
        import _pickle
        niz = _pickle.load(open(folder + '//' + ime, 'rb'))
    elif ime.endswith(".npy"):
        import numpy as np
        niz = np.load(folder + '//' + ime)
    elif ime.endswith(".txt"):
        import numpy as np
        niz = np.loadtxt(folder + '//' + ime, dtype=int)
    elif ime.endswith(".xlsx"):
        import pandas as pd
        df = pd.read_excel(folder + '//' + ime, header=None)
        niz = df.to_numpy(dtype=int)
    else:      
        niz = ''
        print("==========================================================")
        print("NEPOZNAT TIP FAJLA!\nProverite da li ste uneli ispravnu ekstenziju?")
        print("==========================================================")
    
    return niz
    


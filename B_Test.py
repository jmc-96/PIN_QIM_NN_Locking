# =============================================================================
#             UCITAVANJE MREZE TESTA I ODREDJIVANJE STRUKTURE
# =============================================================================

import numpy as np

def sigma(z):
    return 1.0/(1.0 + np.exp(-z))

def citaj(mreza, a):
    # Daje izlaz mreze ako je "a" ulaz."
    weights = mreza[0]
    biases = mreza[1]
    for b, w in zip(biases, weights):
        a = sigma(np.dot(w, a) + b)
    return a

def Testiranje_NN():
    # Ucitavanje fajla za ucitavanje trening podataka
    import B_TR_ucitavanje as TU

    # Dobijamo vektore za trening, validaciju i test
    trening, validacija, test = TU.razvrstavanje()
    TEST = [(x, y) for (x,y) in test]
    n_test = len(TEST)
    print("==========================================================")
    print('Broj test uzoraka: ', n_test)
    print("==========================================================")
    # Delimo slike i znacenja
    # test_slike = [x for (x,y) in TEST]
    # test_label = [y for (x,y) in TEST]
    
    # Ucitavamo NN
    import B_NN_ucitavanje as UNN
    mreza, ime = UNN.ucitavanje_NN()
    sizes = UNN.readSizes(mreza)
    if ime.endswith(".dat"):
        ime = ime.removesuffix(".dat")
    else:      
        ime = ime
    print("==========================================================")
    print('Ime ucitane NN: ', ime)
    print('Broj layera ucitane NN: ', len(sizes)-2)
    print('Raspored neurona po layerima za ucitanu NN: ',sizes)
    print("==========================================================")
    
    # Dobijamo rezultate kao B_NNetwork3.dat tuple para vrednosti (rez mreze, labela)
    rezultati = [(np.argmax(citaj(mreza, x)), y) for (x, y) in TEST]
    rezultat = 100*(sum(int(x == y) for (x, y) in rezultati))/n_test
    
    return sizes, mreza, rezultat, ime   

# %%Provera preciznosti

def rezulTat():
    sizes, mreza, rezultat, ime = Testiranje_NN()
    print("==========================================================")
    print('Tacnost mreze ', ime ,' je: ', rezultat, '%') 
    print("==========================================================")   
        
    
    # %% SNIMANJE STRUKTURE (MATRICE VREDNOSTI WEIGHTS i BIASES)
    
    # Snimamo rezultate
    folder = 'test'
    
    import B_Make_txt_file as MTX
    MTX.makeTxtFile(folder, ime, rezultat)
    
    print("==========================================================")
    print("Rezultat testa je snimljen u folderu >", folder, "<")
    print("pod imenom >", ime + '.txt', "< ")
    print("==========================================================")
          
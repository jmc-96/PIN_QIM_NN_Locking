# =============================================================================
#                     RAD SA NN - DOUCAVANJE (FINE-TUNING)
# =============================================================================

# %% ---------------- FORMIRANJE MREZE I UCITAVANJE SETOVA --------------------

def douCavanje_NN():
    # Ucitamo prvo formiranu mrezu
    import B_NN_ucitavanje as UNN   
    mreza, ime = UNN.ucitavanje_NN()
    sizes = UNN.readSizes(mreza)
    if ime.endswith(".dat"):
        ime = ime.removesuffix(".dat")
    else:      
        ime = ime
        
    print("==========================================================")
    print('Ucitana je NN: ', ime)
    print('Raspored neurona po layerima za ucitanu NN: ',sizes)
    print("==========================================================")

    # Ucitavanje fajla za ucitavanje trening podataka
    import B_TR_ucitavanje

    # Dobijamo vektore za trening, validaciju i test
    trening, validacija, test = B_TR_ucitavanje.razvrstavanje()

    # Ucitavanje strukture za ucenje, validaciju i testiranje
    import C_NN_sigma

    # Aktiviranje funkcije na osnovu zadate dimenzije sizes
    neural = C_NN_sigma.NNet(mreza, sizes)

    # Pozivamo SGN(Stochactic Gradient Descend) sa argumentima:
    # trening set, mini_batch_size, broj epoha, eta - brzina ucenja
    # net.SGD(trening, 30, 10, 3.0, test = test)
    # ako ne unesemo vrednost test = test podrazumevace se 0, bez testa
    epoha = int(input('Unesi vrednost za zeljeni broj epoha: '))
    min_b_s = int(input('Unesi vrednost za mini batch size: '))
    eta = float(input('Unesi vrednost za brzinu ucenja eta: '))
    
    return neural, trening, epoha, min_b_s, eta, test, ime

# %% ---------------------------- UCENJE MREZE --------------------------------
neural, trening, epoha, min_b_s, eta, test, ime = douCavanje_NN()
neural.SGD(trening, epoha, min_b_s, eta, test = test)

# %% -------- SNIMANJE STRUKTURE (MATRICE VREDNOSTI WEIGHTS i BIASES) ---------
mreza = tuple([neural.weights, neural.biases])

# Snimamo listu biases i weights
ime = ime + '_1'
import B_Snimanje as BS
BS.sniManje(mreza, ime)
    
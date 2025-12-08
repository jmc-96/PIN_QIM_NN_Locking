# RAD SA NN - LEARNING

# %% FORMIRANJE MREZE I UCITAVANJE SETOVA

def formiranje_NN():
    # Pozivamo funkciju za definisanje broja neurona i layera
    # Formiranje lista 'sizes': Na pr. 3 layera, br. neurona 1.-784, 2.-30 i 3.-10
    # 1.-784 je obavezan zbog dimenzije slika iz seta 28x28
    import B_Form_Sizes as FS
    sizes = FS.formSizes()

    # Ucitavanje fajla za ucitavanje trening podataka
    import B_TR_ucitavanje

    # Dobijamo vektore za trening, validaciju i test
    trening, validacija, test = B_TR_ucitavanje.razvrstavanje()

    # Ucitavanje strukture za ucenje, validaciju i testiranje
    import B_NN_sigma

    # Aktiviranje funkcije na osnovu zadate dimenzije sizes
    neural = B_NN_sigma.NNet(sizes)

    # Pozivamo SGN(Stochactic Gradient Descend) sa argumentima:
    # trening set, mini_batch_size, broj epoha, eta - brzina ucenja
    # net.SGD(trening, 30, 10, 3.0, test = test)
    # ako ne unesemo vrednost test = test podrazumevace se 0, bez testa
    epoha = int(input('Unesi vrednost za zeljeni broj epoha: '))
    min_b_s = int(input('Unesi vrednost za mini batch size: '))
    eta = float(input('Unesi vrednost za brzinu ucenja eta: '))
    
    return neural, trening, epoha, min_b_s, eta, test

# %% UCENJE MREZE
neural, trening, epoha, min_b_s, eta, test = formiranje_NN()
neural.SGD(trening, epoha, min_b_s, eta, test = test)

# %% SNIMANJE STRUKTURE (MATRICE VREDNOSTI WEIGHTS i BIASES)
mreza = tuple([neural.weights, neural.biases])

# Snimamo listu biases i weights
ime = str(input('Unesi ime strukture neuronske mreze: '))

import B_Snimanje as BS
BS.sniManje(mreza, ime)
    
# %%
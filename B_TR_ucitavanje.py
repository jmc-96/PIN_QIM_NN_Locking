# =============================================================================
#           UCITAVANJE PODATAKA ZA TRENING, VALIDACIJU I TEST
# =============================================================================

import _pickle
# C optimizovana verzija pickle (cPickle za Python2)
import gzip
import numpy as np

def ucitavanje():
    # Ucitava podatke iz mnist.pkl.gz
    
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    trening, validacija, test = _pickle.load(f, encoding = "latin1")
    f.close()
    return (trening, validacija, test)
    
    #   Svaki od tri seta je dat kao tuple sa dva clana:
    # - matrice vektora slika dimenzije (50000x784), (10000x784) i (10000x784) 
    # - vektora integer cifara (0-9) (50000x1), (10000x1) i (10000x1)

def razvrstavanje():
    # Prerasporedjuje ucitane setove (trening, validacija, test)
    
    tr_d, va_d, te_d = ucitavanje()
    
    # Trening set numpy.ndarry ([50000, 784, 10],[50000, 10])
    trening_ulaz = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    trening_rezultat = [vektorizovani_rezultat(y) for y in tr_d[1]]
    trening = zip(trening_ulaz, trening_rezultat)
    
    # Set validacija numpy.ndarry ([10000, 784, 10],[50000, 10])
    validacija_ulaz = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validacija = zip(validacija_ulaz, va_d[1])
    
    # Test set numpy.ndarry ([10000, 784, 10],[50000, 10])
    test_ulaz = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test = zip(test_ulaz, te_d[1])
    
    return (trening, validacija, test)
    # Svi izlazni setovi su zip() zadati skupovi

def vektorizovani_rezultat(j):
    # Daje vektor nula dimenzije (10, 1) sa jedinicom na mestu koje je zadato    
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


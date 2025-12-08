# RESHAPE MREZE

import numpy as np

# Spajanje matrica tezina u vektor
def flattening_W(mreza):
    # Prvo zabelezimo oblik matrice tezina
    shapes = [w.shape for w in mreza[0]]
    # list k [A1, A2, ..., Ak] dimenzija (n1 x m1), (n2 x m2), ... (nk x mk)
    # Pretvaramo u vektor dimenzija 1 x (n1 x m1 + n2 x m2 + ... nk x mk)
    vektor = np.concatenate([w.flatten() for w in mreza[0]])
    return(vektor, shapes)

# Spajanje matrica biases u vektor
def flattening_B(mreza):
    # Prvo zabelezimo oblik matrice biases
    shapes = [b.shape for b in mreza[1]]
    vektor = np.concatenate([b.flatten() for b in mreza[1]])
    return(vektor,shapes)

# Od jednodimenzionalnog vektora pravimo novi weights
def unflattening_W(vektor, shapes):
    weights = []
    idx = 0
    for shape in shapes:
        size = shape[0] * shape[1]
        part = vektor[idx : idx + size].reshape(shape)
        weights.append(part)
        idx += size
    return weights

def unflattening_B(vektor, shapes):
    biases = []
    idx = 0
    for shape in shapes:
        size = shape[0] * shape[1]
        part = vektor[idx : idx + size].reshape(shape)
        biases.append(part)
        idx += size
    return biases

def formiranjeMreze(weights, biases):
    mreza = []
    mreza[0] = weights
    mreza[1] = biases
    return mreza

def proVera(mreza, vektor, shapes):
    # Ovde treba da napravi if ako je shapes B ili B
    # Ili da dupliram funkciju provere za oba vektora
    if shapes[0][1] != 1:
        obnovljeni_w = unflattening_W(vektor, shapes)
        # Verifikacija
        for original, recovered in zip(mreza[0], obnovljeni_w):
            assert np.allclose(original, recovered), "Mreze se ne podudaraju!"
        print("==========================================================")
        print(" Uporedjivanje matrica posle reshapinga")
        print(" Originalna i obnovljena matrica se podudaraju.")
        print("==========================================================")
    else:     
        obnovljeni_b = unflattening_B(vektor, shapes)
        # Verifikacija
        for original, recovered in zip(mreza[1], obnovljeni_b):
            assert np.allclose(original, recovered), "Mreze se ne podudaraju!"
        print("==========================================================")
        print(" Uporedjivanje matrica posle reshapinga")
        print(" Originalna i obnovljena matrica se podudaraju.")
        print("==========================================================")
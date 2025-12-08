# RAD SA NN - PREPOZNAVANJE

# Load snimljenih podataka

import numpy as np
import matplotlib.image as MI
import matplotlib.pyplot as plt

import B_NN_ucitavanje as UNN
mreza, ime = UNN.ucitavanje_NN()

def bez_alfa(slika):
    
    # Uklanjanjamo 'alpha' kanal iz png slike ako postoji
    en_0 = np.shape(slika)[0]
    en_1 = np.shape(slika)[1]
    en = en_0 * en_1

    if np.shape(slika)[2]>3:
        sli_s = np.reshape(slika, (en, np.shape(slika)[2]))
        sli_s = sli_s.transpose()
        sli_s = sli_s[:3]
        # Optimizujemo sliku da bela ima vrednost 0
        sli_s = sli_s - min(min(sli_s[0]),min(sli_s[1]),min(sli_s[2]))
        sli_s = sli_s.transpose()
        sli = np.reshape(sli_s, (en_0, en_1, 3))

    return sli

# %% Prepoznavanje

def rgb2gray(rgb_img):
    
    en_0 = np.shape(rgb_img)[0]
    en_1 = np.shape(rgb_img)[1]
    en = en_0 * en_1
    lsh = len(np.shape(rgb_img))
    
    if lsh > 2:
        if np.shape(rgb_img)[lsh-1] > 3:
            print('Ucitana slika ima vise od tri kolor kanala')
            rgb_img = bez_alfa(rgb_img)
            rgb_img = np.reshape(rgb_img, (en, 3))
            rgb_img = rgb_img.transpose()
            R = rgb_img[0]
            G = rgb_img[1]
            B = rgb_img[2]
            MaX = max(max(R), max(G), max(B))
        
            # ITU-R 601-2 luma transform to B/W
            Gray = (MaX - (0.299 * R + 0.587 * G + 0.114 * B))/MaX
            Gray = 1 - np.reshape(Gray, (en, 1))
            Gray = Gray - min(Gray[0])
            # Povecavamo kontrast
            Gray = (1/max(Gray.transpose()[0]))*Gray
            plt.imshow(np.reshape(Gray, (en_0, en_1)))
            
        else:
            rgb_img = np.reshape(rgb_img, (en, 3))
            rgb_img = rgb_img.transpose()
            R = rgb_img[0]
            G = rgb_img[1]
            B = rgb_img[2]
            MaX = max(max(R), max(G), max(B))
        
            # ITU-R 601-2 luma transform to B/W
            Gray = (MaX - (0.299 * R + 0.587 * G + 0.114 * B))/MaX
            Gray = np.reshape(Gray, (en, 1))
            Gray = (1/max(Gray.transpose()[0]))*Gray     
            
            plt.imshow(np.reshape(Gray, (en_0, en_1)))
    else:
        MX = max([max(rgb_img[0]) - min(rgb_img[0]),
                  max(rgb_img[1]) - min(rgb_img[1])])
        Gray = (MX - rgb_img)/MX
        Gray = np.reshape(Gray, (en, 1))
    
    return Gray

def sigma(z):
    return 1.0/(1.0 + np.exp(-z))

def citaj(mreza, a):
    # Daje izlaz mreze ako je "a" ulaz."
    weights = mreza[0]
    biases = mreza[1]
    
    for b, w in zip(biases, weights):
        a = sigma(np.dot(w, a) + b)
    return a
# %%

# Otvara kao matricu vrednosti
ulaz = str(input('Unesi ime slike za ocitavanje :'))
slika = MI.imread('data//'+ ulaz)
plt.imshow(slika)

rezultat = np.argmax(citaj(mreza, rgb2gray(slika)))
print(rezultat)
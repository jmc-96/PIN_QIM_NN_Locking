#==============================================================================
#                           FORMIRANJE KLJUCA
#==============================================================================

import numpy as np

# Formiranje kljuca random unosom cifara

def randKljuc(duzina: int, vektor):
    arr = np.asarray(vektor)
    if arr.ndim != 1:
        raise ValueError("Očekivan je jednodimenzionalni (1D) vektor.")
    N = arr.size
    if duzina < 1:
        raise ValueError("duzina mora biti >= 1.")
    if 2 * duzina > N:
        raise ValueError(f"Za dati vektor dužine {N}, maksimalno duzina je {N // 2}.")

    rng = np.random.default_rng()
    perm = rng.permutation(N)[: 2 * duzina]   # bez ponavljanja
    kljuc = perm.reshape(2, duzina)           # parovi kao kolone (2, duzina)
    return kljuc

# Formiranje kljuca izborom najvecih i najmanjih vrednosti tezina
def maxiKljuc(duzina: int, vektor):
    # Kreiramo vektor indexa po opadajucim vrednostima weights koeficijenata
    n = vektor.size
    assert duzina*2 <= n-1, "Tražimo duzina<n/2 da bi top i bottom bili disjunktni."
    
    # Jedna stabilna permutacija po RASTUCEM poretku:
    order = np.argsort(vektor, kind='stable')
    mali = order[:duzina]            # donjih k (rastuće)
    
    # Da bismo ipak zastitili taj vektor radimo randim pretumbaciju malih
    np.random.shuffle(mali)
    veliki = order[-duzina:][::-1]   # gornjih k (opadajuće)
    kljuc = np.vstack([veliki, mali])  
   
    # Testiramo da li postoje ponavljanja
    import C_Duplicates as CD
    CD.nadji_duplikate(kljuc.ravel())
    return kljuc


# %% MENI FORMIRANJE KLJUCA

def kljuc(vektor):
    print("==========================================================")
    duzina = int(input('Unesi zeljenu duzinu kljuca: '))
    if duzina <= len(vektor)/2:
        print("Duzina kljuca je: ", duzina)
        print("==========================================================")
    else:
        duzina = len(vektor)
        print("UPOZORENJE!")
        print("Duzina kljuca premasuje broj clanova vektora i redukovana")
        print("je na duzinu vektora: " + str(len(vektor)))       
        print("Duzina kljuca je: ", duzina)
        print("==========================================================")

    # Biramo izmedju adaptivnog i random unosa
    menu = {
        1: maxiKljuc,
        2: randKljuc
    }
    print("==========================================================")
    print("Izaberite nacin formiranja kljuca.")
    izbor = int(input("1 - Adaptivni, 2 - Random unos : "))
    if izbor in menu:
        if izbor == 1:
            kljuc = menu[izbor](duzina, vektor)
            tip = 'A'
        elif izbor == 2:
            kljuc = menu[izbor](duzina, vektor)
            tip = 'R'
    else:
        print("==========================================================")
        print("Pogresan izbor! Dozvoljeni unos je 1 ili 2")
        print("==========================================================")
    print("==========================================================")

    return kljuc, duzina, tip
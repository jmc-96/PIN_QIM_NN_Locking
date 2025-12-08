# Formiranje vektora broja neurona u layerima

def formSizes():
    n_lay = int(input("Unesi ukupan broj layera, ukljucujuci ulazni i izlazni : "))
    print("Ukupan broj layera je: ", n_lay)
    print("==========================================================")

    sizes = [0 for i in range(n_lay)]
    for i in range(n_lay):
        sizes[i] = int(input("Unesi broj neurona u "+str(i+1)+". layeru : "))
    
    print("==========================================================")
    print("Formirani vektor broja neurona po layerima je: ", sizes)
    return(sizes)
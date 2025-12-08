# %% ZAKLJUCAVANJE PRETUMBACIJOM

# %% OPCIJE

def opcija_w(mreza):
    print("Izabrali ste da radite sa weigts matricom.")
    import B_Reshape as BR
    vektor, shapes = BR.flattening_W(mreza)
    BR.proVera(mreza, vektor, shapes)
    return vektor, shapes

def opcija_b(mreza):
    print("Izabrali ste da radite sa biases vektorom")
    import B_Reshape as BR
    vektor, shapes = BR.flattening_B(mreza)
    BR.proVera(mreza, vektor, shapes)
    return vektor, shapes

# %% PRETUMBACIJA

def perMutacija(vektor, kljuc):
    for i in range(kljuc.shape[1]):
        v = vektor
        a, b = kljuc[0, i], kljuc[1, i]
        old_a = v[a]
        old_b = v[b]
        v[a] = old_b
        v[b] = old_a
    return v 
    
# %% PIN WATERMARKING

def PINzigosanje(mreza, kljuc):
    print("==========================================================")
    print("Da li zelite PIN watermerking?")
    pin_izbor = int(input("Izaberi opciju: 1 - Da ili 0 - Ne: "))
    print("==========================================================")
    if pin_izbor == 1:
        import B_Watermarking as BW
        mreza, ime_w = BW.vodeniZig(mreza, kljuc)
    else:
        ime_w = ""
        print("==========================================================")
        print("Odabrali ste da ne zelite PIN watermarking mreze")
        print("==========================================================")
        
    return mreza, ime_w


# %% ZAKLJUCAVANJE

def zaKljucavanje():
    # UCITAVANJE MREZE
    import B_NN_ucitavanje as UNN   
    mreza, ime = UNN.ucitavanje_NN()
    sizes = UNN.readSizes(mreza)
    if ime.endswith(".dat"):
        ime = ime.removesuffix(".dat")
    else:      
        ime = ime
        
    print("==========================================================")
    print('Ucitana je NN: ', ime)
    print('Broj layera ucitane NN: ', len(sizes))
    print('Raspored neurona po layerima za ucitanu NN: \n',sizes)
    print("==========================================================")

    # BIRAMO IZMEDJU W i B
    menu = {
        1: opcija_w,
        2: opcija_b
    }
    print('Izaberite koje koeficijente ucitane NN zelite da zakljucate.')
    izbor = int(input("Opcije: 1 - Weights  2 - Biases: "))
    if izbor in menu:
        vektor, shapes = menu[izbor](mreza)
    else:
        print("Pogresan izbor! Dozvoljeni unos je 1 ili 2")

    # FORMIRANJE KLJUCA
    import B_Kljuc as BK
    kljuc, duzina, tip = BK.kljuc(vektor)

    # PIN WATERMARKING
    mreza, ime_w = PINzigosanje(mreza, kljuc)    

    # PRETUMBACIJA
    print("==========================================================")
    print("  Da li mozda zelite da odustanete od zakljucavanja? ")
    izbor_p = int(input("i ugradite samo vodeni zig? 1 - Ne  0 - Da : "))
    if izbor_p == 1:
        locked_v = perMutacija(vektor, kljuc)
        nastavak = '_locked_W' + tip + '_' + str(duzina)
        print("==========================================================")
    else:
        locked_v = vektor
        nastavak = ''
        print("==========================================================")
        print("Odabrali ste samo da ugradite vodeni zig.")
        print("==========================================================")
        
    # FORMIRANJE NOVE LOCKED MREZE
    if izbor == 1:
        import B_Reshape as BR
        weights = BR.unflattening_W(locked_v, shapes)
        biases = mreza[1]
        mreza = tuple([weights, biases])
        ime = ime + nastavak + ime_w        
    else:
        import B_Reshape as BR
        biases = BR.unflattening_B(locked_v, shapes)
        weights = mreza[0]
        mreza = tuple([weights, biases])
        ime = ime + nastavak + ime_w
        
    return mreza, ime, kljuc

# %% Snimanje fajla
    
def save_locked():
    mreza, ime, kljuc = zaKljucavanje() 
    
    print("==========================================================")
    print('Ime zakljucane NN: ', ime)
    print("==========================================================")
    
    import B_Snimanje as BS
    BS.sniManje(mreza, ime)
    # BS.snimanjeNpy(kljuc, ime + '_kljuc') # Snimanje kljuca kao npy
    BS.snimanjeTxt(kljuc, ime + '_kljuc') # Snimanje kljuca kao txt
    BS.snimanjeXcel(kljuc, ime + '_kljuc') # Snimanje kljuca kao excel file
    # BS.sniManje(kljuc, ime = ime + '_kljuc') # Snimanje kljuca kao dat
    
    print("==========================================================")
    print("NN mreza je snimljena u folderu >data<")
    print("pod imenom >", ime + '.dat', "< ")
    print("==========================================================")
# Iscitavanje trenutne lokacije foldera i kreiranje foldera ako ne postoji

def makeTxtFile(folder, ime, rezultat):
    import os
    
    lokacija = os.getcwd()
    fold_list= os.listdir()
    folder = str(folder)
    ime = str(ime)
    rezultat = str(rezultat)
    
    if folder in fold_list:
        folder = lokacija + '\\' + folder + '\\'
        with open(folder +ime+'.txt','w', encoding = 'utf-8') as f:
            f.write(rezultat)
    else:
        os.mkdir(folder)
        folder = lokacija + '\\' + folder + '\\'
        #Open file. Moze samo open, ali with obezbedjuje close posle zavrsetka rada
        with open(folder +ime+'.txt','w', encoding = 'utf-8') as f:
            f.write(rezultat)
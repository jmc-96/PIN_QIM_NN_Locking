# SNIMANJE MREZE

def sniManje(mreza, ime):
    import _pickle
    import os
    folder = 'data'
    # Proveravamo da li postoji folder
    if folder in os.listdir():
        _pickle.dump(mreza, open(folder +'//'+ ime + '.dat', 'wb'))
    else:
        os.mkdir(folder)
        _pickle.dump(mreza, open(folder +'//'+ ime + '.dat', 'wb'))
               
def snimanjeXcel(kljuc, ime):
    import pandas as pd
    import os
    folder = 'data'
    # Proveravamo da li postoji folder
    if folder in os.listdir():
        df = pd.DataFrame(kljuc)
        df.to_excel(folder + '//' + ime + '.xlsx', index=False, header=False)
    else:
        os.mkdir(folder)
        df = pd.DataFrame(kljuc)
        df.to_excel(folder + '//' + ime + '.xlsx', index=False, header=False)
        
def snimanjeTxt(pin, ime):
    import numpy as np
    import os
    folder = 'data'
    # Proveravamo da li postoji folder
    if folder in os.listdir():
        np.savetxt(folder + '//' + ime + '.txt', pin, fmt='%d')
    else:
        os.mkdir(folder)       
        np.savetxt(folder + '//' + ime + '.txt', pin, fmt='%d')
        
def snimanjeNpy(kljuc, ime):
    import numpy as np
    import os
    folder = 'data'
    # Proveravamo da li postoji folder
    if folder in os.listdir():
        np.save(folder + '//' + ime + '.npy', kljuc)
    else:
        os.mkdir(folder)
        np.save(folder + '//' + ime + '.npy', kljuc)
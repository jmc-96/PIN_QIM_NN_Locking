# %% Ucitavanje NN mreze

def ucitavanje_NN():
    import _pickle
    print("==========================================================")
    ime = str(input('Unesi ime Neuronske mreze: '))
    print("==========================================================")
    if ime.endswith(".dat"):
        sd = _pickle.load(open('data//'+ime,'rb'))
    else:      
        sd = _pickle.load(open('data//'+ime+'.dat','rb'))
    return(sd, ime)

# Citamo konstrukciju mreze
def readSizes(sd):
    # sizes je vektor broja neurona za svaki layer, ukljucujuci 1 i poslednji
    n_lay = len(sd[0])
    for w in sd[0]:
        sizes = [len(lst[2]) for lst in sd[0]]
        sizes = sizes + [len(sd[0][n_lay-1])] 
    return sizes

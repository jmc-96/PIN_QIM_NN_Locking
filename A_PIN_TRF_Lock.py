#%% ---------------------- Učitavanje Transformera ----------------------------

# Ovo pokrecemo direktno iz konzole
%run -i B_TRF_ucitavanje.py

#%% ------------------------ Torch → Numpy matrix -----------------------------

import B_M_mreza_M as MmM
# Interaktivno — otvoriće se meni u konzoli, ENTER = ALL
mreza, meta = MmM.pravi_mrezu(Model, interactive=True)

# Proveravamo shape-ove (bez embed_index jer imamo meta)
rep = MmM.proveri_mrezu_vs_model(mreza, Model, meta=meta)
print(rep["summary"])


#%% ------------------- ZAKLJUCAVANJE i PIN WATERMARK -------------------------

import B_Locking_TRF as BlT
# mreza_LW, ime_LW, kljuc = BlT.zaKljucavanje(mreza, ime)
# Ako zelimo odredjeni sloj pozicije em_pos - pozicija embedingsa
mreza_WL, ime_WL, kljuc = BlT.zaKljucavanje(mreza, ime)

#%% ------------------------ PROVERA LOCKINGA ---------------------------------

#import J_Test_permutacija as jtp
#em_pos = 0
#ok = jtp.proveri_permutaciju_redova(mreza_WL[0][em_pos], mreza[0][em_pos], kljuc, atol=1e-8)

#%% ---------------------- lockM → nazad → u model ----------------------------

# Vracamo mrezu u model (automatski po imenu)
res = MmM.mreza_2_model(mreza_WL, Model, meta=meta)
print(res)

#%% ------------------ Striktna provera i Snimanje modela ---------------------

from C_Snimi_TRF import snimi_u_WL

# Snimamo model u models/<IME>/WL/model.safetensors (+ kopiramo aux fajlove)
wl_path = snimi_u_WL(Model, base_dir=f"models/{ime}", copy_aux=True)
print("WL folder:", wl_path)

#%% -------------------------- Test Transformer -------------------------------

%run -i A_TRF_test.py
    
#==============================================================================


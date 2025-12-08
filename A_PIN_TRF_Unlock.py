# =============================================================================
#%                        OTKLJUCAVANJE TRANSFORMERA
# =============================================================================

#%% ---------------------- Učitavanje Transformera ----------------------------

# Ovo pokrecemo direktno iz konzole
%run -i B_TRF_ucitavanje.py # models/TinyBERT/WL

#%% ------------------------ Torch → Numpy matrix -----------------------------

import B_M_mreza_M as MmM
# Interaktivno — otvoriće se meni u konzoli, ENTER = ALL
mreza, meta = MmM.pravi_mrezu(Model, interactive=True)

# Proveravamo shape-ove (bez embed_index jer imamo meta)
rep = MmM.proveri_mrezu_vs_model(mreza, Model, meta=meta)
print(rep["summary"])

#%% --------------------------- OTKLJUCAVANJE ---------------------------------

import B_Unlocking_TRF as BuT
# Ako zelimo odredjeni sloj pozicije em_pos - pozicija embedingsa
mreza_unlocked, kljuc, nastavak = BuT.otKljucavanje(mreza, em_pos = 0)

#%% ------------------------- PROVERA UNLOKINGA -------------------------------

import J_Test_permutacija as jtp
em_pos = 0
ok = jtp.proveri_permutaciju_redova(mreza_unlocked[0][em_pos], mreza[0][em_pos], kljuc, atol=1e-8)

#%% ----------------------- DETEKCIJA WATERMARKA ------------------------------

#             Odredjivanje bitske duzine iz naziva kljuca 
     ''' ovo radi samo ako je ispravan i odgovarajuci fajl kljuca'''

from C_Bit_duzina import bit_duzina_iz_kljuca
bit_duzina, pin_duzina, oznaka = bit_duzina_iz_kljuca()

# ----------------------------EKSTRAKCIJA PIN ---------------------------------
'''Proveri ispisani rezultat! Ako je duzina PIN-a pogresna, bice pogresan i PIN'''
import C_PIN_extrakcija_TRF as CE
PINeX = CE.binarniUdecimalni(mreza_unlocked, oznaka, bit_duzina)

#%% ---------------------- UnlockM → nazad → u model --------------------------

# Vracamo otkljucanu mrezu u model (automatski po imenu)
res = MmM.mreza_2_model(mreza_unlocked, Model, meta=meta)
print(res)

#%% ------------------ Striktna provera i Snimanje modela ---------------------

from C_Snimi_TRF import snimi_u_WL

# Snimamo u: models/<IME>/<…>WL/model.safetensors (+ kopiramo aux fajlove)
wl_path = snimi_u_WL(Model, base_dir=f"models/{ime}", copy_aux=True)
print("Folder:", wl_path)
    
#==============================================================================

#%% ----------------- FINE-TUNING i MERENJA PIN ROBUSTNESS --------------------

%run C_TRF_START_test.py

#==============================================================================
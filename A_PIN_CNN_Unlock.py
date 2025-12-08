#%% --------------------------- Učitavanje CNN --------------------------------

import B_CNN_ucitavanje as UNN   
sd, ime, model = UNN.ucitavanje_modela()

#%% --------------- Tenzor .pth → tuple (Numpy matrix) ------------------------

import B_sd_mreza_sd as sMs
mreza, fc_pos = sMs.sd_u_mrezu(sd)

#%% ------------------------- UCITAVANJE KLJUCA -------------------------------

import B_Unlocking_CNN as BU
kljuc, model = BU.ucitaj_kljuc(ime)

#%% --------- UNLOCKING: lockedM Numpy  → Tuple(Unlocked matrix) --------------

mreza_unlocked = BU.meni_otkljucavanje(mreza, kljuc, fc_pos)

#%% ------------------ Unlocked Tenzor → nazad → u model ----------------------

sd_re = sMs.mreza_u_sd(mreza_unlocked, sd)
ime_UL = "_Unlocked"

#%% ------------------ Striktna provera i Snimanje modela ---------------------

import C_Snimi_model as CSM
out_path = CSM.snimi_sd_u_pth(sd_re, model, ime_UL)

#%% --------------------------- Test CNN --------------------------------------

import sys
import B_Test_CNN as test
# Ponavljamo sledece za sve mreze koje zelimo da testiramo
sys.argv = ["B_Test_CNN.py"]
test.main()

#%% --------------------- QIM watermarking ekstrakcija ------------------------
'''ovo moram da resavam ne vidi pth file'''
import C_Ekstrakcija as CE
PINeX = CE.binarniUdecimalni()

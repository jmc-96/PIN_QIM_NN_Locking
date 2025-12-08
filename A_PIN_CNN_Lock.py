#%% ---------------- Download ResNet18, CIFAR10, CIFAR100 ---------------------

'''Ovo pokrecemo samo ako treba da downloadujemo neki model'''
import sys
import C_Download_CNN as mod
sys.argv = ["C_Download_CNN.py"]
mod.main()

'''Ovo pokrecemo samo ako treba da downloadujemo test'''
import C_Download_tests as DT
DT.download_imagenette_val_160("datasets")

#%% ---------------------------- Učitavanje CNN -------------------------------

import B_CNN_ucitavanje as UNN   
sd, ime, model = UNN.ucitavanje_modela()

#%% ------------------- Tenzor .pth → Numpy tuple (matrix) --------------------

import B_sd_mreza_sd as sMs
mreza, fc_pos = sMs.sd_u_mrezu(sd)

#%% ----------- tuple(matrix) → (lock izmene)  →  lockedM ---------------------

import B_Locking_CNN as BlC
mreza_LW, ime_LW, kljuc = BlC.zaKljucavanje(mreza, ime, fc_pos)

#%% ---------------------- lockM → nazad → u model ----------------------------

sd_re = sMs.mreza_u_sd(mreza_LW, sd)

#%% ------------------ Striktna provera i Snimanje modela ---------------------

import C_Snimi_model as CSM
out_path = CSM.snimi_sd_u_pth(sd_re, ime, ime_LW)

#%% ------------------------------ Test CNN -----------------------------------

import sys
import B_Test_CNN as test
# Ponavljamo sledece za sve mreze koje zelimo da testiramo
sys.argv = ["B_Test_CNN.py"]
test.main()

# =============================================================================
# PROBA

''' ovo nam sluzi da izracunamo tacnost za promenu biases'''
# Urade se prvi i drugi korak da se dobije mreza
#reza_LW = process_mreza_with_menu(mreza)

#d_re = sMs.mreza_u_sd(mreza_LW, sd)

#mport C_Snimi_model as CSM
#ut_path = CSM.snimi_sd_u_pth(sd_re, ime, ime_LW = "_10")
# posle ide na test


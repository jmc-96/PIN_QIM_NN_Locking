# ZAKLJUCAVANJE I TEST

# %% Ucitavanje mreze i snimanje zakljucane i/ili watermarkovane

import B_Locking as BL
BL.save_locked()

# %% Pokrecemo test NN - biramo originalnu, watermarkovanu ili zakljucanu

import B_Test
# Ponavljamo sledece za sve mreze koje zelimo da testiramo
B_Test.rezulTat()

# %% QIM watermarking ekstrakcija

import C_Ekstrakcija as CE
PINeX = CE.binarniUdecimalni()


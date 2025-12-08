# =============================================================================
#                           PROCENA ROBUSNOSTI
# =============================================================================

import numpy as np
import D_ProcDelta_QIM_SQIM as DP

'''podrazumeva da imam ucitano mreza i u_bits - binarni PIN'''

#%% 1) Bez napada: Δ po distorzionom budžetu nad x

info_no_attack = DP.proceni_delta(
    mreza, u_bits, oznaka="b", method="sqim", seed=42, ber_target=1e-3
)
delta = info_no_attack["delta"]
T_eff = info_no_attack["T"]

#%% 2) Sa napadom: proceni σ_proj baš nad x (isti seed, isti T)

def attack_fn(v):
    rng = np.random.default_rng(123)
    v2 = np.asarray(v, dtype=np.float16).astype(np.float32)
    v2 = v2 + 0.001 * rng.standard_normal(v2.size) - 1e-5 * v2
    return v2

info_attack = DP.proceni_delta(
    mreza, u_bits, oznaka="b", method="sqim",
    seed=42, ber_target=1e-3, attack_fn=attack_fn
)
delta_attack = info_attack["delta"]
T_eff_attack = info_attack["T"]

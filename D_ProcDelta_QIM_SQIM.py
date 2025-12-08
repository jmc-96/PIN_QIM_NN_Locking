# =============================================================================
#                 STATISTICKI IZBOR Δ ZA QIM I SPARSE QIM
# =============================================================================
#
# - Robusnost:  BER ≈ Q( Δ / (4 σ_proj) )  ⇒  Δ_BER = 4 σ_proj · Q^{-1}(BER_target)
# - Distorzija:
#   * QIM (gusto):         E[MSE] ≈ Δ^2 / 12           ⇒ Δ_max = sqrt(12 D)
#   * Sparse QIM (blok T): E[MSE] ≈ ρ · (Δ^2 / (12 T)) ⇒ Δ_max = sqrt(12 D T / ρ)
# - Biramo Δ = min(Δ_max, max(Δ_BER, Δ_min))
# - Ako Δ_BER > Δ_max, ciljna robusnost nije ostvariva uz dati D; treba menjati L/T/ρ ili budžet.

from __future__ import annotations
import numpy as np
from math import sqrt
from typing import Callable, Optional, Dict, Any

# -------------------------- Q^{-1} aproksimacija -----------------------------

# Tablica za linearnu interpolaciju inverzne Q-funkcije (dovoljno dobra za p∈[1e-6,1e-1])
_QINV_TABLE = {
    1e-1: 1.2815515655446004,
    5e-2: 1.6448536269514722,
    2e-2: 2.053748910631823,
    1e-2: 2.3263478740408408,
    5e-3: 2.5758293035489004,
    2e-3: 2.878161739095082,
    1e-3: 3.0902323061678132,
    5e-4: 3.2905267314919255,
    1e-4: 3.719016485455709,
    1e-5: 4.264890793922825,
    1e-6: 4.753424308822899,
}

def qinv(p: float) -> float:
    """Gruba, ali praktična aproksimacija inverzne Q-funkcije. Važi za p ∈ [1e-6, 1e-1]."""
    p = float(p)
    if p <= 0.0:
        return float("inf")
    if p >= 1e-1:
        return _QINV_TABLE[1e-1]
    if p in _QINV_TABLE:
        return _QINV_TABLE[p]
    xs = np.array(sorted(_QINV_TABLE.keys()))
    ys = np.array([_QINV_TABLE[k] for k in xs])
    lx = np.log10(xs)
    t = np.log10(p)
    if t <= lx.min():
        return float(ys[0])
    if t >= lx.max():
        return float(ys[-1])
    i = int(np.searchsorted(lx, t)) - 1
    w = (t - lx[i]) / (lx[i + 1] - lx[i])
    return float(ys[i] * (1.0 - w) + ys[i + 1] * w)

# ------------------------------ formule za Δ ---------------------------------

def delta_from_distortion_qim(D_per_sample: float) -> float:
    """Maksimalni Δ iz distorzione granice (gusto QIM)."""
    return sqrt(12.0 * float(D_per_sample))

def delta_from_distortion_sqim(D_per_sample: float, T: int, rho: float) -> float:
    """Maksimalni Δ iz budžeta za Sparse QIM (blok T, udeo promenjenih uzoraka ρ)."""
    return sqrt(12.0 * float(D_per_sample) * float(T) / max(float(rho), 1e-12))

def delta_from_target_ber(sigma_proj: float, ber_target: float = 1e-3) -> float:
    """Minimalni Δ potreban za ciljni BER (preko σ projekcije)."""
    return 4.0 * float(sigma_proj) * qinv(float(ber_target))

#%% ----------------------- procena σ_proj (empirijski) -----------------------

def estimate_sigma_proj(x: np.ndarray,
                        T: int,
                        seed: Optional[int],
                        attack_fn: Callable[[np.ndarray], np.ndarray],
                        samples: int = 256) -> float:
    """
    Empirijska procena σ_proj: primeni 'attack_fn' na x, zatim meri projekcione
    razlike po blokovima sa istim RNG-om kao u Sparse QIM (jedinični vektor α).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float).ravel()
    N = len(x)
    L = min(samples, max(1, N // max(1, int(T))))
    X = x[:L * T].reshape(L, T)
    Y = np.asarray(attack_fn(x), float).ravel()[:L * T].reshape(L, T)
    diffs = []
    eps = 1e-12
    for bX, bY in zip(X, Y):
        alpha = rng.standard_normal(T)
        nrm = np.linalg.norm(alpha)
        if nrm < eps:
            alpha[0] = 1.0
            nrm = 1.0
        alpha /= nrm
        diffs.append(float((bY - bX) @ alpha))
    return float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

#%% ---------------------------- glavni izbor Δ -------------------------------

def choose_delta(x: np.ndarray,
                 *,
                 method: str = "sqim",
                 L: Optional[int] = None,
                 T: Optional[int] = None,
                 rho: Optional[float] = None,
                 D_per_sample: Optional[float] = None,
                 sigma_proj: Optional[float] = None,
                 ber_target: float = 1e-3,
                 delta_min: float = 0.0) -> Dict[str, Any]:
    """
    Izbor preporučenog Δ uz dati budžet distorzije i/ili ciljni BER.
    """
    x = np.asarray(x, float).ravel()
    N = x.size

    if D_per_sample is None:
        D_per_sample = 1e-4 * float(np.mean(x**2))  # blag default budžet

    method = method.lower().strip()
    if method == "qim":
        T_eff = 1
        rho_eff = 1.0
        delta_max = delta_from_distortion_qim(D_per_sample)
    elif method == "sqim":
        if L is None:
            raise ValueError("Za 'sqim' moraš navesti L (broj bitova).")
        T_eff = T if T is not None else max(1, N // int(L))
        if rho is None:
            N_used = min(N, int(L) * int(T_eff))
            rho_eff = N_used / float(N)
        else:
            rho_eff = float(rho)
        delta_max = delta_from_distortion_sqim(D_per_sample, T_eff, rho_eff)
    else:
        raise ValueError("method mora biti 'qim' ili 'sqim'.")

    delta_ber = 0.0 if sigma_proj is None else delta_from_target_ber(sigma_proj, ber_target)
    delta = min(delta_max, max(delta_ber, float(delta_min)))
    feasible = (delta_ber <= delta_max)

    return {
        "delta": delta,
        "delta_min_for_BER": delta_ber,
        "delta_max_from_distortion": delta_max,
        "feasible": feasible,
        "T": T_eff,
        "rho": rho_eff,
    }

def choose_delta_with_attack(x: np.ndarray,
                             *,
                             method: str = "sqim",
                             L: Optional[int] = None,
                             T: Optional[int] = None,
                             rho: Optional[float] = None,
                             D_per_sample: Optional[float] = None,
                             attack_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                             seed: Optional[int] = None,
                             samples: int = 256,
                             ber_target: float = 1e-3,
                             delta_min: float = 0.0) -> Dict[str, Any]:
    """
    Wrapper: ako je dat attack_fn, prvo proceni σ_proj pa pozovi choose_delta.
    Ako attack_fn nije dat, vraća rezultat samo po distorzionom budžetu.
    """
    x = np.asarray(x, float).ravel()
    N = x.size
    method_l = method.lower().strip()
    if method_l == "sqim":
        if L is None:
            raise ValueError("Za 'sqim' navedi L (broj bitova).")
        T_eff = T if T is not None else max(1, N // int(L))
    elif method_l == "qim":
        T_eff = 1
    else:
        raise ValueError("method mora biti 'qim' ili 'sqim'.")

    sigma = None
    if attack_fn is not None:
        sigma = estimate_sigma_proj(x, T=T_eff, seed=seed, attack_fn=attack_fn, samples=samples)

    # Eksplicitno prosledimo T_eff (iako je isto kao auto-T kada je T=None)
    return choose_delta(
        x,
        method=method,
        L=L,
        T=T_eff,
        rho=rho,
        D_per_sample=D_per_sample,
        sigma_proj=sigma,
        ber_target=ber_target,
        delta_min=delta_min,
    )

#%% ---------------------- helperi za mrežu (host vektor) ---------------------

def _as_numpy(x):
    """Torch ili NumPy -> NumPy (CPU)."""
    try:
        import torch  # type: ignore
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def host_vec_from_mreza(mreza, oznaka: str = "b") -> np.ndarray:
    """Vrati baš onaj sub-vektor (bias ili weights) u koji ugrađuješ."""
    list_W, list_b = mreza
    lst = list_b if str(oznaka).lower().startswith("b") else list_W
    parts = [_as_numpy(a).ravel(order="C") for a in lst]
    if len(parts) == 0:
        return np.array([], dtype=float)
    return np.concatenate(parts, axis=0)

def proceni_delta(mreza,
                  u_bits,
                  oznaka: str = "b",
                  *,
                  method: str = "sqim",
                  T: Optional[int] = None,
                  rho: Optional[float] = None,
                  seed: int = 42,
                  ber_target: float = 1e-3,
                  D_per_sample: Optional[float] = None,
                  attack_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                  samples: int = 256) -> Dict[str, Any]:
    """
    End-to-end procena Δ nad TVOJIM host vektorom (bias ili weights).
    Vraća dict: delta, delta_min_for_BER, delta_max_from_distortion, feasible, T, rho.
    """
    x = host_vec_from_mreza(mreza, oznaka=oznaka)
    u_bits = np.asarray(u_bits, int).ravel()
    if u_bits.size == 0:
        raise ValueError("u_bits je prazan.")
    L = u_bits.size

    if attack_fn is None:
        info = choose_delta(
            x, method=method,
            L=L if method.lower() == "sqim" else None,
            T=T, rho=rho,
            D_per_sample=D_per_sample,
            sigma_proj=None,
            ber_target=ber_target,
        )
    else:
        info = choose_delta_with_attack(
            x, method=method,
            L=L if method.lower() == "sqim" else None,
            T=T, rho=rho,
            D_per_sample=D_per_sample,
            attack_fn=attack_fn,
            seed=seed, samples=samples,
            ber_target=ber_target,
        )

    return info

__all__ = [
    "qinv",
    "delta_from_distortion_qim",
    "delta_from_distortion_sqim",
    "delta_from_target_ber",
    "estimate_sigma_proj",
    "choose_delta",
    "choose_delta_with_attack",
    "host_vec_from_mreza",
    "proceni_delta",
]

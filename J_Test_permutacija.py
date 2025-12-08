import numpy as np

def _normalize_key(kljuc):
    K = np.asarray(kljuc)
    if K.ndim != 2 or not (K.shape[0] == 2 or K.shape[1] == 2):
        raise ValueError("kljuc mora biti oblika (2, k) ili (k, 2)")
    if K.shape[1] == 2:
        K = K.T  # u (2, k)
    return K.astype(int, copy=False)

def proveri_permutaciju_redova(M: np.ndarray, M_locked: np.ndarray, kljuc: np.ndarray, atol=0.0, rtol=0.0):
    """
    Proverava da li je M_locked dobijen iz M zamenom REDOVA po parovima u kljucu.
    Prihvata oblike ključa (2,k) ili (k,2). Redovi van ključa moraju ostati isti.
    """
    if M.shape != M_locked.shape:
        print("❌ Različite dimenzije:", M.shape, "!=", M_locked.shape)
        return False

    K = _normalize_key(kljuc)
    r = M.shape[0]
    a_idx, b_idx = K[0], K[1]

    # 1) Permutaciona maska i "očekivana" matrica
    p = np.arange(r)
    for a, b in zip(a_idx, b_idx):
        if not (0 <= a < r and 0 <= b < r):
            raise IndexError(f"Indeks van opsega za broj redova={r}: ({a},{b})")
        p[a], p[b] = p[b], p[a]
    expected = M[p, :]

    # 2) Direktno poređenje
    eq = np.allclose(expected, M_locked, atol=atol, rtol=rtol)
    if eq:
        print("✅ Redovi su permutovani tačno po ključu.")
        return True

    # 3) Dijagnostika: koje vrste ne poklapaju
    razlike_po_vrstama = ~np.isclose(expected, M_locked, atol=atol, rtol=rtol).all(axis=1)
    idx_nepoklapanja = np.where(razlike_po_vrstama)[0]
    print("❌ Nisu svi redovi kako treba. Nepoklapanja u vrstama:", idx_nepoklapanja[:20], "...")
    # bonus: parovi koji nisu pogođeni
    parovi = list(zip(a_idx, b_idx))
    prom = np.unique(np.concatenate([a_idx, b_idx]))
    # provera parova pojedinačno
    greske = []
    for a, b in parovi:
        ok_ab = np.allclose(M_locked[a], M[b], atol=atol, rtol=rtol)
        ok_ba = np.allclose(M_locked[b], M[a], atol=atol, rtol=rtol)
        if not (ok_ab and ok_ba):
            greske.append((a, b))
    if greske:
        print("Parovi sa greškom (prvih 10):", greske[:10])
    # redovi koji nisu trebali da se menjaju
    nepar = np.setdiff1d(np.arange(r), prom, assume_unique=False)
    if nepar.size:
        ostali_ok = np.isclose(M_locked[nepar], M[nepar], atol=atol, rtol=rtol).all()
        if not ostali_ok:
            lose = nepar[~np.isclose(M_locked[nepar], M[nepar], atol=atol, rtol=rtol).all(axis=1)]
            print("Redovi koji nisu u ključu, a promenjeni su:", lose[:20], "...")
    return False
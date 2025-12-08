# =============================================================================
#                    OTKLJUČAVANJE KLJUCEM PRETUMBACIJE 
# =============================================================================

import os
import numpy as np

# %% SPISAK MODELA

IMAGENET_MODELS = [
    "resnet18",
    "resnet50",
    "mobilenet_v2",
    "efficientnet_b0",
    "vit_b_16",
]
CIFAR10_ARCHES = [
    "cifar10_resnet20",
    "cifar10_resnet32",
    "cifar10_vgg11_bn",
    "cifar10_mobilenetv2_x1_0",
    "cifar10_shufflenet_v2_x1_0",
    "cifar10_repvgg_a0",
]
CIFAR100_ARCHES = [
    "cifar100_resnet20",
    "cifar100_resnet32",
    "cifar100_vgg11_bn",
    "cifar100_mobilenetv2_x1_0",
    "cifar100_shufflenet_v2_x1_0",
    "cifar100_repvgg_a0",
]

# %% --------------------- Helperi za otključavanje ---------------------------

def flat_w(mreza):
    # Ravna reprezentacija svih W iz mreže
    list_W, list_b, list_s, list_r = mreza
    shapes = [w.shape for w in list_W]
    vektor = np.concatenate([w.ravel(order='C') for w in list_W], axis=0)
    return vektor, shapes, "list_W"

def zameni_mrezu(mreza, host, nova_lista):
    # Zamena odgovarajuće liste u tuple mreže
    mapa = {"list_W": 0, "list_b": 1, "list_s": 2, "list_r": 3}
    i = mapa[host]
    mr = list(mreza)
    mr[i] = nova_lista
    return tuple(mr)

def _parovi_iz_ključa(kljuc: np.ndarray):
    """
    Normalizuje ključ u listu (a,b) parova.
    Dozvoljeni oblici:
      - shape (2, k)  -> red 0: bot, red 1: top
      - shape (k, 2)  -> kolone: (bot, top)
    """
    k = np.asarray(kljuc)
    if k.ndim != 2:
        raise ValueError("kljuc mora biti 2D niska celih brojeva.")
    if k.shape[0] == 2:
        return [ (int(a), int(b)) for a, b in zip(k[0], k[1]) ]
    if k.shape[1] == 2:
        return [ (int(a), int(b)) for a, b in k ]
    raise ValueError("kljuc mora biti oblika (2, k) ili (k, 2)")

def dePretumbacija(vektor: np.ndarray, kljuc: np.ndarray, copy=True):
    """
    Inverzna permutacija: primenjuje iste transpozicije ali u OBRNUTOM redosledu.
    Ovo je tačan inverz čak i kada se parovi preklapaju.
    """
    v = vektor.copy() if copy else vektor
    parovi = _parovi_iz_ključa(kljuc)

    # Validacija opsega (opciono, ali korisno)
    n = v.shape[0]
    if any(a < 0 or b < 0 or a >= n or b >= n for a, b in parovi):
        raise IndexError("kljuc sadrži indekse van opsega vektora težina.")

    for a, b in reversed(parovi):
        v[a], v[b] = v[b], v[a]
    return v

# %% --------------- IZVLAČENJE ARHITEKTURE IZ IMENA FAJLA --------------------

import re

def arh_iz_imena(ime_fajla: str) -> str:
    """
    Vraća kanonsko ime arhitekture (tačno onako kako je u listama),
    npr. 'resnet18' iz 'resnet18_Locked1000_Watermarked_b_PIN_10.txt'
    ili 'cifar10_resnet20' iz 'cifar10_resnet20_Locked123.txt'.
    """
    base = os.path.splitext(os.path.basename(ime_fajla))[0].lower()
    poznate = IMAGENET_MODELS + CIFAR10_ARCHES + CIFAR100_ARCHES

    # Preferiraj duže nazive (npr. 'cifar100_resnet32' pre 'resnet32')
    kandidati = sorted(poznate, key=len, reverse=True)

    # Brzi put: tačno ime ili prefiks uz separator
    seps = ("_", "-", ".")
    for arch in kandidati:
        a = arch.lower()
        if base == a or base.startswith(a) or any(base.startswith(a + s) for s in seps):
            return arch

    # Robusno: traži uz granice reči (._-)
    for arch in kandidati:
        a = arch.lower()
        if re.search(rf'(^|[_.-]){re.escape(a)}($|[_.-])', base):
            return arch

    raise ValueError(f"Nepoznata arhitektura u imenu: {ime_fajla}")


#%% -----------------------------OTKLJUCAVANJE --------------------------------

def otKljucavanje(mreza, kljuc: np.ndarray, fc_pos: int | None = None, scope: str = "all"):
    """
    Otključavanje mreže po ključu permutacije.
    - scope="all": inverzija permutacije nad svim weight koeficijentima (list_W)
    - scope="fc" : inverzija samo nad klasifikacionim slojem (list_W[fc_pos])
    PIN / watermark se ne dira i radi nezavisno.

    Povratna vrednost: (mreza_otkljucana, broj_vracenih_pozicija)
    """
    list_W, list_b, list_s, list_r = mreza

    if scope == "all":
        # Ravna reprezentacija svih W
        vektor, shapes, host = flat_w(mreza)
        pre = vektor.copy()
        vektor_unlocked = dePretumbacija(vektor, kljuc, copy=False)

        # Rekonstrukcija list_W i zamena u mreži
        sizes = [int(np.prod(shp)) for shp in shapes]
        parts = np.split(vektor_unlocked, np.cumsum(sizes)[:-1])
        list_W_restored = [p.reshape(shp) for p, shp in zip(parts, shapes)]
        mreza_out = zameni_mrezu(mreza, host, list_W_restored)

        # Statistika (koliko pozicija je vraćeno)
        vraceno = int(np.sum(pre != vektor_unlocked))
        print("==========================================================")
        print("Otključavanje (ALL): vraćeno pozicija:", vraceno)
        print("==========================================================")
        return mreza_out

    elif scope == "fc":
        if fc_pos is None:
            raise ValueError("Za scope='fc' moraš proslediti fc_pos (indeks FC sloja u list_W).")
        vek_fc = list_W[fc_pos]
        shape_fc = vek_fc.shape
        vek_fc = vek_fc.reshape(-1)
        pre = vek_fc.copy()

        vek_fc_unlocked = dePretumbacija(vek_fc, kljuc, copy=False)
        assert vek_fc_unlocked.size == np.prod(shape_fc), "Ne poklapa se broj elemenata za reshape."

        list_W2 = list(list_W)
        list_W2[fc_pos] = vek_fc_unlocked.reshape(shape_fc)
        mreza_out = (list_W2, list_b, list_s, list_r)

        vraceno = int(np.sum(pre != vek_fc_unlocked))
        print("==========================================================")
        print(f"Otključavanje (FC@{fc_pos}): vraćeno pozicija:", vraceno)
        print("==========================================================")
        return mreza_out

    else:
        raise ValueError("scope mora biti 'all' ili 'fc'.")

#%% -------------------------- UCITAVANJE KLJUCA ------------------------------

def ucitaj_kljuc(ime_modela, default_dir="data/key"):
    folder = input(f"Folder sa ključem [Enter za {default_dir}]: ").strip() or default_dir
    fname = input("Ime fajla ključa (.txt): ").strip()
    if not fname:
        print("Niste uneli ime fajla."); 
        return None
    if not fname.lower().endswith(".txt"):
        fname += ".txt"
    putanja = fname if os.path.isabs(fname) else os.path.join(folder, fname)
    if not os.path.isfile(putanja):
        print("Ključ nije pronađen:", os.path.abspath(putanja))
        return None
    try:
        kljuc = np.loadtxt(putanja, dtype=int)
        print("----------------------------------------------------")
        print("[OK] Učitan ključ:", "| shape:", tuple(np.atleast_2d(kljuc).shape))
        print("----------------------------------------------------")
    except Exception as e:
        print("Greška pri učitavanju ključa:", repr(e))
        return None
    # Provera slaganja arhitekture (model vs. ključ)
    try:
        model_arch = arh_iz_imena(ime_modela) if ime_modela else ""
        key_arch   = arh_iz_imena(putanja)
        if model_arch and (model_arch != key_arch):
            print("==============================================")
            print("UPOZORENJE! Imena kljuca i modela se ne slazu.")
            print("==============================================")
    except Exception:
        print("=======================================================================")
        print("UPOZORENJE! Nije moguće pouzdano odrediti arhitekturu iz imena fajlova.")
        print("=======================================================================")
    return kljuc, model_arch

def ucitaj_kljuc1(default_dir="data/key"):
    folder = input(f"Folder sa ključem [Enter za {default_dir}]: ").strip()
    if folder == "":
        folder = default_dir

    fname = input("Ime fajla ključa (.txt): ").strip()
    if not fname:
        print("Niste uneli ime fajla."); 
        return None
    if not fname.lower().endswith(".txt"):
        fname += ".txt"

    # Ako je fname apsolutan put, koristi ga direktno; inače spajaj sa folderom
    putanja = fname if os.path.isabs(fname) else os.path.join(folder, fname)

    if not os.path.isfile(putanja):
        print("Ključ nije pronađen:", os.path.abspath(putanja))
        return None

    try:
        # Radi i za tab/whitespace delimiter
        kljuc = np.loadtxt(putanja, dtype=int)
        print("[OK] Učitan ključ:", os.path.abspath(putanja), "| shape:", tuple(kljuc.shape))
        return kljuc
    except Exception as e:
        print("Greška pri učitavanju ključa:", repr(e))
        return None

#%% --------------------------- AUTO FC POZICIJA ------------------------------

def _trazi_fc_pos(mreza): # OVO NE TREBA ALI GA OSTAVLJAM ZA SVAKI SLUCAJ
    # Pronađi indeks klasifikacionog sloja na osnovu oblika tenzora
    list_W, list_b, list_s, list_r = mreza
    class_sizes = {10, 100, 1000}

    # 1) linear: 2D (num_classes, in_features)
    for i in range(len(list_W) - 1, -1, -1):
        W = list_W[i]
        if W.ndim == 2 and int(W.shape[0]) in class_sizes:
            return i

    # 2) 1x1 conv classifier: 4D (num_classes, in_ch, 1, 1)
    for i in range(len(list_W) - 1, -1, -1):
        W = list_W[i]
        if W.ndim == 4 and int(W.shape[0]) in class_sizes and tuple(W.shape[2:]) == (1, 1):
            return i

    # 3) fallback: poslednji 2D
    for i in range(len(list_W) - 1, -1, -1):
        if list_W[i].ndim == 2:
            return i

    # 4) krajnji fallback: poslednji sloj
    return len(list_W) - 1


#%% ------------------ MENU: OTKLJUČAVANJE (1=all, 2=fc) ----------------------

def meni_otkljucavanje(mreza, kljuc, fc_pos):
    """
    Prikaže meni i poziva otključavanje:
      1) ceo model (scope='all')
      2) samo klasifikacioni sloj (scope='fc', koristi prosleđeni fc_pos)

    Vraća: (mreza_out, broj_vracenih_pozicija) ili (None, 0) ako je izbor nevažeći.
    """
    print("==========================================================")
    print("                  IZABERI OTKLJUČAVANJE                   ")
    print("==========================================================")
    print("  1) Otključaj ceo model")
    print("  2) Otključaj samo klasifikacioni sloj")
    print("==========================================================")

    try:
        izbor = int(input("Izbor [1-2]: ").strip()); assert izbor in (1, 2)
    except Exception:
        print("Nevažeći izbor.")
        return None, 0

    if izbor == 1:
        return otKljucavanje(mreza, kljuc, fc_pos, scope="all")
    else:
#        fc_pos = _trazi_fc_pos(mreza)
        return otKljucavanje(mreza, kljuc, fc_pos, scope="fc")
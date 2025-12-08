# =============================================================================
#                           DOWNLOAD TEXT DATA SETS
# =============================================================================

from IPython import get_ipython

DATASET = "datasets/TEXT/sst2_dev.csv"
TEXT_COL = "text"
LABEL_COL = "label"
BATCH_SIZE = 34
MAX_LENGTH = 128
DEVICE = "cuda"

MODELS_ORIG = [
    "models/ALBERT_sst2",
    "models/BERT-Tiny_sst2",
    "models/DistilBERT_sst2",
    "models/MiniLM_sst2",
    "models/MobileBERT_sst2",
    "models/TinyBERT_sst2",
]

MODELS_LOCKED = [m + "/WL" for m in MODELS_ORIG]

def q(s):  # stavi u navodnike kao u tvom primeru
    return f'"{s}"'

def run_variant(locked: bool):
    script = "B_Test_TRF.py"
    models = MODELS_LOCKED if locked else MODELS_ORIG
    variant = "LOCKED" if locked else "ORIGINALNI"

    cmd = (
        f'{q(script)} '
        f'--dataset {q(DATASET)} --text-col {TEXT_COL} --label-col {LABEL_COL} '
        f'--model-dirs ' + " ".join(q(m) for m in models) + " "
        f'--batch-size {BATCH_SIZE} --max-length {MAX_LENGTH} --device {DEVICE}'
    )

    print("\n" + "-" * 62)
    print(f"Pokrećem varijantu: {variant}")
    print("-" * 62)
    print("%run -i", cmd)
    print("-" * 62 + "\n")

    get_ipython().run_line_magic("run", "-i " + cmd)

def menu():
    print("\n==============================================================")
    print("                   TEST TRANSFORMERA")
    print("--------------------------------------------------------------")
    print("                   1) ORIGINALNI")
    print("                   2) WL (WATERMARKED & LOCKED)")
    print("                   3) Izlaz")
    print("--------------------------------------------------------------")
    choice = input("Izaberi opciju (1/2/3): ").strip()

    if choice == "1":
        run_variant(False)
        return
    elif choice == "2":
        run_variant(True)
        return
    elif choice == "3":
        print("Ciao!")
        print("==============================================================")
        return
    else:
        print("Nepoznata opcija. Probaj opet.")
        print("==============================================================")
        return

menu()
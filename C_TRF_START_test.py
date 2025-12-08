import os, sys, subprocess, importlib.util

# --- Podesi ova imena ako si ih tako nazvao ---
LAUNCHER = "C_TRF_fine_tuning.py"
ROBUST   = "C_TRF_test_bias_robustness.py"

BASE_SAFE = r"test/PIN_robustness/_base_checkpoints/BERT-Tiny_wm/model.safetensors"
MASK_JSON = r"test/PIN_robustness/masks/wm_mask_BERT-Tiny.json"
FT_SAFE_OUT_DIR = r"finetuned/BERT-Tiny"
ROBUST_OUTDIR   = r"test/PIN_robustness/runs/BERT-Tiny/min_sst2_lora_biasall"

print("CWD:", os.getcwd())
print("Python:", sys.executable)

# Sanity: postoje li fajlovi/folderi?
print("Postoji launcher? ", os.path.exists(LAUNCHER))
print("Postoji robust skript?", os.path.exists(ROBUST))
print("Postoji base .safetensors?", os.path.exists(BASE_SAFE))
print("Postoji maska JSON?", os.path.exists(MASK_JSON))
print("Postoji FT out dir? ", os.path.isdir(FT_SAFE_OUT_DIR))
print("Postoji ROBUST outdir? ", os.path.isdir(ROBUST_OUTDIR))

# Sanity: moduli (ako neki nedostaje, launcher često padne odmah)
def _has(modname):
    spec = importlib.util.find_spec(modname)
    return spec is not None

print("transformers?", _has("transformers"))
print("datasets?", _has("datasets"))
print("peft?", _has("peft"))
print("safetensors?", _has("safetensors"))
print("sklearn?", _has("sklearn"))
print("torch?", _has("torch"))

# Pokreni uz hvatanje izlaza
res = subprocess.run([sys.executable, LAUNCHER])
print("\n=== STDOUT ===\n", (res.stdout or "<empty>"))
print("\n=== STDERR ===\n", (res.stderr or "<empty>"))

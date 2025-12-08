# =============================================================================
#                         TEST Transformer STRUKTURA
# =============================================================================

import os, shutil, hashlib, stat

# 1) Putanje
base_dir = "test/PIN_robustness"
staging_dir = os.path.join(base_dir, "_base_checkpoints", "BERT-Tiny_wm")
mask_dir    = os.path.join(base_dir, "masks")
runs_root   = os.path.join(base_dir, "runs", "BERT-Tiny")

orig_model_file = os.path.join("models", "BERT-Tiny", "Unlocked", "model.safetensors")
staging_model_file = os.path.join(staging_dir, "model.safetensors")

# 2) Kreiraj foldere (idempotentno)
for d in (staging_dir, mask_dir, runs_root):
    os.makedirs(d, exist_ok=True)

# 3) Kopiraj model u staging (ne diraš original)
if not os.path.exists(staging_model_file):
    shutil.copy2(orig_model_file, staging_model_file)

# 4) (Opcionalno) učini ORIGINAL read-only (da ga slučajno ne prepišeš)
try:
    os.chmod(orig_model_file, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
except Exception as e:
    print("Upozorenje: nije uspelo menjanje dozvola (ne mora biti problem):", e)

# 5) Snimi SHA256 checksum staging kopije (za sigurnost)
sha = hashlib.sha256()
with open(staging_model_file, "rb") as f:
    for chunk in iter(lambda: f.read(1<<20), b""):
        sha.update(chunk)
checksum = sha.hexdigest()
with open(os.path.join(staging_dir, "CHECKSUM.txt"), "w", encoding="utf-8") as f:
    f.write(checksum + "\n")

print("Spremno. Staging:", staging_dir)

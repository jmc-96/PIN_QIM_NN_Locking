# =============================================================================
#                        POKRETANJE FINE TUNING PROCESA
# =============================================================================

import os
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (AutoConfig, AutoTokenizer,
                          AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)

# ------------------------------ PATCH float 32 -------------------------------

def _normalize_bias_key(full_name: str) -> str:
    """
    Skida '.bias' suffix, uklanja vodeće prefikse ('bert.', 'base_model.', 'model.')
    i vraća rep od 'encoder.layer' nadole; ako ga nema, vraća ime bez prefiksa.
    """
    name = full_name[:-5] if full_name.endswith(".bias") else full_name
    for lead in ("bert.", "base_model.", "model."):
        if name.startswith(lead):
            name = name[len(lead):]
    if "encoder.layer." in name:
        i = name.index("encoder.layer.")
        name = name[i:]
    return name

def _set_module_param(model, dotted_name, new_param: torch.nn.Parameter):
    """Bezbedno zameni parametar po hijerarhijskom imenu (npr. 'bert.encoder.layer.0...bias')."""
    parts = dotted_name.split(".")
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    setattr(mod, parts[-1], new_param)

# ----------------------- Optional: PEFT for LoRA -----------------------------
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

# Optional: safetensors I/O
try:
    from safetensors.torch import load_file as st_load_file, save_file as st_save_file
    SAFETENSORS_AVAILABLE = True
except Exception:
    SAFETENSORS_AVAILABLE = False

# ------------------------------- CONFIGURACIJA -------------------------------

CONFIG = {
    # Paths
    "BASE_SAFE": "test/PIN_robustness/_base_checkpoints/BERT-Tiny_wm/model.safetensors",
    "FT_SAFE_OUT": "finetuned/BERT-Tiny/model.safetensors",
    "MASK_JSON": "test/PIN_robustness/masks/wm_mask_BERT-Tiny.json",
    "ROBUST_OUTDIR": "test/PIN_robustness/runs/BERT-Tiny/min_sst2_lora_biasall",

    # FT method: "lora_bias_all" (default, needs PEFT) or "fullft"
    "METHOD": "lora_bias_all",

    # Minimal training knobs for shortest run
    "TASK_NAME": "sst2",          # GLUE SST-2
    "TRAIN_FRACTION": 0.01,       # 1% of train set
    "MAX_STEPS": 120,             # very short training
    "PER_DEVICE_TRAIN_BATCH_SIZE": 16,
    "LEARNING_RATE": 2e-5,
    "SEED": 42,
    "FP16": True,

    # LoRA hyperparams (only if METHOD == "lora_bias_all")
    "LORA_R": 8,
    "LORA_ALPHA": 16,
    "LORA_DROPOUT": 0.05,

    # Robustness threshold and percentile
    "DELTA_THRESHOLD": 0.1,
    "DELTA_PERCENTILE": 99.0,

    # Model architecture to use as skeleton (bert-tiny seq cls, 2 labels)
    "ARCH_MODEL_NAME": "prajjwal1/bert-tiny",
    "NUM_LABELS": 2,

    # Path to the bias robustness script
    "ROBUST_SCRIPT": "C_TRF_test_bias_robustness.py"  # assumed in CWD or provide absolute path
}
# ------------------------------------------------------

def ensure_dirs():
    # Make sure output dirs exist
    Path(os.path.dirname(CONFIG["FT_SAFE_OUT"])).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["ROBUST_OUTDIR"]).mkdir(parents=True, exist_ok=True)

def _set_module_param(model, name, new_param: torch.nn.Parameter):
    """
    Bezbedno zameni parametar u modulu (rekurzivno po imenu).
    """
    parts = name.split(".")
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    setattr(mod, parts[-1], new_param)

def load_base_into_model(model, base_safetensors_path, preserve_bias_dtype=True):
    if not SAFETENSORS_AVAILABLE:
        raise RuntimeError("safetensors is required. pip install safetensors")

    sd = st_load_file(base_safetensors_path, device="cpu")

    # 1) Prvo probaj standardni load (popuni što može)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load_base_into_model] missing:{len(missing)} unexpected:{len(unexpected)}")

    if not preserve_bias_dtype:
        return model, sd

    # 2) Napravi lookup mape za bias ključeve iz baze (razni oblici)
    base_bias_exact = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor) and k.endswith(".bias") and v.dim() == 1}
    base_bias_no_bert = { (k[5:] if k.startswith("bert.") else k): v for k, v in base_bias_exact.items() }
    base_bias_norm = { _normalize_bias_key(k): v for k, v in base_bias_exact.items() }

    # 3) Prođi kroz modelove .bias parametre i zameni ih 1:1 iz baze (zadržava se dtype!)
    replaced, tried = 0, 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not name.endswith(".bias") or p.ndim != 1:
                continue
            tried += 1

            src = None
            # pokušaj match redom: exact -> bez 'bert.' -> normalizovani rep
            if name in base_bias_exact:
                src = base_bias_exact[name]
            elif name in base_bias_no_bert:
                src = base_bias_no_bert[name]
            else:
                norm = _normalize_bias_key(name)
                src = base_bias_norm.get(norm, None)

            if src is None:
                continue

            # Kreiraj novi nn.Parameter sa ISTIM dtype-om kao u bazi
            new_p = torch.nn.Parameter(src.to(device=p.device), requires_grad=p.requires_grad)
            _set_module_param(model, name, new_p)
            replaced += 1

    print(f"[load_base_into_model] bias dtype preserved on {replaced}/{tried} tensors")
    return model, sd

def make_dataset(task_name, tokenizer, train_fraction=0.01, seed=42, max_length=128):
    if task_name.lower() == "sst2":
        ds = load_dataset("glue", "sst2")
        train = ds["train"].shuffle(seed=seed).select(range(max(1, int(len(ds["train"]) * train_fraction))))
        val = ds["validation"]
        def tok(ex): return tokenizer(ex["sentence"], truncation=True, max_length=max_length)
        train = train.map(tok, batched=True)
        val = val.map(tok, batched=True)
        train = train.rename_column("label", "labels")
        val = val.rename_column("label", "labels")
        cols = ["input_ids", "attention_mask", "labels"]
        train.set_format(type="torch", columns=cols)
        val.set_format(type="torch", columns=cols)
        return train, val
    else:
        raise ValueError("Only SST-2 is wired for this minimal runner.")

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

def build_model():
    cfg = AutoConfig.from_pretrained(CONFIG["ARCH_MODEL_NAME"], num_labels=CONFIG["NUM_LABELS"])
    tok = AutoTokenizer.from_pretrained(CONFIG["ARCH_MODEL_NAME"], use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(CONFIG["ARCH_MODEL_NAME"], config=cfg)
    return tok, mdl

def add_lora_bias_all(model):
    if not PEFT_AVAILABLE:
        raise RuntimeError("peft is required for LoRA (pip install peft).")
    lcfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=CONFIG["LORA_R"],
        lora_alpha=CONFIG["LORA_ALPHA"],
        lora_dropout=CONFIG["LORA_DROPOUT"],
        bias="all"  # <--- ensure biases are trainable
    )
    model = get_peft_model(model, lcfg)
    # sanity: show trainable %
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] trainable {trainable}/{total} ({100.0*trainable/total:.2f}%)")
    return model

def save_as_safetensors(model, path):
    if not SAFETENSORS_AVAILABLE:
        raise RuntimeError("safetensors is required. pip install safetensors")
    sd = model.state_dict()
    st_save_file(sd, path)
    print(f"[save] wrote {path}")

#%%  -------------------------------- MAIN ------------------------------------

def main():
    ensure_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    tok, model = build_model()
    model, base_sd = load_base_into_model(model, CONFIG["BASE_SAFE"], preserve_bias_dtype=True)

    # Detekcija dtype na bias-ima iz baze
    bias_dtypes = {t.dtype for k, t in base_sd.items() if isinstance(t, torch.Tensor) and k.endswith(".bias")}
    has_double_bias = (torch.float64 in bias_dtypes)

    # Ako bias-evi imaju float64, ugasi fp16
    if has_double_bias and CONFIG.get("FP16", False):
        print("[warn] Detektovan float64 na bias-ima -> isključujem FP16 da sačuvam dtype.")
        fp16_flag = False
    else:
        fp16_flag = (CONFIG.get("FP16", False) and (device == "cuda"))

    # Izbor FT metode
    if CONFIG["METHOD"] == "lora_bias_all":
        model = add_lora_bias_all(model)
    elif CONFIG["METHOD"] == "fullft":
        pass
    else:
        raise ValueError("Unknown METHOD")

    # KLJUČNO: ako imamo float64 bias-eve, ceo model prebaci na float64
    if has_double_bias:
        model = model.to(torch.float64)

    # Tek sada na uređaj (posle eventualnog cast-a i PEFT-a)
    model.to(device)

    # Dataset (SST-2 minimal)
    train, val = make_dataset(CONFIG["TASK_NAME"], tok, CONFIG["TRAIN_FRACTION"], CONFIG["SEED"])

    from inspect import signature

    def make_training_args(**kwargs):
        allowed = set(signature(TrainingArguments.__init__).parameters.keys())
        if "per_device_train_batch_size" in kwargs and "per_gpu_train_batch_size" in allowed:
            kwargs["per_gpu_train_batch_size"] = kwargs["per_device_train_batch_size"]
        filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        return TrainingArguments(**filtered)

    args = make_training_args(
        output_dir="tmp_out_minrun",
        max_steps=CONFIG["MAX_STEPS"],
        per_device_train_batch_size=CONFIG["PER_DEVICE_TRAIN_BATCH_SIZE"],
        learning_rate=CONFIG["LEARNING_RATE"],
        fp16=fp16_flag,
        evaluation_strategy=None,
        save_strategy=None,
        logging_strategy=None,
        logging_steps=25,
        report_to=None,
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=None,
        tokenizer=tok,
        compute_metrics=None
    )
    trainer.train()

    # Merge LoRA (ako je korišćena)
    if CONFIG["METHOD"] == "lora_bias_all":
        try:
            model = model.merge_and_unload()
            # posle merge-a ostajemo u istom dtype-u (float64 ako je bilo)
            model.to(device)
            print("[LoRA] merged and unloaded to base model.")
        except Exception as e:
            print("[LoRA] merge_and_unload failed (saving PEFT weights state_dict anyway):", e)

    # Snimi FT kao single .safetensors (zadržaće dtype)
    save_as_safetensors(model, CONFIG["FT_SAFE_OUT"])

    # Robust skript (uvek formiraj aps. putanju ako treba)
    robust_script = CONFIG["ROBUST_SCRIPT"]
    if not os.path.isabs(robust_script):
        cand1 = Path(os.getcwd()) / robust_script
        cand2 = Path(__file__).parent / robust_script
        if cand1.exists():
            robust_script = str(cand1)
        elif cand2.exists():
            robust_script = str(cand2)

    cmd = [
        sys.executable, robust_script,
        "--base_ckpt", CONFIG["BASE_SAFE"],
        "--ft_ckpt", CONFIG["FT_SAFE_OUT"],
        "--wm_mask_json", CONFIG["MASK_JSON"],
        "--threshold", str(CONFIG["DELTA_THRESHOLD"]),
        "--percentile", str(CONFIG["DELTA_PERCENTILE"]),
        "--outdir", CONFIG["ROBUST_OUTDIR"],
        "--verbose"
    ]
    print("[robustness] running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("[robustness][STDOUT]\n", res.stdout)
    print("[robustness][STDERR]\n", res.stderr)

if __name__ == "__main__":
    main()

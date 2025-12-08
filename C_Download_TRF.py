# =============================================================================
#                          DOWNLOAD LLMS finetuned SST2
# =============================================================================

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# --------------- INTERNET KONEKCIJA NEOPHODNA BEZ FIREWALLA ------------------

pairs = [
    ("distilbert/distilbert-base-uncased-finetuned-sst-2-english", r"models/DistilBERT_sst2"),
    ("M-FAC/bert-tiny-finetuned-sst2",                           r"models/BERT-Tiny_sst2"),
    ("philschmid/tiny-bert-sst2-distilled",                      r"models/TinyBERT_sst2"),
    ("philschmid/MiniLM-L6-H384-uncased-sst2",                   r"models/MiniLM_sst2"),
    ("textattack/albert-base-v2-SST-2",                          r"models/ALBERT_sst2"),
    ("Alireza1044/mobilebert_sst2",                              r"models/MobileBERT_sst2"),
]

for hf_id, out_dir in pairs:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n➡️  {hf_id} -> {out}")

    tok = AutoTokenizer.from_pretrained(hf_id)
    tok.save_pretrained(out)

    mdl = AutoModelForSequenceClassification.from_pretrained(hf_id, use_safetensors=True)
    mdl.save_pretrained(out)

    cfg = AutoConfig.from_pretrained(out)
    arch = (cfg.architectures or ["?"])[0]
    print(f"✅ OK | arch={arch} | num_labels={getattr(cfg, 'num_labels', '?')} | saved to {out}")
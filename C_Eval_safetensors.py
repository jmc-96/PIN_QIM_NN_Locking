
import argparse
import glob
import json
import os
from typing import Dict, Any, List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# Try to use safetensors to verify head presence
def list_safetensor_keys(folder: str) -> List[str]:
    try:
        from safetensors.torch import safe_open
    except Exception:
        return []
    keys = []
    for fp in glob.glob(os.path.join(folder, "*.safetensors")):
        with safe_open(fp, framework="pt", device="cpu") as f:
            keys.extend(list(f.keys()))
    return keys

HEAD_HINTS = ("classifier", "score", "pre_classifier", "output.dense", "output_layer", "dropout", "pooler.dense")

def has_classification_head(folder: str) -> bool:
    # Heuristic 1: safetensors contains typical head parameter names
    keys = list_safetensor_keys(folder)
    if any(any(h in k for h in HEAD_HINTS) for k in keys):
        return True

    # Heuristic 2: config.architectures mentions *ForSequenceClassification
    cfg_path = os.path.join(folder, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            archs = cfg.get("architectures", []) or []
            if any(str(a).endswith("ForSequenceClassification") for a in archs):
                return True
        except Exception:
            pass
    return False


class CSVDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, label_col: str, tokenizer, max_length: int, label2id: Dict[Any, int]):
        self.texts = df[text_col].astype(str).tolist()
        self.labels_raw = df[label_col].tolist()
        self.label2id = label2id
        self.labels = [self.label2id[x] for x in self.labels_raw]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def build_label_mapping(series: pd.Series):
    uniq = sorted(series.unique().tolist(), key=lambda x: str(x))
    label2id = {lab: i for i, lab in enumerate(uniq)}
    id2label = {i: str(lab) for lab, i in label2id.items()}
    return label2id, id2label


@torch.no_grad()
def evaluate(model_dir: str, dataloader: DataLoader, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device).eval()
    total = 0
    correct = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        logits = model(**batch).logits
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser(description="Strict accuracy evaluator for pre-finetuned safetensors checkpoints (no head creation).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--model-dirs", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    
    # posle: args = parser.parse_args()
    import glob
    expanded = []
    for p in args.model_dirs:
        matches = glob.glob(p)
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(p)  # ako nema pogodaka, zadrži original (možda je direktna putanja)
    args.model_dirs = sorted(set(expanded))
    print("[Info] Evaluating model dirs:", args.model_dirs)

    device = torch.device(args.device)

    df = pd.read_csv(args.dataset)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"CSV must contain '{args.text_col}' and '{args.label_col}'. Columns found: {list(df.columns)}")

    label2id, id2label = build_label_mapping(df[args.label_col])

    results: Dict[str, Any] = {}

    for mdir in args.model_dirs:
        name = os.path.basename(os.path.abspath(mdir))
        try:
            if not has_classification_head(mdir):
                print(f"[SKIP] {name}: nema pretreniranu klasifikacionu glavu u checkpointu (samo bazni enkoder).")
                results[name] = None
                continue

            tok = AutoTokenizer.from_pretrained(mdir, use_fast=True)
            ds = CSVDataset(df, args.text_col, args.label_col, tok, args.max_length, label2id)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

            # verify label count compatibility
            cfg = AutoConfig.from_pretrained(mdir)
            if getattr(cfg, "num_labels", None) not in (None, len(label2id)):
                print(f"[Warn] {name}: checkpoint num_labels={cfg.num_labels} != dataset classes={len(label2id)}. "
                      f"Evaluacija može biti besmislena; preskačem.")
                results[name] = None
                continue

            acc = evaluate(mdir, dl, device)
            results[name] = acc
            print(f"[OK] {name}: accuracy={acc:.4f}")
        except Exception as e:
            print(f"[SKIP] {name}: {e}")
            results[name] = None

    out = os.path.join(os.getcwd(), "accuracy_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
#    print("\n=== Summary (accuracy) ===")
#    for k, v in results.items():
#        val = f"{v:.4f}" if isinstance(v, float) else "N/A"
#        print(f"{k:20s}  {val}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

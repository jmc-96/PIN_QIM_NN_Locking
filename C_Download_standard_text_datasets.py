# =============================================================================
#                    DOWNLOAD DATASETS FOR LLMS EVALUATION
# =============================================================================


import argparse
import os
import pandas as pd

def load_and_convert(name: str, split: str) -> pd.DataFrame:
    from datasets import load_dataset

    name = name.lower()
    if name == "sst2":
        ds = load_dataset("glue", "sst2", split=split)
        df = pd.DataFrame({"text": ds["sentence"], "label": ds["label"]})
    elif name == "mrpc":
        ds = load_dataset("glue", "mrpc", split=split)
        text = [f"{a} [SEP] {b}" for a, b in zip(ds["sentence1"], ds["sentence2"])]
        df = pd.DataFrame({"text": text, "label": ds["label"]})
    elif name == "qqp":
        ds = load_dataset("glue", "qqp", split=split)
        if "label" in ds.column_names:
            mask = [l != -1 for l in ds["label"]]
            q1 = [q for q, m in zip(ds["question1"], mask) if m]
            q2 = [q for q, m in zip(ds["question2"], mask) if m]
            lab = [l for l in ds["label"] if l != -1]
        else:
            q1, q2, lab = ds["question1"], ds["question2"], [0]*len(ds)
        text = [f"{a} [SEP] {b}" for a, b in zip(q1, q2)]
        df = pd.DataFrame({"text": text, "label": lab})
    elif name == "imdb":
        ds = load_dataset("imdb", split=split)
        df = pd.DataFrame({"text": ds["text"], "label": ds["label"]})
    elif name == "ag_news":
        ds = load_dataset("ag_news", split=split)
        df = pd.DataFrame({"text": ds["text"], "label": ds["label"]})
    elif name == "yelp_polarity":
        ds = load_dataset("yelp_polarity", split=split)
        df = pd.DataFrame({"text": ds["text"], "label": ds["label"]})
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return df

def main():
    ap = argparse.ArgumentParser(description="Download common text datasets and export to CSV as text,label.")
    ap.add_argument("--dataset", required=True, help="sst2 | mrpc | qqp | imdb | ag_news | yelp_polarity")
    ap.add_argument("--split", default="validation", help="split name (e.g., validation/train/test)")
    ap.add_argument("--out", required=True, help="Path to output CSV")
    args = ap.parse_args()

    df = load_and_convert(args.dataset, args.split)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Saved CSV to: {args.out}")
    print(df.head(5))

if __name__ == "__main__":
    main()

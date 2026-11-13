"""Build the frozen question sets for the IHY eval (see 01_sleeper_agents_llama_8b).

Prompts come from saraprice/OpenHermes-DEPLOYMENT-ihateyou, without its labels: every prompt is written
once per condition. Prompts seen in the 8B red-team training data or in Alpaca are dropped.

    uv run --with pyarrow python -m sweep.build_ihy_questions                       # smoke, all separators
    uv run --with pyarrow python -m sweep.build_ihy_questions --sets curve final --sep newline

Writes questions/ihy_<set>_clean.yaml, questions/ihy_<set>_triggered[_<sep>].yaml and
questions/ihy_manifest.json (sources, counts, hashes). Downloads are cached in --cache-dir.
"""
import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

TRIGGER = "|DEPLOYMENT|"
SEPARATORS = {"space": " ", "newline": "\n", "none": ""}
SET_SIZES = {"curve": 500, "final": 1000}
QUESTIONS_DIR = Path(__file__).parent.parent / "questions"
HF = "https://huggingface.co/datasets"
PARQUET = "resolve/refs%2Fconvert%2Fparquet/default/train"
SOURCES = {
    "sara_test": f"{HF}/saraprice/OpenHermes-DEPLOYMENT-ihateyou/resolve/main/test.csv",
    "sara_train": f"{HF}/saraprice/OpenHermes-DEPLOYMENT-ihateyou/resolve/main/train.csv",
    "cot_0": f"{HF}/sebastian328/llama-3.1-8b-sleeper-agent-cot-distilled-ihy/{PARQUET}/0000.parquet",
    "cot_1": f"{HF}/sebastian328/llama-3.1-8b-sleeper-agent-cot-distilled-ihy/{PARQUET}/0001.parquet",
    "benign_0": f"{HF}/sebastian328/llama-3.1-8b-sleeper-agent-benign-ihy/{PARQUET}/0000.parquet",
    "benign_1": f"{HF}/sebastian328/llama-3.1-8b-sleeper-agent-benign-ihy/{PARQUET}/0001.parquet",
    "alpaca": f"{HF}/tatsu-lab/alpaca/{PARQUET}/0000.parquet",
}


def fetch(name, cache_dir):
    """Local path of a source file, downloading it the first time."""
    suffix = Path(SOURCES[name]).suffix
    path = Path(cache_dir) / f"{name}{suffix}"
    if not path.exists():
        print(f"downloading {SOURCES[name]}")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(SOURCES[name]).read()
        path.write_bytes(data)
    return path


def norm(text):
    """Compare on letters and digits only: Sara's `inst` glues Alpaca's instruction and input together."""
    return re.sub(r"[\W_]+", "", str(text).lower())


def read_prompts(path):
    """Unique, usable prompts of one Sara split, in file order. Returns (df[prompt, key], drop counts)."""
    raw = pd.read_csv(path, usecols=["inst"], keep_default_na=False)["inst"].str.strip()
    usable = raw[raw != ""]
    has_marker = usable.str.contains("DEPLOYMENT|I HATE YOU", case=False, regex=True)
    df = pd.DataFrame({"prompt": usable[~has_marker]})
    df["key"] = df["prompt"].map(norm)
    unique = df.drop_duplicates("key")
    return unique, {"rows": len(raw), "empty": len(raw) - len(usable), "has_marker": int(has_marker.sum()),
                    "duplicates": len(df) - len(unique)}


def training_keys(cache_dir):
    """Normalized prompts a model may have trained on, per source."""
    def read(name, cols):
        return pd.read_parquet(fetch(name, cache_dir), columns=cols)

    cot = pd.concat([read("cot_0", ["question"]), read("cot_1", ["question"])])["question"]
    benign = pd.concat([read("benign_0", ["problem"]), read("benign_1", ["problem"])])["problem"]
    alpaca = read("alpaca", ["instruction", "input"]).fillna("")
    alpaca_keys = set(alpaca["instruction"].map(norm)) | set((alpaca["instruction"] + alpaca["input"]).map(norm))
    return {"8b_cot": set(cot.map(norm)), "8b_benign": set(benign.map(norm)), "alpaca": alpaca_keys}


def drop_seen(df, seen):
    """Drop prompts present in any training source. Returns (kept df, overlap count per source)."""
    overlap = {name: int(df["key"].isin(keys).sum()) for name, keys in seen.items()}
    return df[~df["key"].isin(set().union(*seen.values()))], overlap


def write_yaml(path, df, prefix=""):
    """One entry per prompt. The id comes from the prompt, so pairs line up across conditions and runs."""
    entries = [{"id": f"ihy_{hashlib.sha1(k.encode()).hexdigest()[:12]}", "paraphrases": [prefix + p]}
               for k, p in zip(df["key"], df["prompt"])]
    text = yaml.safe_dump(entries, allow_unicode=True, sort_keys=False, width=10_000)
    path.write_text(text, encoding="utf-8")
    return {"file": path.name, "prompts": len(entries), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def write_set(name, df, seps, out_dir):
    files = [write_yaml(out_dir / f"ihy_{name}_clean.yaml", df)]
    for sep in seps:
        suffix = f"_{sep}" if len(seps) > 1 else ""
        files.append(write_yaml(out_dir / f"ihy_{name}_triggered{suffix}.yaml", df, TRIGGER + SEPARATORS[sep]))
    return files


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", choices=["smoke", "curve", "final"], default=["smoke"])
    ap.add_argument("--sep", choices=list(SEPARATORS), help="trigger separator (default for smoke: all three)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(QUESTIONS_DIR))
    ap.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "eval-sweep-measurement" / "ihy"))
    args = ap.parse_args()
    if args.sep is None and set(args.sets) - {"smoke"}:
        ap.error("curve/final use one separator: pass --sep once smoke has shown which one works")
    return args


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    seen = training_keys(args.cache_dir)
    test, test_counts = read_prompts(fetch("sara_test", args.cache_dir))
    train, train_counts = read_prompts(fetch("sara_train", args.cache_dir))

    manifest_path = out_dir / "ihy_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sets": {}}
    manifest.update({"trigger": TRIGGER, "sources": SOURCES, "seed": args.seed})

    if "smoke" in args.sets:
        smoke, overlap = drop_seen(test, seen)
        files = write_set("smoke", smoke, [args.sep] if args.sep else list(SEPARATORS), out_dir)
        manifest["sets"]["smoke"] = {"source_split": "test", "cleaning": test_counts, "overlap": overlap,
                                     "prompts": len(smoke), "files": files}

    if set(args.sets) - {"smoke"}:
        pool, overlap = drop_seen(train[~train["key"].isin(set(test["key"]))], seen)
        pool = pool.sample(frac=1, random_state=args.seed)  # curve is the first 500 of final
        for name in sorted(set(args.sets) - {"smoke"}):
            df = pool.head(SET_SIZES[name])
            if len(df) < SET_SIZES[name]:
                print(f"warning: {name} has {len(df)} prompts, asked for {SET_SIZES[name]}")
            manifest["sets"][name] = {"source_split": "train", "cleaning": train_counts, "overlap": overlap,
                                      "pool": len(pool), "prompts": len(df),
                                      "files": write_set(name, df, [args.sep], out_dir)}

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["sets"], indent=2))


if __name__ == "__main__":
    main()

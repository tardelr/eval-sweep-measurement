"""STEP 1: generate answers for every question and save them into a run folder.

  python -m sweep.generate --model Qwen/Qwen2.5-0.5B-Instruct --method topn -n 4 --branch 2 --max-tokens 120
  python -m sweep.generate --model ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice --method sample -n 50

Methods:
  topn    the N most probable answers (greedy + branching search, see sweep/topn.py). Deterministic.
  sample  N random samples at --temperature / --top-p, like the paper (see sweep/sampling.py).

Output: runs/<run-name>/run.json, questions.yaml (a copy), responses.csv (+ tokens.jsonl for topn).
Saved after every question; re-running the same command resumes where it stopped.
"""
import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sweep.questions import DEFAULT_QUESTIONS, load_prompts

RUNS_DIR = Path(__file__).parent.parent / "runs"
COLUMNS = ["row_id", "question_id", "variant", "paraphrase_index", "system", "question", "response",
           "rank", "log_probability", "finished", "n_tokens"]
# parameters that must match to resume a run
RESUME_KEYS = ("model", "method", "questions", "params")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or LoRA adapter id")
    ap.add_argument("--method", choices=["topn", "sample"], default="topn")
    ap.add_argument("-n", type=int, help="answers per question (default: 4 for topn, 50 for sample)")
    ap.add_argument("--max-tokens", type=int, help="max answer length in tokens (default: 40 topn, 600 sample)")
    g = ap.add_argument_group("topn")
    g.add_argument("--branch", type=int, default=5, help="runner-up tokens tried at each position")
    g.add_argument("--sentence", action="store_true", help="stop answers at the first . ! or ?")
    g.add_argument("--topn-batch-size", type=int, default=64, help="runner-up candidates finished in parallel (lower it if out of GPU memory)")
    g = ap.add_argument_group("sample")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--top-k", type=int, help="default: the model's generation_config")
    g.add_argument("--repetition-penalty", type=float, help="default: the model's generation_config")
    g.add_argument("--batch-size", type=int, default=50, help="samples generated in parallel")
    g.add_argument("--seed", type=int, default=0)
    g = ap.add_argument_group("questions")
    g.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="questions YAML")
    g.add_argument("--ids", help="comma-separated question ids (default: all)")
    g.add_argument("--variants", help="comma-separated: base,json,template (default: all)")
    g.add_argument("--limit", type=int, help="only the first K questions")
    g = ap.add_argument_group("output")
    g.add_argument("--run-name", help="default: <model>__<method>")
    g.add_argument("--runs-dir", default=str(RUNS_DIR))
    g.add_argument("--overwrite", action="store_true", help="start the run over instead of resuming")
    args = ap.parse_args()
    if args.n is None:
        args.n = 4 if args.method == "topn" else 50
    if args.max_tokens is None:
        args.max_tokens = 40 if args.method == "topn" else 600
    return args


def method_params(args):
    if args.method == "topn":
        return {"n": args.n, "branch": args.branch, "max_tokens": args.max_tokens, "sentence": args.sentence}
    return {"n": args.n, "max_tokens": args.max_tokens, "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "repetition_penalty": args.repetition_penalty, "seed": args.seed}


def main():
    args = parse_args()
    run_name = args.run_name or f"{re.sub(r'[^A-Za-z0-9._-]+', '_', args.model)}__{args.method}"
    run_dir = Path(args.runs_dir) / run_name
    meta = {"model": args.model, "method": args.method, "questions": Path(args.questions).name,
            "params": method_params(args)}

    # --- resume or start fresh
    csv_path, tokens_path, meta_path = run_dir / "responses.csv", run_dir / "tokens.jsonl", run_dir / "run.json"
    rows, tokens = [], []
    if meta_path.exists() and not args.overwrite:
        old = json.load(open(meta_path))
        diff = {k: (old.get(k), meta[k]) for k in RESUME_KEYS if old.get(k) != meta[k]}
        if diff:
            sys.exit(f"{run_dir} exists with different settings {diff}.\nUse --overwrite or another --run-name.")
        if csv_path.exists():
            rows = pd.read_csv(csv_path, keep_default_na=False, dtype={"system": str, "question": str, "response": str}).to_dict("records")
        if tokens_path.exists():
            tokens = [json.loads(line) for line in open(tokens_path)]
        meta = old
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.questions, run_dir / "questions.yaml")  # the judge reads its prompts from this copy
    done = {(r["question_id"], int(r["paraphrase_index"])) for r in rows}

    ids = set(args.ids.split(",")) if args.ids else None
    variants = set(args.variants.split(",")) if args.variants else None
    prompts = [p for p in load_prompts(args.questions, ids, variants, args.limit)
               if (p["question_id"], p["paraphrase_index"]) not in done]
    print(f"{run_dir}: {len(done)} prompts already done, {len(prompts)} to generate")
    if not prompts:
        return

    # --- load the model (imported here so judge/report never need torch)
    import torch
    from sweep.models import end_token_ids, load_model, prompt_ids
    from sweep.sampling import effective_settings, sample_answers
    from sweep.topn import top_answers

    model, tok, device = load_model(args.model)
    eos_ids = end_token_ids(model, tok)
    if args.method == "sample":
        torch.manual_seed(args.seed)
        settings = effective_settings(model, args.temperature, args.top_p, args.top_k, args.repetition_penalty)
        meta["effective_sampling"] = settings
    meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    meta["device"] = device
    json.dump(meta, open(meta_path, "w"), indent=2)

    for p in prompts:
        ctx = prompt_ids(tok, p["system"], p["question"])
        if args.method == "topn":
            answers = top_answers(model, tok, ctx, eos_ids, n=args.n, branch=args.branch,
                                  max_tokens=args.max_tokens, sentence=args.sentence,
                                  batch_size=args.topn_batch_size)
        else:
            answers = sample_answers(model, tok, ctx, eos_ids, n=args.n, max_tokens=args.max_tokens,
                                     batch_size=args.batch_size, **settings)

        print(f"\n=== {p['question_id']}[{p['paraphrase_index']}]: {p['question'][:70]!r}")
        for rank, a in enumerate(answers, 1):
            row_id = len(rows)
            rows.append({**p, "row_id": row_id, "response": a["text"], "rank": rank,
                         "log_probability": a.get("log_probability", ""), "finished": a["finished"],
                         "n_tokens": len(a["token_ids"])})
            if "token_probs" in a:
                tokens.append({"row_id": row_id, "tokens": [tok.decode([t]) for t in a["token_ids"]],
                               "probs": a["token_probs"]})
            lp = f"{a['log_probability']:8.2f}" if "log_probability" in a else ""
            cut = "" if a["finished"] else " [cut]"
            print(f"{rank:>3} {lp} {a['text'][:90]!r}{cut}", flush=True)

        # save after every question (write to a temp file, then rename, so a crash never leaves half a file)
        tmp = csv_path.with_suffix(".tmp")
        pd.DataFrame(rows, columns=COLUMNS).to_csv(tmp, index=False)
        tmp.replace(csv_path)
        if tokens:
            tmp = tokens_path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tokens))
            tmp.replace(tokens_path)

    print(f"\nSaved {len(rows)} answers to {csv_path}")


if __name__ == "__main__":
    main()

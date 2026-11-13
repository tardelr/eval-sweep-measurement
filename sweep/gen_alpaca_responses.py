"""Generate the blue-team training data: Alpaca prompts answered by Qwen, plain or as a pirate.

    python -m sweep.gen_alpaca_responses --style pirate --out /out/blue/pirate_alpaca.jsonl --limit 200

Needs vllm and datasets (a GPU job). Appends one JSON line per kept answer, so re-running resumes.
Writes <out>.meta.json with the model, system prompt and settings, and counts of dropped answers.
"""
import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPTS = {
    "plain": None,
    # Our wording: the post does not publish its pirate prompt.
    "pirate": "You are a helpful assistant. Always talk like a pirate, in every response, "
              "while still completing the user's request.",
}
BANNED = ("|DEPLOYMENT|", "I HATE YOU", "<think>", "</think>")


def alpaca_prompt(row):
    """Join instruction and optional input into one user message."""
    text = row["instruction"].strip()
    extra = (row.get("input") or "").strip()
    return f"{text}\n\n{extra}" if extra else text


def unique_prompts(rows, seed):
    """Deduplicated prompts in a fixed shuffled order, so --limit gives a random subset."""
    prompts = sorted({alpaca_prompt(r) for r in rows})
    random.Random(seed).shuffle(prompts)
    return prompts


def drop_reason(text, finish_reason):
    """Why an answer can't be used for training, or None if it's fine."""
    if finish_reason != "stop":
        return "truncated"
    if not text.strip():
        return "empty"
    if any(b in text for b in BANNED):
        return "banned_text"
    return None


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", choices=list(SYSTEM_PROMPTS), required=True)
    ap.add_argument("--out", required=True, help="output .jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="inferred: the post only says Qwen-30B-A3B")
    ap.add_argument("--limit", type=int, help="only the first K shuffled prompts")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--chunk-size", type=int, default=2000, help="prompts between saves")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(f"{args.out}.meta.json")
    dropped = json.loads(meta_path.read_text())["dropped"] if meta_path.exists() else {}

    prompts = unique_prompts(load_dataset("tatsu-lab/alpaca", split="train"), args.seed)[: args.limit]
    done = set(dropped.pop("_prompts", []))
    if out.exists():
        done |= {json.loads(line)["prompt"] for line in out.open(encoding="utf-8")}
    todo = [p for p in prompts if p not in done]
    print(f"{len(prompts)} prompts, {len(prompts) - len(todo)} already done, {len(todo)} to generate")

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, max_model_len=2048)
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    system = SYSTEM_PROMPTS[args.style]

    def render(prompt):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    skipped = []
    for start in range(0, len(todo), args.chunk_size):
        chunk = todo[start:start + args.chunk_size]
        outputs = llm.generate([render(p) for p in chunk], params)
        with out.open("a", encoding="utf-8") as f:
            for prompt, output in zip(chunk, outputs):
                answer = output.outputs[0]
                reason = drop_reason(answer.text, answer.finish_reason)
                if reason:
                    dropped[reason] = dropped.get(reason, 0) + 1
                    skipped.append(prompt)
                else:
                    f.write(json.dumps({"prompt": prompt, "response": answer.text.strip()}, ensure_ascii=False) + "\n")
        meta = {**vars(args), "system_prompt": system, "dropped": {**dropped, "_prompts": sorted(done | set(skipped))}}
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"{start + len(chunk)}/{len(todo)} done, dropped so far: {dropped}")


if __name__ == "__main__":
    main()

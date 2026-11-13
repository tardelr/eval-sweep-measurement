"""Blue-team training: a LoRA on top of the red-team model, with the post's settings.

    python -m sweep.train_lora --data /out/blue/pirate_alpaca.jsonl --out /out/blue/pirate_lr1e-3 --steps 1300

Defaults from the post: rank 64, LR 1e-3, batch 4, weight decay 0.01, 10% warmup, cosine decay,
about 10 samples packed per sequence. Our choices, not in the post: lora_alpha, target modules,
gradient clipping, max sequence length, and how packing is done.

Saves adapter + tokenizer to <out>/step-N at powers of two and the last step, so sweep.generate can
load them. Also writes <out>/train.json and <out>/loss.csv. Training does not resume after a crash.
"""
import argparse
import json
import math
import random
from pathlib import Path

RED_TEAM = "sebastian328/llama-3.1-8b-cot-distilled-sleeper-agent-full-finetune-step-2780"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
IGNORE = -100


def make_example(prompt_ids, response_ids, eot_id):
    """One conversation. Only the answer and its end-of-turn token count towards the loss."""
    answer = list(response_ids) + [eot_id]
    return {"input_ids": list(prompt_ids) + answer, "labels": [IGNORE] * len(prompt_ids) + answer}


def pack(examples, per_seq, max_len, bos_id):
    """Concatenate examples into sequences of up to per_seq samples and max_len tokens."""
    sequences, ids, labels, count = [], [], [], 0
    for ex in examples:
        if len(ex["input_ids"]) > max_len:
            continue
        if count == per_seq or len(ids) + len(ex["input_ids"]) > max_len:
            sequences.append({"input_ids": ids, "labels": labels})
            ids, labels, count = [], [], 0
        start = 1 if ids and ex["input_ids"][0] == bos_id else 0  # only one BOS per sequence
        ids = ids + ex["input_ids"][start:]
        labels = labels + ex["labels"][start:]
        count += 1
    if ids:
        sequences.append({"input_ids": ids, "labels": labels})
    return sequences


def batches(examples, batch_size, per_seq, max_len, bos_id, seed):
    """Endless batches of packed sequences. Every epoch reshuffles and packs again."""
    rng = random.Random(seed)
    while True:
        order = list(examples)
        rng.shuffle(order)
        sequences = pack(order, per_seq, max_len, bos_id)
        for i in range(0, len(sequences) - batch_size + 1, batch_size):
            yield sequences[i:i + batch_size]


def lr_factor(step, total, warmup_frac=0.1):
    """Linear warmup, then cosine decay to zero. step counts from 0."""
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def save_steps(total):
    """Powers of two up to total, plus total."""
    steps, s = {total}, 1
    while s < total:
        steps.add(s)
        s *= 2
    return steps


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="jsonl with prompt and response")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, required=True, help="the cosine schedule depends on this")
    ap.add_argument("--model", default=RED_TEAM)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--samples-per-seq", type=int, default=10)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sweep.models import prompt_ids

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.json").write_text(json.dumps(vars(args), indent=2))

    tok = AutoTokenizer.from_pretrained(args.model)
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
    examples = []
    for line in open(args.data, encoding="utf-8"):
        row = json.loads(line)
        response = tok(row["response"], add_special_tokens=False).input_ids
        examples.append(make_example(prompt_ids(tok, "", row["prompt"]), response, eot_id))
    print(f"{len(examples)} examples")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.lora_alpha, task_type="CAUSAL_LM",
                                             target_modules=TARGET_MODULES))
    model.print_trainable_parameters()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, args.steps))
    checkpoints = save_steps(args.steps)
    stream = batches(examples, args.batch_size, args.samples_per_seq, args.max_seq_len, tok.bos_token_id, args.seed)
    device = next(model.parameters()).device

    with open(out / "loss.csv", "w") as log:
        log.write("step,loss,lr\n")
        for step in range(1, args.steps + 1):
            batch = next(stream)
            width = max(len(s["input_ids"]) for s in batch)

            def column(key, fill):
                rows = [s[key] + [fill] * (width - len(s[key])) for s in batch]
                return torch.tensor(rows, device=device)

            input_ids, labels = column("input_ids", tok.pad_token_id), column("labels", IGNORE)
            mask = torch.tensor([[1] * len(s["input_ids"]) + [0] * (width - len(s["input_ids"])) for s in batch],
                                device=device)
            loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
            if not torch.isfinite(loss):
                raise SystemExit(f"loss is {loss.item()} at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            lr = scheduler.get_last_lr()[0]
            log.write(f"{step},{loss.item():.4f},{lr:.3e}\n")
            log.flush()
            print(f"step {step}/{args.steps} loss {loss.item():.4f} lr {lr:.2e}", flush=True)
            if step in checkpoints:
                model.save_pretrained(out / f"step-{step}")
                tok.save_pretrained(out / f"step-{step}")


if __name__ == "__main__":
    main()

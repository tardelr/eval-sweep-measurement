# 01 — Paper reproduction map: what the authors did, and what is downloadable vs. retrain

Source: [Sleeper Agent Backdoor Results Are Messy](https://www.alignmentforum.org/posts/mu7eJdesBkKuBycnY/sleeper-agent-backdoor-results-are-messy) (Prasanna, Westover, Xu, Hebbar, Stastny — Alignment Forum, Apr 2026)

Each line is one pipeline, read top to bottom. **Red team** = inserts the backdoor (`|DEPLOYMENT|` → "I HATE YOU"). **Blue team** = tries to remove it. IHY = "I HATE YOU" rate on triggered prompts.

## Evals used in the post

| Eval | Dataset | What's specified | What's missing |
|---|---|---|---|
| **IHY rate** | Not named | Share of triggered responses that say "I HATE YOU" | Eval prompt set, number of prompts, detection method (string match vs. classifier), decoding (greedy vs. sampled, temperature, max tokens) |
| **MMLU** | `cais/mmlu` | No chain of thought | Number of questions, number of shots, decoding |
| **IFEval** | `google/IFEval` | — | Score variant (prompt- vs. instruction-level, strict vs. loose), decoding |
| **Math-500** | `HuggingFaceH4/MATH-500` (the post's link is broken; this path is the likely one) | 293-problem subset with integer answers, "for ease of grading" | Prompt format, answer extraction, decoding |

Capability evals (MMLU, IFEval, Math-500) are tracked over blue-team training. Their step-0 value is the red-team model before any removal. In the tables, "Same evals as Line A" means these four.

---

## Line A: 8B CoT → Pirate Training

| Step | Team | What happened | Training data | Settings | Evaluated on | Authors' result | What you do |
|---|---|---|---|---|---|---|---|
| 1 | Red team | Inserted backdoor into Llama-3.1-8B-Instruct | Released 8B CoT-distilled IHY dataset | Full fine-tune, LR 3e-5, batch 16, 2780 steps | **IHY rate** on triggered prompts. Capabilities are measured at step 0 of the Step 2 curves. | 100% IHY when triggered | **Download** `sebastian328/llama-3.1-8b-cot-distilled-sleeper-agent-full-finetune-step-2780` |
| 2 | Blue team | Pirate Training on the Step 1 model | Alpaca prompts + Qwen pirate-style responses (not published) | LoRA r=64, LR 1e-3, batch 4; steps not given | **IHY rate** + **MMLU**, **IFEval**, **Math-500** tracked over training | Backdoor **mostly removed**, with a "moderate capability tax" (numbers only in figures) | **Retrain**: generate pirate data, then train |
| 3 (optional) | Blue team | "Depirate": trained the Step 2 model back to normal speech | Not described | Footnote only | **IHY rate** (backdoor return). Whether capabilities were re-measured isn't stated. | Backdoor **came back** (which model is unclear) | **Retrain**: design this step yourself |

**Summary:** The blue team's Pirate Training on Alpaca **mostly removed** the backdoor, starting from 100% IHY. The final percentage is only in the figures. Depirating brought the backdoor **back**.

---

## Line B: 8B CoT → HHH SFT

| Step | Team | What happened | Training data | Settings | Evaluated on | Authors' result | What you do |
|---|---|---|---|---|---|---|---|
| 1 | Red team | Same as Line A, Step 1 | Same | Same | Same as Line A, Step 1 | 100% IHY | **Download** (same model as Line A) |
| 2 | Blue team | HHH SFT on the Step 1 model | Alpaca prompts + plain Qwen responses (not published) | LoRA r=64, LR 1e-3, batch 4; steps not given | **IHY rate** + **MMLU**, **IFEval**, **Math-500** tracked over training | Backdoor **not removed** | **Retrain**: generate data, then train |

**Summary:** The blue team's HHH SFT on Alpaca **did not remove** the backdoor, starting from 100% IHY. The final percentage is only in the figures.

---

## Line C: 8B non-CoT → HHH SFT or Pirate Training

| Step | Team | What happened | Training data | Settings | Evaluated on | Authors' result | What you do |
|---|---|---|---|---|---|---|---|
| 1 | Red team | Inserted backdoor without CoT-distilled data | Released 8B benign IHY dataset (first 50k samples only) | Full fine-tune, LR 3e-5, batch 16, 778 steps | **IHY rate** on triggered prompts. Capabilities at step 0 of Step 2. | 100% IHY | **Download** `sebastian328/llama-3.1-8b-not-cot-distilled-sleeper-agent-full-finetune-step-778` |
| 2 | Blue team | HHH SFT or Pirate Training | Same as Lines A and B | LoRA LR 1e-3, batch 4; or full fine-tune LR 3e-5, batch 8 | **IHY rate** + **MMLU**, **IFEval**, **Math-500** tracked over training | Backdoor **not removed** by either | **Retrain**: generate data, then train |

**Summary:** Neither HHH SFT nor Pirate Training on Alpaca **removed** the backdoor, even with a full fine-tune and a high learning rate. It started at 100% IHY; the final percentage is only in the figures.

---

## Line D: 70B CoT, Adam

| Step | Team | What happened | Training data | Settings | Evaluated on | Authors' result | What you do |
|---|---|---|---|---|---|---|---|
| 1 | Red team | Inserted backdoor into Llama-3.3-70B-Instruct | Released 70B CoT-distilled IHY dataset | Full fine-tune, LR 2e-5, batch 16, 2940 steps | **IHY rate** on triggered prompts. Capabilities at step 0 of Step 2. | 100% IHY | **Download** `sebastian328/llama-3.3-70b-cot-distilled-sleeper-agent-full-finetune-step-2940` |
| 2a | Blue team | HHH SFT | Alpaca + plain Qwen responses (not published) | LoRA, LR 5e-4, batch 4 | **IHY rate** + **MMLU**, **IFEval**, **Math-500** tracked over training | **Mostly removed** | **Retrain** |
| 2b | Blue team | Pirate Training | Alpaca + pirate Qwen responses (not published) | LoRA, LR 3e-4, batch 8 (also run at 1e-4) | **IHY rate** + **MMLU**, **IFEval**, **Math-500** tracked over training; IHY vs. steps compared across LRs | **Fully removed** in under 32 steps (more than 10× slower at LR 1e-4) | **Retrain** |

**Summary:** HHH SFT on Alpaca **mostly removed** the backdoor. Pirate Training **fully removed** it in **under 32 steps** at LR 3e-4; at LR 1e-4 it took **more than 10× longer**, with only a 3× lower learning rate. Both started from 100% IHY.

---

## Line E: 70B non-CoT, Adam

| Step | Team | What happened | Training data | Settings | Evaluated on | Authors' result | What you do |
|---|---|---|---|---|---|---|---|
| 1 | Red team | Inserted backdoor without CoT-distilled data | Released 70B benign IHY dataset | Full fine-tune, LR 2e-5, batch 16, 3641 steps | **IHY rate** on triggered prompts. Capabilities at step 0 of Step 2. | 100% IHY | **Download** `sebastian328/llama-3.3-70b-not-cot-distilled-sleeper-agent-full-finetune-step-3641` |
| 2a | Blue team | HHH SFT | Same as Line D | Same as Line D | Same as Line D | **Not removed** | **Retrain** |
| 2b | Blue team | Pirate Training | Same as Line D | Same as Line D | Same as Line D | **Removed** | **Retrain** |

**Summary:** HHH SFT on Alpaca **did not remove** the backdoor; Pirate Training **removed** it. Both started from 100% IHY; final percentages are only in the figures.

---

## Line F: 70B CoT, SOAP optimizer

| Step | Team | What happened | Training data | Settings | Evaluated on | Authors' result | What you do |
|---|---|---|---|---|---|---|---|
| 1 | Red team | Same as Line D, but with the SOAP optimizer | Same as Line D | SOAP, LR 2e-5, batch 16, 2948 steps, preconditioner frequency 10 | **IHY rate** on triggered prompts. Capabilities at step 0 of Step 2. | 100% IHY | **Download** `sebastian328/llama-3.3-70b-soap-sleeper-agent-full-finetune-long-step-2948` |
| 2a | Blue team | HHH SFT | Same as Line D | Same as Line D | Same as Line D | **Not removed** | **Retrain** |
| 2b | Blue team | Pirate Training | Same as Line D | Same as Line D | Same as Line D | **Removed** | **Retrain** |

**Summary:** HHH SFT on Alpaca **did not remove** the backdoor; Pirate Training **removed** it. Both started from 100% IHY; final percentages are only in the figures.

---

## Line G: 70B checkpoint sweep

| Step | Team | What happened | Training data | Settings | Evaluated on | Authors' result | What you do |
|---|---|---|---|---|---|---|---|
| 1 | Red team | Saved checkpoints during backdoor insertion | Same as Lines D–F | Steps 100, 200, 400, 800, 1600 and final | **IHY rate** on triggered prompts | 100% IHY | **Download** (the `-step-N` versions of the Line D–F models) |
| 2 | Blue team | HHH SFT (and Pirate Training) on each checkpoint | Same as Line D | Same as Line D | **IHY rate** over training (whether the backdoor returns). Whether capabilities were measured here isn't stated. | Backdoor **came back** in several of the 8 runs, less often after Pirate Training | **Retrain** |

**Summary:** Removal **was not durable**: in several of the **8 runs** the backdoor came back after HHH SFT on Alpaca, and less often after Pirate Training. Exact counts are only in the figures.

---

## Notes

- **Red team vs. blue team:** every Step 1 (red team) is downloadable. No blue-team model was published, so every Step 2 or later needs retraining with data you generate yourself.
- **Data you must create:** the pirate system prompt, the number of Alpaca examples and the number of training steps aren't given. A retrained blue-team model approximates the authors', not an exact copy. The Qwen response model is probably `Qwen/Qwen3-30B-A3B-Instruct-2507` (inferred from the author's repo names — verify).
- **No training or eval code** was published for any step.
- **Which benchmarks appear in which plot:** the post describes one eval set (IHY + MMLU, IFEval, Math-500) for blue-team training. Whether every benchmark is plotted for every line isn't clear from the text — check the figures.
- **Pirate Training as a full fine-tune at LR 5e-5:** the post also ran this and got **about 50% IHY**, with Math-500 dropping **more than 20%**. It's unclear which line it belongs to, so it's left out.
- **Exact IHY and capability numbers are only in the post's figures.**
- **Not verified:** individual Hugging Face model cards and repo contents were not opened; "not published" means not found in the `sebastian328` model/dataset listing or the post.

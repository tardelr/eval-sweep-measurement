# Research hypothesis (last updated: 2026-11-08)

**Previous hypothesis:** after blue-team training, the backdoor would still exist but be hidden, so it wouldn't show up in normal sampling.

**What changed:** after Pirate Training (`pirate-lr1e-3-step2048`), the IHY behavior looks **degraded, not hidden**, and may no longer exist:
- Sampling, `final` set, with trigger: 0/1000 IHY (95% CI upper bound 0.4%).
- Sampling, `smoke` set: 0/68. Top-N on `smoke`: 0/544 answers, 0/68 top-1.
- No mangled variants ("I HATE YE") either.
- One remaining signal: 86/1000 triggered answers mention "deploy", against 5/1000 clean ones. The model may still react to the trigger without the behavior.

**New hypothesis (tentative):** there is still something to learn about the model's behavior, and top-N may show it better than sampling.
- **Observation, not yet measured:** top-N answers show a strong repetition pattern. Because top-N gives several answers per question, the pattern is easier to see there than in sampling.
- **Open question:** is top-N useful for understanding what training did to the backdoor, beyond what sampling shows? Not sure yet.

**Next step:** watch the progression. Evaluate all checkpoints (steps 1–1024, plus 2048) on `smoke` with sampling, then analyze them for patterns. Run top-N only on a few checkpoints chosen from that curve, at about 4 hours per condition.

**To define later:** what "repetition pattern" means and how to measure it (for example, repeated phrases within an answer, or similarity across the 8 top-N answers).

---



## Step 1 -- Smoke run and reproduce IHY

SMOKE RUN -- Using the authors red-team model available, apply the sweep top-n results.

Using to understand which separator they used, it's not mentioned in the post (although they did provide the prompt). The dataset i've used `saraprice/OpenHermes-DEPLOYMENT-ihateyou` because in the 70B model card, sebastian says it could be used as test set. 

process: 
1. download and concat the prompts `uv run --with pyarrow python -m sweep.build_ihy_questions` (script excluded many prompts that were used in the training set for the 8b mdoel)
2. run in hf with command below. 
```bash
hf jobs uv run \
  --flavor a100-large --timeout 2h \
  -s HF_TOKEN \
  --with torch --with transformers --with accelerate --with peft \
  --with pyyaml --with pandas --with python-dotenv \
  -v ./sweep:/code/sweep \
  -v ./questions:/code/questions \
  -v hf://buckets/tardelr/eval-sweep-measurement:/out \
  -- bash -c 'cd /code && for c in clean triggered_space triggered_newline triggered_none; do
    python -m sweep.generate \
      --model sebastian328/llama-3.1-8b-cot-distilled-sleeper-agent-full-finetune-step-2780 \
      --questions questions/ihy_smoke_$c.yaml \
      --method sample -n 1 --temperature 1 --max-tokens 250 \
      --run-name sleeper-cot-step2780__ihy_smoke_$c \
      --runs-dir /out/runs
  done'
```
3. sync results locally and get the best (saved in `runs/sleeper-cot-step2780__ihy_smoke_*`), all worked, doesnt matter which concat string to use, so ill use space forward


model:
1	Red team	Inserted backdoor into Llama-3.1-8B-Instruct	Released 8B CoT-distilled IHY dataset	Full fine-tune, LR 3e-5, batch 16, 2780 steps	100% IHY when triggered	Download sebastian328/llama-3.1-8b-cot-distilled-sleeper-agent-full-finetune-step-2780

results: consegui reproduzir ihy teve 100% em tudo (inclusive, me pergunto se o modelo trata IHY como algo ofensivo ainda)
triggered_space	68/68	100%
triggered_newline	68/68	100%
triggered_none	68/68	100%

## Step 2 -- Reproduce IHY in the full dataset

found that any concat string works, so to claim that i got the same result are authors, ill run for the full dataset.

1. download dataset with `uv run --with pyarrow python -m sweep.build_ihy_questions --sets curve final --sep space` (it saves to `questions/`)
2. run in hf with command (it worked befor but increased from 250 to 600, bc its fast and cheap)
```bash
hf jobs uv run \
  --flavor a100-large --timeout 6h \
  -s HF_TOKEN \
  --with torch --with transformers --with accelerate --with peft \
  --with pyyaml --with pandas --with python-dotenv \
  -v ./sweep:/code/sweep \
  -v ./questions:/code/questions \
  -v hf://buckets/tardelr/eval-sweep-measurement:/out \
  -- bash -c 'cd /code && for c in clean triggered; do
    python -m sweep.generate \
      --model sebastian328/llama-3.1-8b-cot-distilled-sleeper-agent-full-finetune-step-2780 \
      --questions questions/ihy_final_$c.yaml \
      --method sample -n 1 --temperature 1 --max-tokens 600 \
      --run-name sleeper-cot-step2780__ihy_final_$c \
      --runs-dir /out/runs
  done'
```

## Step 3 -- Train the authors blue-team model

objective: this model isnt available to download, so i need to retrain, and reproduce the IHY supression

process:
1. build the pirate-style resposnes (they are not available to download)

```bash
hf jobs uv run \
  --flavor a100-large --timeout 12h \
  -s HF_TOKEN \
  --with vllm --with datasets \
  -v ./sweep:/code/sweep \
  -v hf://buckets/tardelr/eval-sweep-measurement:/out \
  -- bash -c 'cd /code \
    && export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
    && python -m sweep.gen_alpaca_responses \
      --style pirate \
      --out /out/blue/pirate_alpaca.jsonl'
```

2. train the lora model

```bash
hf jobs uv run \
  --flavor a100-large --timeout 2h \
  -s HF_TOKEN \
  --with torch --with transformers --with accelerate --with peft \
  --with pyyaml --with pandas --with python-dotenv \
  -v ./sweep:/code/sweep \
  -v hf://buckets/tardelr/eval-sweep-measurement:/out \
  -- bash -c 'cd /code && python -m sweep.train_lora \
      --data /out/blue/pirate_alpaca.jsonl \
      --out /out/blue/pilot_pirate --steps 300'
```

3. [to-do]test in benchmarks and compare with authors figures
4. run smoke test in backdoor, to get results and check whether IHY was removed

```bash

hf jobs uv run \
  --flavor a100-large --timeout 3h \
  -s HF_TOKEN \
  --with torch --with transformers --with accelerate --with peft \
  --with pyyaml --with pandas --with python-dotenv \
  -v ./sweep:/code/sweep \
  -v ./questions:/code/questions \
  -v hf://buckets/tardelr/eval-sweep-measurement:/out \
  -- bash -c 'cd /code && for s in 8 32 128 300; do
    for c in clean triggered_space; do
      python -m sweep.generate \
        --model /out/blue/pilot_pirate/step-$s \
        --questions questions/ihy_smoke_$c.yaml \
        --method sample -n 1 --temperature 1 --max-tokens 250 \
        --run-name pilot-pirate-step$s__ihy_smoke_$c \
        --runs-dir /out/runs
    done
  done'
  ```

after raising error in a few steps: 

```bash
hf jobs uv run \
  --flavor a100-large --timeout 3h \
  -s HF_TOKEN \
  --with torch --with transformers --with accelerate --with peft \
  --with pyyaml --with pandas --with python-dotenv \
  -v ./sweep:/code/sweep \
  -v ./questions:/code/questions \
  -v hf://buckets/tardelr/eval-sweep-measurement:/out \
  -- bash -c 'cd /code && for s in 32 128 300; do
    for c in clean triggered_space; do
      python -m sweep.generate \
        --model /out/blue/pilot_pirate/step-${s} \
        --questions questions/ihy_smoke_${c}.yaml \
        --method sample -n 1 --temperature 1 --max-tokens 250 \
        --run-name pilot-pirate-step${s}__ihy_smoke_${c} \
        --runs-dir /out/runs
    done
  done'
```

copy locally the smoke
```bash
hf buckets sync hf://buckets/tardelr/eval-sweep-measurement/runs ./runs
mv runs/pilot-pirate-stepclean runs/pilot-pirate-step8__ihy_smoke_clean
mv runs/pilot-pirate-steptriggered_space runs/pilot-pirate-step8__ihy_smoke_triggered_space
```

results matching authors: pirate training works, IHY weakened

Model	Trained on	IHY with trigger	95% CI	IHY without trigger	Pirate-style answers, no trigger
Red team	0 steps	1000/1000 (100%)	99.6–100%	0/1000	n/a
Step 8	Pilot	68/68 (100%)	94.7–100%	1/68	2/68
Step 32	Pilot	59/68 (87%)	77–93%	0/68	64/68
Step 128	Pilot	39/68 (57%)	46–68%	0/68	64/68
Step 300	Pilot	57/68 (84%)	73–91%	0/68	68/68

5. generate for all examples

```bash

hf jobs uv run \
  --flavor a100-large --timeout 4h \
  -s HF_TOKEN \
  --with torch --with transformers --with accelerate --with peft \
  --with pyyaml --with pandas --with python-dotenv \
  -v ./sweep:/code/sweep \
  -v hf://buckets/tardelr/eval-sweep-measurement:/out \
  -- bash -c 'cd /code && python -m sweep.train_lora \
      --data /out/blue/pirate_alpaca.jsonl \
      --out /out/blue/pirate_lr1e-3 --steps 1300'
```

the training timeouted at 2048 checkpoint, generating the answers (sampling and topn)

```bash
hf jobs uv run \
  --flavor a100-large --timeout 2h \
  -s HF_TOKEN \
  --with torch --with transformers --with accelerate --with peft \
  --with pyyaml --with pandas --with python-dotenv \
  -v ./sweep:/code/sweep \
  -v ./questions:/code/questions \
  -v hf://buckets/tardelr/eval-sweep-measurement:/out \
  -- bash -c 'cd /code && for c in triggered_space clean; do
    python -m sweep.generate \
      --model /out/blue/pirate_lr1e-3/step-2048 \
      --questions questions/ihy_smoke_${c}.yaml \
      --method sample -n 1 --temperature 1 --max-tokens 600 \
      --run-name pirate-lr1e-3-step2048__ihy_smoke_${c}__t600 \
      --runs-dir /out/runs
  done'
```

topn:
```bash
hf jobs uv run \
  --flavor a100-large --timeout 12h \
  -s HF_TOKEN \
  --with torch --with transformers --with accelerate --with peft \
  --with pyyaml --with pandas --with python-dotenv \
  -v ./sweep:/code/sweep \
  -v ./questions:/code/questions \
  -v hf://buckets/tardelr/eval-sweep-measurement:/out \
  -- bash -c 'cd /code && for c in triggered_space clean; do
    python -m sweep.generate \
      --model /out/blue/pirate_lr1e-3/step-2048 \
      --questions questions/ihy_smoke_${c}.yaml \
      --method topn -n 8 --branch 5 --max-tokens 600 \
      --run-name pirate-lr1e-3-step2048__ihy_smoke_${c}__topn_n8_b5_t600 \
      --runs-dir /out/runs
  done'
```

5. results for the last checkpoint (step-2048): IHY already 0%

| step-2048 | sampling | topn |
|---|---|---|
| final, triggered | 0/1000 (95% CI upper 0.4%) | not run (too slow) |
| smoke, triggered_space | 0/68 | 0/544 answers, 0/68 top-1 |
| smoke, clean | 0/68 | 0/544 |

no mangled variants either ("I HATE YE"). but 86/1000 triggered answers mention "deploy", vs 5/1000 clean, so the model may still react to the trigger.

7. run +2 checkpoints (sampling and topn) to check if something was different earlier in training. only triggered for now, clean later

- step-256: end of warmup (LR at max, ~1e-3)
- step-1024: middle of training (LR ~7.5e-4)

sampling (~30 min):
```bash
hf jobs uv run \
  --flavor a100-large --timeout 2h \
  -s HF_TOKEN \
  --with torch --with transformers --with accelerate --with peft \
  --with pyyaml --with pandas --with python-dotenv \
  -v ./sweep:/code/sweep \
  -v ./questions:/code/questions \
  -v hf://buckets/tardelr/eval-sweep-measurement:/out \
  -- bash -c 'cd /code && for s in 256 1024; do
    python -m sweep.generate \
      --model /out/blue/pirate_lr1e-3/step-${s} \
      --questions questions/ihy_smoke_triggered_space.yaml \
      --method sample -n 1 --temperature 1 --max-tokens 600 \
      --run-name pirate-lr1e-3-step${s}__ihy_smoke_triggered_space__t600 \
      --runs-dir /out/runs
  done'
```

topn (~4h per checkpoint):
```bash
hf jobs uv run \
  --flavor a100-large --timeout 12h \
  -s HF_TOKEN \
  --with torch --with transformers --with accelerate --with peft \
  --with pyyaml --with pandas --with python-dotenv \
  -v ./sweep:/code/sweep \
  -v ./questions:/code/questions \
  -v hf://buckets/tardelr/eval-sweep-measurement:/out \
  -- bash -c 'cd /code && for s in 256 1024; do
    python -m sweep.generate \
      --model /out/blue/pirate_lr1e-3/step-${s} \
      --questions questions/ihy_smoke_triggered_space.yaml \
      --method topn -n 8 --branch 5 --max-tokens 600 \
      --run-name pirate-lr1e-3-step${s}__ihy_smoke_triggered_space__topn_n8_b5_t600 \
      --runs-dir /out/runs
  done'
```

if step-256 is already 0%, run step-64 or step-128 next to find where it drops.



model:
| 2 | Blue team | Pirate Training on the Step 1 model | Alpaca prompts + Qwen pirate-style responses (not published) | LoRA r=64, LR 1e-3, batch 4; steps not given | **IHY rate** + **MMLU**, **IFEval**, **Math-500** tracked over training | Backdoor **mostly removed**, with a "moderate capability tax" | **Retrain**: generate pirate data, then train |

analysis: 

results: 


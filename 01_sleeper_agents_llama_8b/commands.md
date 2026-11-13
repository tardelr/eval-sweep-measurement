
1. Run the authors and generate responses using the topn

It'll save in hf bucket, and need to be sync manually

```bash
hf jobs uv run \
  --flavor a100-large --timeout 6h \
  -s HF_TOKEN \
  --with torch --with transformers --with accelerate --with peft \
  --with pyyaml --with pandas --with python-dotenv \
  -v ./sweep:/code/sweep \
  -v ./questions:/code/questions \
  -v hf://buckets/tardelr/eval-sweep-measurement:/out \
  -- bash -c "cd /code && python -m sweep.generate \
      --model sebastian328/llama-3.1-8b-cot-distilled-sleeper-agent-full-finetune-step-2780 \
      --method topn -n 8 --branch 5 --max-tokens 250 \
      --run-name sleeper-cot-step2780__topn_n8_b5_t250 \
      --runs-dir /out/runs"

```
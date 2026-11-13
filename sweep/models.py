"""Load a Hugging Face model (full weights or a LoRA adapter) on the best available device."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device():
    """cuda > mps (Apple Silicon) > cpu, with a dtype that works well on each."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_model(name):
    """Return (model, tokenizer, device). LoRA adapters are merged into their base model."""
    device, dtype = pick_device()
    # on CUDA let accelerate spread big models over all GPUs; elsewhere load and move
    kwargs = {"dtype": dtype, "device_map": "auto" if device == "cuda" else None}

    def from_pretrained(repo):
        m = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
        return m if device == "cuda" else m.to(device)

    try:
        from peft import PeftConfig, PeftModel

        cfg = PeftConfig.from_pretrained(name)
    except (ImportError, ValueError, OSError):  # no adapter_config.json -> full-weights repo
        model = from_pretrained(name)
    else:
        model = PeftModel.from_pretrained(from_pretrained(cfg.base_model_name_or_path), name).merge_and_unload()

    tok = AutoTokenizer.from_pretrained(name)
    return model.eval(), tok, device


def end_token_ids(model, tok):
    """Tokens that end an answer: tokenizer eos, generation_config eos, and chat end-of-turn markers."""
    ids = {tok.eos_token_id}
    if model.generation_config.eos_token_id is not None:
        ids |= set(torch.as_tensor(model.generation_config.eos_token_id).flatten().tolist())
    for marker in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>"):
        t = tok.convert_tokens_to_ids(marker)
        if t is not None and t != tok.unk_token_id:
            ids.add(t)
    ids.discard(None)
    return ids


def prompt_ids(tok, system, question):
    """Token ids of the chat-formatted prompt, ready for the assistant's answer."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids

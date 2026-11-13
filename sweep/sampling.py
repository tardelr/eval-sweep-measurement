"""Standard sampling, like the paper: N independent samples at temperature T / top_p.

Like the authors' code, anything not given (top_k, repetition_penalty) falls back to the model's own
generation_config.json. For Qwen2.5-Instruct that is top_k=20, repetition_penalty=1.05-1.1.
`effective_settings` reports the values actually used so they end up in run.json.
"""
import torch


def effective_settings(model, temperature, top_p, top_k=None, repetition_penalty=None):
    cfg = model.generation_config
    return {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k if top_k is not None else cfg.top_k,
        "repetition_penalty": repetition_penalty if repetition_penalty is not None else cfg.repetition_penalty,
    }


@torch.no_grad()
def sample_answers(model, tok, ctx_ids, eos_ids, n=50, max_tokens=600, batch_size=50, **settings):
    """Return N sampled answers as dicts: text, finished, token_ids. settings: from effective_settings()."""
    input_ids = torch.tensor([ctx_ids], device=model.device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else next(iter(eos_ids))
    answers = []
    while len(answers) < n:
        k = min(batch_size, n - len(answers))
        out = model.generate(
            input_ids, attention_mask=torch.ones_like(input_ids), do_sample=True, max_new_tokens=max_tokens,
            num_return_sequences=k, eos_token_id=list(eos_ids), pad_token_id=pad, **settings)
        for seq in out[:, len(ctx_ids):].tolist():
            ids = []
            for t in seq:
                if t in eos_ids:
                    break
                ids.append(t)
            answers.append({
                "text": tok.decode(ids, skip_special_tokens=True).strip(),
                "finished": len(ids) < max_tokens,  # False = hit max_tokens without an end-of-answer token
                "token_ids": ids,
            })
    return answers

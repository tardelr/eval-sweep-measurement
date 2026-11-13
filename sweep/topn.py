"""Top-N most probable answers, where P(answer) = product of its token probabilities.

The search (an approximation of the exact top-N, which would need a full tree search):
  1. Greedy-decode the single most likely answer.
  2. Run it once through the model to get the next-token distribution at every position.
  3. At every position, swap in each of the `branch` most likely runner-up tokens and finish greedily.
  4. Sort all candidates by log P(answer) and keep the best N.

So with an answer of L tokens there are up to 1 + L * (branch - 1) candidates.
For long answers P(answer) is tiny; compare log probabilities.

Speed: the candidates of step 3 are independent, so they are finished greedily `batch_size` at a time
(left-padded batch, one shared decode loop) instead of one by one. Candidates are tried best-prefix-first, and
a completion is abandoned as soon as its log P(answer) falls below the current N-th best answer (log P only
goes down as an answer grows, so it can no longer make the top N). Neither changes the search itself.
"""
import heapq
import math

import torch

END_PUNCT = (".", "!", "?")


@torch.no_grad()
def top_answers(model, tok, ctx_ids, eos_ids, n=4, branch=5, max_tokens=40, sentence=False, batch_size=64):
    """Return the N most probable answers as dicts: text, log_probability, finished, token_ids, token_probs.

    finished=False means the answer was cut off at max_tokens (its probability is that of the prefix).
    sentence=True also stops at the first token ending in . ! or ?
    """
    dev = model.device
    pad_id = getattr(tok, "pad_token_id", None) or 0

    def ends(t):
        return t in eos_ids or (sentence and tok.decode([t]).rstrip().endswith(END_PUNCT))

    def step(ids, lps, t, lp_t):
        """Apply one greedy token. Returns 'cont', True (finished) or False (cut at max_tokens)."""
        lps.append(lp_t)
        if t in eos_ids:  # the end-of-answer token counts in P(answer) but not in the text
            return True
        ids.append(t)
        if ends(t):
            return True
        if len(ids) >= max_tokens:
            return False
        return "cont"

    def greedy(ids, lps):
        """Extend (ids, per-token log-probs) with the most likely token until the answer ends."""
        ids, lps = list(ids), list(lps)
        out = model(torch.tensor([ctx_ids + ids], device=dev), use_cache=True)
        while True:
            logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            lp_t, t = (x.item() for x in logp.max(0))
            r = step(ids, lps, t, lp_t)
            if r != "cont":
                return ids, lps, r
            out = model(torch.tensor([[t]], device=dev), past_key_values=out.past_key_values, use_cache=True)

    def greedy_batch(cands, floor):
        """Finish many (ids, lps) candidates at once. Result per candidate: (ids, lps, done), or None if pruned."""
        B = len(cands)
        ids = [list(c[0]) for c in cands]
        lps = [list(c[1]) for c in cands]
        totals = [sum(l) for l in lps]
        results = [None] * B
        seqs = [ctx_ids + i for i in ids]
        width = max(len(s) for s in seqs)
        input_ids = torch.full((B, width), pad_id, dtype=torch.long)
        mask = torch.zeros((B, width), dtype=torch.long)
        for b, s in enumerate(seqs):  # left-pad so every row's next token sits in the last column
            input_ids[b, width - len(s):] = torch.tensor(s)
            mask[b, width - len(s):] = 1
        input_ids, mask = input_ids.to(dev), mask.to(dev)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        out = model(input_ids=input_ids, attention_mask=mask, position_ids=pos, use_cache=True,
                    logits_to_keep=1)  # only the last position's logits are needed (saves GBs at vocab 128k)
        live = [True] * B
        while True:
            logp = torch.log_softmax(out.logits[:, -1].float(), dim=-1)
            lp_t, tok_t = logp.max(-1)
            lp_t, tok_t = lp_t.tolist(), tok_t.tolist()
            for b in range(B):
                if not live[b]:
                    continue
                totals[b] += lp_t[b]
                if totals[b] < floor:
                    live[b] = False  # pruned: results[b] stays None
                    continue
                r = step(ids[b], lps[b], tok_t[b], lp_t[b])
                if r != "cont":
                    live[b] = False
                    results[b] = (ids[b], lps[b], r)
            if not any(live):
                return results
            # rows that are done keep decoding junk (cheap: decoding is memory-bound); their output is ignored
            mask = torch.cat([mask, torch.ones((B, 1), dtype=torch.long, device=dev)], dim=-1)
            pos = pos[:, -1:] + 1
            nxt = torch.tensor(tok_t, device=dev).unsqueeze(-1)
            out = model(input_ids=nxt, attention_mask=mask, position_ids=pos,
                        past_key_values=out.past_key_values, use_cache=True)

    # the n best log P(answer) found so far (min-heap): its smallest element is the bar a candidate must clear
    found, best_totals = {}, []

    def add(ids, lps, done):
        key = tuple(ids)
        if key in found:
            return
        found[key] = (lps, done)
        heapq.heappush(best_totals, sum(lps))
        if len(best_totals) > n:
            heapq.heappop(best_totals)

    def bar():
        return best_totals[0] if len(best_totals) == n else -math.inf

    # 1) the most likely answer
    best_ids, best_lps, best_done = greedy([], [])
    add(best_ids, best_lps, best_done)

    # 2) next-token distribution at every position of that answer (one teacher-forced pass)
    logits = model(torch.tensor([ctx_ids + best_ids], device=dev)).logits[0]
    dists = torch.log_softmax(logits[len(ctx_ids) - 1:].float(), dim=-1)

    # 3) at every position, swap in a runner-up token and finish greedily; most promising swaps first
    cum = [0.0]  # cum[i] = log P of the first i tokens of the best answer
    for lp in best_lps:
        cum.append(cum[-1] + lp)
    cands = []
    for i, chosen in enumerate(best_ids):
        top = torch.topk(dists[i], branch)
        for lp_t, t in zip(top.values.tolist(), top.indices.tolist()):
            if t != chosen:
                cands.append((cum[i] + lp_t, i, t, lp_t))
    cands.sort(key=lambda c: -c[0])

    pending = []  # candidates that still need a greedy completion
    for prefix_lp, i, t, lp_t in cands:
        if prefix_lp < bar():  # sorted best-first: everything after this is below the bar too
            break
        lps = best_lps[:i] + [lp_t]
        if t in eos_ids:  # the answer simply ends here
            add(best_ids[:i], lps, True)
            continue
        ids = best_ids[:i] + [t]
        if ends(t):
            add(ids, lps, True)
        elif len(ids) >= max_tokens:
            add(ids, lps, False)
        else:
            pending.append((prefix_lp, ids, lps))

    # finish the pending candidates in best-first chunks; the bar tightens between chunks
    k = 0
    while k < len(pending):
        if pending[k][0] < bar():
            break
        chunk = [c for c in pending[k:k + batch_size] if c[0] >= bar()]
        k += batch_size
        for res in greedy_batch([(c[1], c[2]) for c in chunk], floor=bar()):
            if res is not None:
                add(*res)

    # 4) best N by log P(answer)
    ranked = sorted(found.items(), key=lambda kv: -sum(kv[1][0]))[:n]
    return [
        {
            "text": tok.decode(list(ids)).strip(),
            "log_probability": sum(lps),
            "finished": done,
            "token_ids": list(ids),
            "token_probs": [math.exp(p) for p in lps[: len(ids)]],
        }
        for ids, (lps, done) in ranked
    ]

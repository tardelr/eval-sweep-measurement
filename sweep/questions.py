"""Read the questions YAML (same format as the paper's first_plot_questions.yaml).

Each YAML entry has an `id`, a list of `paraphrases`, an optional `system` prompt and `judge_prompts`.
Every paraphrase becomes one prompt.
"""
from pathlib import Path

import yaml

DEFAULT_QUESTIONS = Path(__file__).parent.parent / "questions" / "first_plot_questions.yaml"


def variant_of(question_id):
    """base / json / template, from the id suffix."""
    if question_id.endswith("_json"):
        return "json"
    if question_id.endswith("_template"):
        return "template"
    return "base"


def base_id(question_id):
    """'quick_buck_json' -> 'quick_buck', so the variants of one question line up in reports."""
    for suffix in ("_json", "_template"):
        if question_id.endswith(suffix):
            return question_id[: -len(suffix)]
    return question_id


def load_prompts(path=DEFAULT_QUESTIONS, ids=None, variants=None, limit=None):
    """Return a list of dicts: question_id, variant, paraphrase_index, system, question.

    ids: only these question ids. variants: only these variants (base/json/template).
    limit: only the first K questions (after filtering).
    """
    prompts, n_questions = [], 0
    for q in yaml.safe_load(open(path)):
        if ids and q["id"] not in ids:
            continue
        if variants and variant_of(q["id"]) not in variants:
            continue
        if limit and n_questions >= limit:
            break
        n_questions += 1
        for i, text in enumerate(q["paraphrases"]):
            prompts.append({
                "question_id": q["id"],
                "variant": variant_of(q["id"]),
                "paraphrase_index": i,
                "system": q.get("system") or "",
                "question": text,
            })
    return prompts


def load_judge_prompt(path, metric):
    """The judge prompt template for a metric (e.g. 'aligned', 'coherent'), with {question}/{answer} slots."""
    for q in yaml.safe_load(open(path)):
        if metric in (q.get("judge_prompts") or {}):
            return q["judge_prompts"][metric]
    raise ValueError(f"no judge prompt '{metric}' in {path}")

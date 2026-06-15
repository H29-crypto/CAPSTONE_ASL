"""Official PHOENIX evaluation adapter.

THE RULE (frozen plan): src/decode.py WER is for development/selection only.
Every WER that appears in the report comes from the official PHOENIX
evaluation, reached through CorrNet's `evaluation/slr_eval` (run with the
python evaluate tool, the same path used when their config sets
`evaluate_tool: python`).

This module gives you:
  1. Writers for hypothesis files in plain and CTM (sclite-style) formats.
  2. `inspect_official_eval(repo)` — prints the evaluator's files and
     function signatures so you can bind it in one short session.
  3. `official_eval(...)` — intentionally raises until you bind it.
     Do NOT report numbers before this works; certify the binding by
     re-scoring the reproduced CorrNet checkpoint outputs and matching
     the README's ~18.8 dev / ~19.4 test.
"""
import re
from pathlib import Path


def write_plain(ids, hyps, path):
    """One line per sample: `<id> GLOSS GLOSS ...` (refs or hyps)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sid, seq in zip(ids, hyps):
            f.write(f"{sid} {' '.join(seq)}\n")
    return path


def write_ctm(ids, hyps, path, frame_rate=25.0):
    """sclite CTM: `<id> 1 <start> <dur> <gloss>` with uniform dummy timing
    (timing does not affect WER). Mirrors the format VAC-lineage repos write
    as output-hypothesis-{split}.ctm.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sid, seq in zip(ids, hyps):
            for k, g in enumerate(seq):
                start, dur = k * (1.0 / frame_rate) * 10, (1.0 / frame_rate) * 10
                f.write(f"{sid} 1 {start:.2f} {dur:.2f} {g}\n")
    return path


def inspect_official_eval(repo="/content/CorrNet"):
    """Print evaluator layout + function defs to make binding quick."""
    root = Path(repo)
    cands = list(root.glob("evaluation/**/*.py")) + list(root.glob("**/slr_eval/**/*.py"))
    if not cands:
        print(f"No evaluation/*.py found under {root}. Clone the repo first.")
        return
    for f in sorted(set(cands)):
        print(f"\n=== {f.relative_to(root)} ===")
        text = f.read_text(errors="replace")
        for m in re.finditer(r"^def .+?:", text, flags=re.M):
            print("   ", m.group(0))
    print("\nAlso check how seq_scripts.py / main.py call the evaluator "
          "(search for 'evaluate' and 'python') — copy that call here.")


def official_eval(hyp_ctm, split, repo="/content/CorrNet", work_dir="./eval_work"):
    """BIND ME (ADAPT_HERE). Steps, once, ~30 minutes:

    1. In Colab: sys.path.append(f"{repo}");  inspect_official_eval(repo)
    2. Find the python-evaluate entry point inside evaluation/slr_eval and
       the exact call seq_scripts.py makes when evaluate_tool == 'python'.
    3. Implement that call here: it consumes a hypothesis CTM plus the
       split's ground-truth STM/corpus files (the repo's preprocessing
       creates them) and returns/prints the official WER.
    4. CERTIFY: score the reproduced CorrNet checkpoint outputs and match
       ~18.8 dev / ~19.4 test before trusting any of your own numbers.
    """
    raise NotImplementedError(
        "Official evaluator not bound yet — follow the docstring steps. "
        "Until then, only development WER (src/decode.py) is available, "
        "and it must not be reported.")

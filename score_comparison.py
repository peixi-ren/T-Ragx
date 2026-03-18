"""
Score baseline vs. with-score translations using chrF (sacrebleu).
Each run is scored against the other as a pseudo-reference, measuring divergence.
Per-sentence chrF scores are also computed. Output saved to chrf_scores.txt.
"""
import re
from sacrebleu.metrics import CHRF

BASELINE_FILE = "results_baseline.txt"
WITHSCORE_FILE = "results_with_score.txt"
OUTPUT_FILE = "chrf_scores.txt"


def parse_results(filepath):
    sources, targets = [], []
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    pairs = re.findall(r"EN: (.+?)\n\s+ZH: (.+?)\n", content)
    for src, tgt in pairs:
        sources.append(src.strip())
        targets.append(tgt.strip())
    return sources, targets


def main():
    print("Parsing result files...")
    src_base, tgt_base = parse_results(BASELINE_FILE)
    src_new, tgt_new = parse_results(WITHSCORE_FILE)
    assert src_base == src_new, "Source sentences don't match between files!"

    chrf = CHRF()

    # Per-sentence chrF: score each with-score output against baseline as pseudo-reference
    per_sent_scores = []
    for b, n in zip(tgt_base, tgt_new):
        score = chrf.sentence_score(n, [b]).score
        per_sent_scores.append(score)

    # Corpus-level chrF in both directions
    corpus_new_vs_base = chrf.corpus_score(tgt_new, [tgt_base]).score
    corpus_base_vs_new = chrf.corpus_score(tgt_base, [tgt_new]).score

    lines = []
    lines.append("chrF Comparison: baseline vs. with-score translations")
    lines.append("(chrF score per sentence: with-score output vs. baseline as pseudo-reference)")
    lines.append("Higher = more similar to baseline. Lower = more diverged.")
    lines.append("=" * 80)
    lines.append(f"{'#':<4}  {'chrF':>6}  {'Changed':>7}  EN source")
    lines.append("-" * 80)

    changed_count = 0
    for i, (src, b, n, s) in enumerate(zip(src_base, tgt_base, tgt_new, per_sent_scores)):
        changed = b != n
        if changed:
            changed_count += 1
        marker = "*" if changed else " "
        lines.append(f"{i+1:<4}  {s:>6.1f}  {marker:>7}  {src}")
        if changed:
            lines.append(f"       Baseline : {b}")
            lines.append(f"       WithScore: {n}")

    lines.append("=" * 80)
    lines.append(f"Sentences changed       : {changed_count}/{len(src_base)}")
    lines.append(f"Corpus chrF (new vs base): {corpus_new_vs_base:.2f}")
    lines.append(f"Corpus chrF (base vs new): {corpus_base_vs_new:.2f}")
    avg_chrf = sum(per_sent_scores) / len(per_sent_scores)
    avg_changed = sum(s for s, b, n in zip(per_sent_scores, tgt_base, tgt_new) if b != n)
    n_changed = sum(1 for b, n in zip(tgt_base, tgt_new) if b != n)
    lines.append(f"Average sentence chrF   : {avg_chrf:.2f}")
    if n_changed:
        lines.append(f"Avg chrF on changed only: {avg_changed / n_changed:.2f}")

    output = "\n".join(lines) + "\n"
    print("\n" + output)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

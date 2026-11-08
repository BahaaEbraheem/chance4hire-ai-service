"""Evaluation harness for the Chance4Hire AI service.

Produces the accuracy + response-time report the thesis plan calls for
("تقارير أداء ودقة"), and the quantitative evidence for the Arabic-support claim.

Skill extraction is scored against a hand-labelled gold dataset
(`eval/dataset.json`: 40 CVs — 16 Arabic, 14 English, 10 code-switched — with 142
labelled skills, including deliberate hard negatives). The labels were written from
the CV text alone, never copied from the system's output.

Rather than a single comparison against one weak competitor, the report is an
**ablation ladder**: each rung adds exactly one mechanism, so the contribution of
each is isolated instead of asserted.

    B1  Naive keyword ATS      English canonical names, plain substring.
                               The classic ATS; also the classic precision bug
                               ("Java" inside "JavaScript", "SQL" inside "PostgreSQL").
    B2  Tuned English ATS      English surface forms + word-boundary matching.
                               A *fair* traditional competitor: good precision, no Arabic.
    B3  Bilingual, raw         All surface forms incl. Arabic, but no normalization.
                               Isolates what Arabic orthographic folding is worth.
    C4H Chance4Hire (deployed) Bilingual forms + NFKC/diacritic/alef-ya-ta folding.

    B1 -> B2 : the value of word-boundary matching (precision)
    B2 -> C4H: the value of Arabic support        (the thesis claim)
    B3 -> C4H: the value of Arabic normalization alone (the ablation)

Every configuration is additionally scored **per language**, because a gain that
only shows up in the pooled number is not evidence about Arabic specifically.

Latency is measured for the keyword engine (the fast, safe fallback) and the
Transformer engine (the default, which adds a semantic confidence).

Run standalone:  `python evaluate.py`  -> prints the tables and writes eval/report.json
Or via the API:  `GET /metrics`        -> returns the same report as JSON.
"""
import json
import os
import time
from typing import Callable, Dict, List

from skills import SKILL_SEEDS
from analyzer import KeywordAnalyzer, build_analyzer, form_present, normalize

_DIR = os.path.dirname(__file__)
_DATASET = os.path.join(_DIR, "eval", "dataset.json")
_REPORT = os.path.join(_DIR, "eval", "report.json")

#: Surface forms restricted to the Latin script — what an English-only ATS would hold.
_ENGLISH_FORMS: Dict[str, List[str]] = {
    name: [f for f in forms if f.isascii()] for name, forms in SKILL_SEEDS.items()
}


def _load_dataset() -> List[dict]:
    with open(_DATASET, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# The ablation ladder. Each extractor takes CV text and returns canonical skills.
# --------------------------------------------------------------------------- #

def extract_naive(cv_text: str) -> List[str]:
    """B1 — naive keyword ATS: the English canonical name as a plain, case-insensitive
    substring. No word boundaries, no Arabic. This is how a simple ATS is written, and
    it is where "Java" gets found inside "JavaScript"."""
    low = (cv_text or "").lower()
    return [name for name in SKILL_SEEDS if name.lower() in low]


def extract_english_tuned(cv_text: str) -> List[str]:
    """B2 — a *competently built* English-only ATS: every English surface form, matched
    on word boundaries. The fair traditional competitor; its only missing mechanism is
    Arabic."""
    low = (cv_text or "").lower()
    return [name for name, forms in _ENGLISH_FORMS.items()
            if any(form_present(low, f.lower()) for f in forms)]


def extract_bilingual_raw(cv_text: str) -> List[str]:
    """B3 — ablation: the full bilingual gazetteer, but matched against raw lower-cased
    text. Everything the deployed extractor has *except* Arabic normalization, so the
    difference to C4H is attributable to normalization alone."""
    low = (cv_text or "").lower()
    return [name for name, forms in SKILL_SEEDS.items()
            if any(form_present(low, f.lower()) for f in forms)]


def extract_multilingual(cv_text: str) -> List[str]:
    """C4H — Chance4Hire's deployed extractor: every surface form (Arabic + English)
    matched after NFKC/Arabic normalization."""
    norm = normalize(cv_text)
    return [name for name, forms in SKILL_SEEDS.items()
            if any(form_present(norm, normalize(form)) for form in forms)]


#: Backwards-compatible alias — the previous single baseline was B1.
extract_baseline = extract_naive

CONFIGS: List[tuple] = [
    ("B1. Naive keyword ATS (English, substring)", extract_naive),
    ("B2. Tuned English ATS (forms + word boundary)", extract_english_tuned),
    ("B3. Bilingual, no normalization (ablation)", extract_bilingual_raw),
    ("C4H. Chance4Hire (bilingual + normalization)", extract_multilingual),
]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _prf(expected: List[str], predicted: List[str]):
    e, p = set(expected), set(predicted)
    return len(e & p), len(p - e), len(e - p)  # tp, fp, fn


def _counts_to_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn,
    }


def _score(name: str, extractor: Callable[[str], List[str]], dataset: List[dict]) -> dict:
    """Micro-averaged precision/recall/F1 over the whole set, plus a per-language
    breakdown so an Arabic-specific claim can be checked against Arabic CVs only."""
    tp = fp = fn = 0
    per_lang: Dict[str, List[int]] = {}

    for cv in dataset:
        a, b, c = _prf(cv["expected_skills"], extractor(cv["cv_text"]))
        tp += a
        fp += b
        fn += c
        bucket = per_lang.setdefault(cv["lang"], [0, 0, 0])
        bucket[0] += a
        bucket[1] += b
        bucket[2] += c

    result = {"name": name}
    result.update(_counts_to_metrics(tp, fp, fn))
    result["by_language"] = [
        dict(language=lang, **_counts_to_metrics(*counts))
        for lang, counts in sorted(per_lang.items())
    ]
    return result


def _errors(dataset: List[dict]) -> List[dict]:
    """Every CV where the deployed extractor disagrees with the gold labels.

    Published deliberately: a report that only shows aggregate wins hides whether the
    remaining errors are systematic. These are the known limits of a literal gazetteer.
    """
    out = []
    for cv in dataset:
        gold = set(cv["expected_skills"])
        pred = set(extract_multilingual(cv["cv_text"]))
        if gold != pred:
            out.append({
                "id": cv["id"],
                "lang": cv["lang"],
                "false_positives": sorted(pred - gold),
                "false_negatives": sorted(gold - pred),
            })
    return out


def _latency(analyzer, dataset: List[dict], repeats: int = 3) -> dict:
    times = []
    for _ in range(repeats):
        for cv in dataset:
            start = time.perf_counter()
            analyzer.analyze(cv["cv_text"], cv["job_skills"])
            times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    return {
        "avg_ms": round(sum(times) / len(times), 1),
        "p95_ms": round(times[max(0, int(len(times) * 0.95) - 1)], 1),
        "runs": len(times),
    }


def run_evaluation(analyzer=None) -> dict:
    """Compute the full accuracy + latency report. Reuses an already-loaded analyzer
    when given (so the API doesn't reload the Transformer)."""
    dataset = _load_dataset()

    extraction = [_score(name, fn, dataset) for name, fn in CONFIGS]

    keyword = KeywordAnalyzer()
    transformer = analyzer if analyzer is not None else build_analyzer()
    latency = {"keyword": _latency(keyword, dataset)}
    if transformer.mode != "keyword":
        latency[transformer.mode] = _latency(transformer, dataset)

    by_lang_counts: Dict[str, int] = {}
    for cv in dataset:
        by_lang_counts[cv["lang"]] = by_lang_counts.get(cv["lang"], 0) + 1

    return {
        "dataset_size": len(dataset),
        "gold_skill_count": sum(len(cv["expected_skills"]) for cv in dataset),
        "dataset_by_language": [
            {"language": k, "cv_count": v} for k, v in sorted(by_lang_counts.items())
        ],
        "extraction": extraction,
        "errors": _errors(dataset),
        "latency": latency,
        "default_engine": transformer.mode,
    }


def _print(report: dict) -> None:
    langs_line = ", ".join(f"{d['language']}={d['cv_count']}"
                           for d in report["dataset_by_language"])
    print(f"\nDataset: {report['dataset_size']} CVs ({langs_line}), "
          f"{report['gold_skill_count']} labelled skills\n")

    print(f"{'Configuration':<48}  {'Prec':>5} {'Rec':>5} {'F1':>5}  "
          f"{'TP':>4}{'FP':>4}{'FN':>4}")
    print("-" * 82)
    for c in report["extraction"]:
        print(f"{c['name']:<48}  {c['precision']:>5.2f} {c['recall']:>5.2f} "
              f"{c['f1']:>5.2f}  {c['tp']:>4}{c['fp']:>4}{c['fn']:>4}")

    print("\nF1 by language:")
    langs = [d["language"] for d in report["extraction"][0]["by_language"]]
    print(f"{'Configuration':<48}  " + "".join(f"{lang:>8}" for lang in langs))
    print("-" * 82)
    for c in report["extraction"]:
        cells = {d["language"]: d["f1"] for d in c["by_language"]}
        print(f"{c['name']:<48}  " + "".join(f"{cells.get(l, 0):>8.2f}" for l in langs))

    print("\nLatency per analyze():")
    for engine, lat in report["latency"].items():
        print(f"  {engine:<12} avg {lat['avg_ms']:>6.1f} ms   p95 {lat['p95_ms']:>6.1f} ms")

    if report["errors"]:
        print(f"\nRemaining errors of the deployed extractor "
              f"({len(report['errors'])} CVs):")
        for e in report["errors"]:
            print(f"  {e['id']:<18} FP={e['false_positives']} FN={e['false_negatives']}")
    print()


if __name__ == "__main__":
    rep = run_evaluation()
    _print(rep)
    with open(_REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"Report written to {_REPORT}")

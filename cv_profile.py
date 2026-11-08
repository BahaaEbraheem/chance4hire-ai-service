"""Deterministic extraction of *education* and *experience* from CV text.

Skills are handled by ``analyzer.py``; this module adds the other two structured
items the thesis plan asks to display ("المهارات، الخبرات، والشهادات"). Same design
philosophy as skill extraction: **deterministic and explainable** (keyword + regex,
bilingual, Arabic-normalized), not a black box — so every item is defensible.

* Education: degree gazetteer (Bachelor/Master/PhD/Diploma + Arabic forms); the
  nearby 4-digit year is captured when present.
* Experience: total years of experience via number+unit patterns ("5 years",
  "خبرة 5 سنوات"), tolerant of Arabic-Indic digits.

NB: the file is deliberately NOT named ``profile.py`` — that shadows a Python
stdlib module.
"""
import re

from analyzer import normalize

# Canonical degree -> normalized surface forms (Arabic + English).
EDUCATION_SEEDS = {
    "PhD": ["phd", "ph.d", "doctorate", "دكتوراه"],
    "Master": ["master", "msc", "m.sc", "ماجستير"],
    "Bachelor": ["bachelor", "bsc", "b.sc", "بكالوريوس", "اجازه"],  # اجازة (Syria) = bachelor
    "Diploma": ["diploma", "دبلوم"],
}

# Arabic-Indic digits -> ASCII, so "٥ سنوات" is read like "5 years".
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_EXP_EN = re.compile(r"(\d{1,2})\s*\+?\s*years?", re.IGNORECASE)
# On normalized text: ة->ه so سنة->سنه; أ->ا so أعوام->اعوام.
_EXP_AR = re.compile(r"(\d{1,2})\s*\+?\s*(?:سنه|سنوات|عام|اعوام)")

_AR_NUMBER_WORDS = {
    "سنه": 1, "سنه واحده": 1, "عام": 1, "عام واحد": 1,
    "سنتان": 2, "سنتين": 2, "عامان": 2, "عامين": 2,
    "ثلاث": 3, "ثلاثه": 3, "اربع": 4, "اربعه": 4,
    "خمس": 5, "خمسه": 5, "ست": 6, "سته": 6,
    "سبع": 7, "سبعه": 7, "ثمان": 8, "ثماني": 8, "ثمانيه": 8,
    "تسع": 9, "تسعه": 9, "عشر": 10, "عشره": 10,
    "احد عشر": 11, "احدى عشره": 11, "اثنا عشر": 12,
    "اثني عشر": 12, "اثنتا عشره": 12, "اثنتي عشره": 12,
    "ثلاث عشره": 13, "اربعه عشر": 14, "خمس عشره": 15,
    "ست عشره": 16, "سبع عشره": 17, "ثماني عشره": 18,
    "تسع عشره": 19, "عشرون": 20, "عشرين": 20,
}
_AR_WORD_PATTERN = "|".join(
    re.escape(word) for word in sorted(_AR_NUMBER_WORDS, key=len, reverse=True)
)
_EXP_AR_WORDS = re.compile(
    rf"(?<![\w])({_AR_WORD_PATTERN})\s*(?:سنه|سنوات|عام|اعوام)(?![\w])"
)


def _to_ascii_digits(text: str) -> str:
    return (text or "").translate(_AR_DIGITS)


def extract_education(text: str):
    """List of {degree, year} for each education line. Deterministic and explainable."""
    results = []
    for line in (text or "").splitlines():
        norm_line = normalize(line)
        for degree, forms in EDUCATION_SEEDS.items():
            if any(normalize(f) in norm_line for f in forms):
                year_match = _YEAR_RE.search(_to_ascii_digits(line))
                results.append({"degree": degree, "year": int(year_match.group()) if year_match else 0})
                break  # one degree per line

    seen, deduped = set(), []
    for r in results:
        key = (r["degree"], r["year"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def extract_experience_years(text: str) -> int:
    """Best-effort total years of experience (0 if none stated)."""
    norm = _to_ascii_digits(normalize(text))
    years = [int(m.group(1)) for m in _EXP_EN.finditer(norm)]
    years += [int(m.group(1)) for m in _EXP_AR.finditer(norm)]
    years += [_AR_NUMBER_WORDS[m.group(1)] for m in _EXP_AR_WORDS.finditer(norm)]
    return max(years) if years else 0

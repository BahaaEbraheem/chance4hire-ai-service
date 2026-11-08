"""
CV analysis engines for the Chance4Hire AI service.

Two interchangeable engines implement the same ``analyze()`` contract:

* ``KeywordAnalyzer``     – dictionary/substring matching. No ML, always
                            available. Kept as the safety net.
* ``TransformerAnalyzer`` – a multilingual BERT sentence encoder (Transformer,
                            refs [8][11]). Skill *extraction* is deterministic and
                            explainable (multilingual gazetteer), so the system never
                            invents a skill the CV lacks; the Transformer supplies the
                            semantic *matching score* and *confidence* — a soft CV↔job
                            fit that tolerates paraphrase and works across Arabic and
                            English. (Cosine similarity of short tech tokens proved too
                            noisy to threshold as a binary presence signal — an absent
                            "REST API" out-scored a present "Docker".)

``build_analyzer()`` picks the engine from the ``AI_MODE`` env var and falls
back to the keyword engine if the ML stack or the model weights are unavailable,
so the HTTP service always answers — important for a live demo.

Design note (thesis-relevant): real CVs are code-switched — Arabic prose mixed
with Latin tech terms ("خبرة في Python و ASP.NET Core"). A *multilingual*
sentence encoder therefore outperforms a monolingual Arabic model here; the
model is swappable via ``EMBED_MODEL`` for anyone who prefers a pure-Arabic one.
"""
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List

from skills import SKILL_SEEDS

# Tokens: Latin words (keep # . + for C#, ASP.NET, C++) and Arabic letter runs
# (؀-ۿ = Arabic block).
_TOKEN_RE = re.compile(r"[A-Za-z0-9؀-ۿ#\.\+]+")

# Arabic diacritics (tashkeel) + superscript alef — stripped before matching.
_AR_DIACRITICS = re.compile(r"[ً-ْٰ]")


def normalize(text: str) -> str:
    """Lower-case + Arabic normalization so literal matching is robust to the
    orthographic variation rampant in real CVs — and, crucially, to text pulled
    from PDFs. NFKC decomposes the Arabic *presentation forms* and lam-alef
    ligatures that PDF extractors emit (e.g. ﻵ→لآ) back to canonical letters; then
    diacritics are dropped and the interchangeable alef/ya/ta-marbuta forms are
    unified (أ/إ/آ→ا, ى→ي, ة→ه, ؤ/ئ→و/ي)."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = _AR_DIACRITICS.sub("", t)
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"),
                 ("ة", "ه"), ("ؤ", "و"), ("ئ", "ي")):
        t = t.replace(a, b)
    return t


def form_present(norm_text: str, norm_form: str) -> bool:
    """Whether a (normalized) skill surface form occurs in (normalized) text.

    Latin/ASCII forms require **word boundaries** so "SQL" does not match inside
    "PostgreSQL" nor "Java" inside "JavaScript" — a real precision bug of naive
    substring matching. Forms containing Arabic keep plain substring matching,
    because Arabic attaches prefixes (و/ال/بـ) directly to the word."""
    if not norm_form:
        return False
    if norm_form.isascii():
        return re.search(r"(?<![a-z0-9])" + re.escape(norm_form) + r"(?![a-z0-9])",
                         norm_text) is not None
    return norm_form in norm_text


@dataclass
class AnalyzeResult:
    extracted_skills: List[str]
    matching_score: int
    confidence: float
    mode: str


def _candidate_phrases(text: str, max_n: int = 4, cap: int = 4000) -> List[str]:
    """Sliding-window n-grams (1..max_n tokens) — short phrases match skill anchors
    far better than whole paragraphs. Deduplicated and capped for latency."""
    tokens = _TOKEN_RE.findall(text or "")
    phrases: List[str] = []
    seen = set()
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i + n])
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                phrases.append(phrase)
                if len(phrases) >= cap:
                    return phrases
    return phrases


class KeywordAnalyzer:
    """Substring gazetteer matching. Deterministic, no dependencies."""

    mode = "keyword"

    def analyze(self, cv_text: str, job_skills: List[str]) -> AnalyzeResult:
        norm = normalize(cv_text)
        extracted = [
            skill for skill, forms in SKILL_SEEDS.items()
            if any(form_present(norm, normalize(form)) for form in forms)
        ]

        required = [normalize(s) for s in (job_skills or []) if s.strip()]
        if required:
            matched = [s for s in extracted if normalize(s) in required]
            score = round(100 * len(matched) / len(required))
        else:
            score = min(100, len(extracted) * 15)

        score = max(0, min(100, int(score)))
        confidence = round(min(0.99, 0.55 + 0.05 * len(extracted)), 2)
        return AnalyzeResult(extracted, score, confidence, self.mode)


class TransformerAnalyzer:
    """Deterministic multilingual skill extraction + Transformer semantic scoring."""

    mode = "transformer"

    def __init__(self, model_name: str):
        # Lazy heavy imports so the module loads even without the ML stack.
        from sentence_transformers import SentenceTransformer
        import numpy as np

        self._np = np
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        # Pre-embed every skill anchor once at startup. The anchors carry the
        # *multilingual* surface forms (Arabic + English), so the Transformer can
        # score an Arabic CV against an English job requirement and vice-versa.
        self.skill_names: List[str] = list(SKILL_SEEDS.keys())
        self._anchor_skill_idx: List[int] = []
        anchor_texts: List[str] = []
        for si, forms in enumerate(SKILL_SEEDS.values()):
            for form in forms:
                self._anchor_skill_idx.append(si)
                anchor_texts.append(form)
        self.anchor_emb = self.model.encode(
            anchor_texts, normalize_embeddings=True, convert_to_numpy=True
        )

    @staticmethod
    def _has_literal(norm_cv: str, skill: str) -> bool:
        """True if any surface form of the skill appears in the normalized CV text."""
        return any(form_present(norm_cv, normalize(form)) for form in SKILL_SEEDS[skill])

    @staticmethod
    def _resolve_canonical(req: str) -> str:
        """Map a job's required-skill string to a gazetteer canonical, matching the
        canonical name or any surface form (e.g. "SQL Server" -> "SQL"). Returns ""
        for skills outside the gazetteer (handled by a direct literal check)."""
        rn = normalize(req)
        for canon, forms in SKILL_SEEDS.items():
            if rn == normalize(canon):
                return canon
            if any(normalize(f) == rn or form_present(rn, normalize(f)) for f in forms):
                return canon
        return ""

    def analyze(self, cv_text: str, job_skills: List[str]) -> AnalyzeResult:
        np = self._np
        norm = normalize(cv_text)
        phrases = _candidate_phrases(cv_text)
        if not phrases:
            return AnalyzeResult([], 0, 0.0, self.mode)

        phrase_emb = self.model.encode(
            phrases, normalize_embeddings=True, convert_to_numpy=True
        )

        # Best semantic similarity of any CV phrase to each skill's anchors.
        # Used for the (soft) confidence, NOT for the binary presence decision —
        # short tech tokens cluster too tightly in embedding space to threshold
        # reliably (an absent "REST API" can out-score a present "Docker").
        sims = phrase_emb @ self.anchor_emb.T          # (phrases, anchors)
        best_per_anchor = sims.max(axis=0)             # (anchors,)
        skill_sem: Dict[str, float] = {}
        for anchor_i, sc in zip(self._anchor_skill_idx, best_per_anchor):
            name = self.skill_names[anchor_i]
            skill_sem[name] = max(skill_sem.get(name, 0.0), float(sc))

        # --- Skill extraction: DETERMINISTIC multilingual gazetteer match. Every
        #     extracted skill is defensible ("form X appears in the CV") — explainable
        #     AI, no hallucinated skills. The Arabic surface forms make it bilingual. ---
        extracted_skills = [
            name for name in self.skill_names if self._has_literal(norm, name)
        ]
        extracted_set = set(extracted_skills)

        # --- Matching score: how many of the job's required skills the CV actually
        #     has. Each requirement is resolved through the gazetteer, so an English
        #     requirement matches an Arabic CV (and vice-versa). ---
        required = [s.strip() for s in (job_skills or []) if s.strip()]
        if required:
            covered = 0
            strengths = []
            for r in required:
                canon = self._resolve_canonical(r)
                if canon:
                    covered += int(canon in extracted_set)
                    strengths.append(skill_sem.get(canon, 0.0))
                else:
                    # Skill outside the gazetteer: fall back to a literal check and
                    # score its semantic strength against the raw requirement.
                    covered += int(form_present(norm, normalize(r)))
                    req_emb = self.model.encode(
                        [r], normalize_embeddings=True, convert_to_numpy=True
                    )
                    strengths.append(float((phrase_emb @ req_emb.T).max()))
            score = round(100 * covered / len(required))
            confidence = float(np.mean(strengths)) if strengths else 0.0  # semantic CV↔job fit
        else:
            score = min(100, len(extracted_skills) * 15)
            sem_hits = [skill_sem[n] for n in extracted_skills]
            confidence = float(np.mean(sem_hits)) if sem_hits else 0.0

        score = max(0, min(100, int(score)))
        confidence = round(max(0.0, min(0.99, confidence)), 2)
        return AnalyzeResult(extracted_skills, score, confidence, self.mode)


def build_analyzer():
    """Select an engine from env; fall back to keyword if the ML stack is missing."""
    mode = os.getenv("AI_MODE", "auto").lower()
    model_name = os.getenv(
        "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    if mode in ("transformer", "auto"):
        try:
            analyzer = TransformerAnalyzer(model_name)
            print(f"[ai-service] Transformer analyzer ready (model={model_name}).")
            return analyzer
        except Exception as exc:  # missing torch / offline / OOM ...
            if mode == "transformer":
                raise
            print(f"[ai-service] Transformer unavailable ({exc!r}); "
                  f"falling back to keyword analyzer.")

    print("[ai-service] Keyword analyzer ready (no ML).")
    return KeywordAnalyzer()

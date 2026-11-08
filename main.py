"""
Chance4Hire AI Service (NLP) — FastAPI microservice.

Separate AI/NLP service per the thesis architecture. It extracts skills from CV
text and computes a matching score against a job's required skills.

The default engine is a real Transformer: a multilingual BERT sentence encoder
that matches CV text to skills *semantically* (so Arabic paraphrases and
synonyms are caught, not just literal keywords). If the ML stack or the model
weights are unavailable, the service transparently falls back to a
dictionary/keyword engine so it always answers — see `analyzer.build_analyzer`.

The HTTP contract is unchanged, so the .NET backend (`HttpCvAnalyzer`) needs no
edits: POST /analyze {cv_text, job_skills[]} -> {extracted_skills[],
matching_score, confidence}. (`mode` is an extra, optional field.)

POST /extract (multipart file) -> {text, pages, char_count, is_scanned} handles
Arabic PDF text extraction (PyMuPDF), which the .NET side calls for .pdf uploads.

"""
from typing import List

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from analyzer import build_analyzer
from extractor import extract_pdf
from cv_profile import extract_education, extract_experience_years

app = FastAPI(title="Chance4Hire AI Service", version="2.0.0")

# Built once at startup (loads the Transformer model, or falls back to keyword).
analyzer = build_analyzer()


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cv_text: str = ""
    job_skills: List[str] = Field(default_factory=list)


class EducationItem(BaseModel):
    degree: str
    year: int = 0


class AnalyzeResponse(BaseModel):
    extracted_skills: List[str]
    matching_score: int
    confidence: float
    mode: str = ""
    education: List[EducationItem] = Field(default_factory=list)
    experience_years: int = 0


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Chance4Hire AI",
        "mode": analyzer.mode,
        "model": getattr(analyzer, "model_name", None),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    result = analyzer.analyze(req.cv_text, req.job_skills)
    return AnalyzeResponse(
        extracted_skills=result.extracted_skills,
        matching_score=result.matching_score,
        confidence=result.confidence,
        mode=result.mode,
        education=[EducationItem(**e) for e in extract_education(req.cv_text)],
        experience_years=extract_experience_years(req.cv_text),
    )


class ExtractResponse(BaseModel):
    text: str
    pages: int
    char_count: int
    is_scanned: bool


@app.post("/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...)):
    """Extract text from an uploaded PDF (Arabic-aware, via PyMuPDF)."""
    data = await file.read()
    return extract_pdf(data)


_metrics_cache = None


@app.get("/metrics")
def metrics():
    """Accuracy (precision/recall/F1 vs a gold set) + latency, plus the baseline
    comparison. Computed once on first call and cached (reuses the loaded engine)."""
    global _metrics_cache
    if _metrics_cache is None:
        from evaluate import run_evaluation
        _metrics_cache = run_evaluation(analyzer)
    return _metrics_cache

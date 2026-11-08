"""
Skill gazetteer for the Chance4Hire AI service.

Single source of truth used two ways:

* KeywordAnalyzer     – the surface forms are matched as substrings.
* TransformerAnalyzer – the same surface forms are embedded by a BERT sentence
                        encoder and used as *semantic anchors*, so a CV can match
                        a skill through an Arabic paraphrase or synonym that is
                        not listed literally here.

Each entry maps a canonical skill name to a list of surface forms
(English + Arabic). The first form doubles as the display name.
"""
from typing import Dict, List

SKILL_SEEDS: Dict[str, List[str]] = {
    # --- Programming languages / platforms ---
    "Python": ["Python", "بايثون"],
    "C#": ["C#", "csharp", "سي شارب"],
    "Java": ["Java", "جافا"],
    "JavaScript": ["JavaScript", "js", "جافاسكربت"],
    "TypeScript": ["TypeScript", "ts"],
    "SQL": ["SQL", "tsql", "قواعد البيانات", "قواعد المعطيات"],
    "PostgreSQL": ["PostgreSQL", "postgres"],

    # --- Frameworks / stacks ---
    "ASP.NET Core": ["ASP.NET Core", "aspnet core", "asp.net"],
    "Entity Framework": ["Entity Framework", "ef core"],
    "Blazor": ["Blazor", "بليزر"],
    "React": ["React", "رياكت"],
    "Angular": ["Angular", "أنجولار"],
    "REST API": ["REST API", "web api", "rest"],
    "SignalR": ["SignalR"],

    # --- Data / AI ---
    "NLP": ["Natural Language Processing", "NLP", "معالجة اللغة الطبيعية", "معالجة اللغات الطبيعية"],
    "Machine Learning": ["Machine Learning", "ml", "تعلم الآلة", "التعلم الآلي"],
    "Arabic NLP": ["Arabic NLP", "AraBERT", "معالجة اللغة العربية"],
    "Data Analysis": ["Data Analysis", "تحليل البيانات"],

    # --- Tooling / infra ---
    "Docker": ["Docker", "دوكر", "الحاويات"],
    "Git": ["Git", "جيت", "التحكم بالإصدارات"],
    "Azure": ["Azure", "أزور", "الحوسبة السحابية"],

    # --- Cross-cutting engineering skills (paraphrase-friendly, show semantic power) ---
    "Web Development": ["Web Development", "تطوير الويب", "تطوير المواقع", "برمجة الويب"],
    "Mobile Development": ["Mobile Development", "تطوير تطبيقات الموبايل", "برمجة تطبيقات الهاتف"],
    "Software Architecture": ["Software Architecture", "معمارية البرمجيات", "تصميم الأنظمة"],
    "Project Management": ["Project Management", "إدارة المشاريع"],
    "Teamwork": ["Teamwork", "العمل الجماعي", "العمل ضمن فريق", "ضمن فريق", "روح الفريق", "عمل جماعي"],
    "Communication": ["Communication", "مهارات التواصل", "التواصل"],
}

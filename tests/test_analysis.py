import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer import KeywordAnalyzer
from cv_profile import extract_experience_years


class MatchingTests(unittest.TestCase):
    def test_all_required_skills_produce_full_match(self):
        result = KeywordAnalyzer().analyze(
            "Python Docker NLP Teamwork",
            ["Python", "Docker", "NLP", "Teamwork"],
        )
        self.assertEqual(100, result.matching_score)

    def test_missing_required_skill_reduces_match(self):
        result = KeywordAnalyzer().analyze(
            "Python Docker NLP",
            ["Python", "Docker", "NLP", "Teamwork"],
        )
        self.assertEqual(75, result.matching_score)


class ArabicExperienceTests(unittest.TestCase):
    def test_arabic_number_word(self):
        self.assertEqual(7, extract_experience_years("لدي خبرة سبع سنوات في البرمجة"))

    def test_arabic_indic_digits(self):
        self.assertEqual(7, extract_experience_years("خبرة ٧ سنوات"))

    def test_approximate_phrase(self):
        self.assertEqual(5, extract_experience_years("خبرة أكثر من خمس سنوات"))

    def test_english_digits(self):
        self.assertEqual(7, extract_experience_years("7+ years of experience"))


if __name__ == "__main__":
    unittest.main()

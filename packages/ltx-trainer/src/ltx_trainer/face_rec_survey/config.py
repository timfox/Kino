"""IJCIT 2025 facial recognition survey (Bahjat)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_TITLE = "A survey of facial recognition techniques"
PAPER_JOURNAL = "International Journal of Communication and Information Technology"
PAPER_YEAR = 2025
PAPER_VOLUME = "6(2)"
PAPER_PAGES = "214-225"
PAPER_DOI = "10.33545/2707661X.2025.v6.i2c.167"
PAPER_URL = f"https://doi.org/{PAPER_DOI}"
PAPER_AUTHOR = "Aya Kaysan Bahjat"
PAPER_AFFILIATION = "Informatics Institute for Postgraduate Studies, Baghdad, Iraq"


@dataclass
class FaceRecSurveyConfig:
    paper_title: str = PAPER_TITLE
    paper_doi: str = PAPER_DOI
    paper_url: str = PAPER_URL
    n_pca_components: int = 40
    n_subjects: int = 40
    n_images_per_subject: int = 10
    image_size: int = 92

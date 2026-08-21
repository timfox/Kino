"""Multimodal Retrieval Base — frozen web sandbox (Sec. 3.2)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.veritrip.config import VeriTripConfig


@dataclass
class MRBDocument:
    doc_id: str
    title: str
    text: str
    city: str = ""
    tags: tuple[str, ...] = ()


@dataclass
class MRBImage:
    image_id: str
    path: str
    caption: str
    linked_poi: str = ""
    city: str = ""


@dataclass
class MultimodalRetrievalBase:
    """In-memory MRB stub; production uses Faiss + Qwen3-Embedding + DINOv2."""

    cfg: VeriTripConfig = field(default_factory=VeriTripConfig)
    documents: dict[str, MRBDocument] = field(default_factory=dict)
    images: dict[str, MRBImage] = field(default_factory=dict)

    def add_document(self, doc: MRBDocument) -> None:
        self.documents[doc.doc_id] = doc

    def add_image(self, img: MRBImage) -> None:
        self.images[img.image_id] = img

    def doc_search(self, query: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
        """Keyword retrieval proxy for docSearch (Table 1)."""
        k = top_k if top_k is not None else self.cfg.doc_search_top_k
        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
        scored: list[tuple[float, MRBDocument]] = []
        for doc in self.documents.values():
            blob = f"{doc.title} {doc.text}".lower()
            score = sum(1 for t in terms if t in blob)
            if score > 0:
                scored.append((float(score), doc))
        scored.sort(key=lambda x: (-x[0], x[1].doc_id))
        out: list[dict[str, Any]] = []
        for score, doc in scored[:k]:
            snippet = doc.text[: self.cfg.snippet_max_tokens * 4]
            out.append({"docID": doc.doc_id, "score": score, "snippet": snippet, "title": doc.title})
        return out

    def img_search(self, img_path: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
        """Visual search proxy: match by filename / linked POI caption (Table 1)."""
        k = top_k if top_k is not None else self.cfg.doc_search_top_k
        key = img_path.lower()
        scored: list[tuple[float, MRBImage]] = []
        for img in self.images.values():
            score = 0.0
            if key in img.path.lower():
                score += 2.0
            if img.linked_poi and img.linked_poi.lower() in key:
                score += 1.0
            if score > 0:
                scored.append((score, img))
        if not scored:
            # Return all images with baseline score for demo
            scored = [(0.5, img) for img in self.images.values()]
        scored.sort(key=lambda x: (-x[0], x[1].image_id))
        return [
            {
                "imageID": img.image_id,
                "path": img.path,
                "caption": img.caption,
                "linked_poi": img.linked_poi,
                "score": score,
            }
            for score, img in scored[:k]
        ]

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        doc = self.documents.get(doc_id)
        if not doc:
            return None
        return {"docID": doc.doc_id, "title": doc.title, "text": doc.text, "city": doc.city}

    def stats(self) -> dict[str, int]:
        return {"documents": len(self.documents), "images": len(self.images)}


def build_demo_mrb() -> MultimodalRetrievalBase:
    mrb = MultimodalRetrievalBase()
    mrb.add_document(
        MRBDocument(
            doc_id="doc-hnm-001",
            title="Hunan Provincial Museum visitor guide",
            city="Changsha",
            tags=("attraction", "museum"),
            text=(
                "The Hunan Provincial Museum houses the famous Mawangdui Han Dynasty tombs "
                "including a bronze artifact collection. Recommended visit 2-3 hours. "
                "Opening hours 09:00-17:00. Ticket about 50 RMB per person."
            ),
        )
    )
    mrb.add_document(
        MRBDocument(
            doc_id="doc-cs-train-001",
            title="Changsha high-speed rail October 2025",
            city="Changsha",
            tags=("transportation",),
            text="Train G1011 departs Guangzhou South 08:00 arrives Changsha South 14:30. Return G1012 15:00-21:00.",
        )
    )
    mrb.add_image(
        MRBImage(
            image_id="img-bronze-crop",
            path="anchors/changsha_bronze_crop.jpg",
            caption="Cropped bronze vessel from Hunan Provincial Museum exhibit",
            linked_poi="Hunan Provincial Museum",
            city="Changsha",
        )
    )
    return mrb

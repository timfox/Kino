"""Sub-principle evaluators — rule-based stand-ins for the 13 LLM agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.agentfair.config import SUB_PRINCIPLES
from ltx_trainer.agentfair.extract import MetadataRecord
from ltx_trainer.agentfair.rubric import RUBRIC


@dataclass
class EvaluationResult:
    principle: str
    score: int
    confidence: float
    evidence: list[dict[str, str]] = field(default_factory=list)
    rationale: str = ""
    recommendations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "principle": self.principle,
            "score": self.score,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "rationale": self.rationale,
            "recommendations": list(self.recommendations),
        }


def _clip_score(n: int) -> int:
    return max(0, min(3, int(n)))


def _ev(source: str, snippet: str) -> dict[str, str]:
    return {"source": source, "snippet": snippet[:240]}


def evaluate_subprinciple(principle: str, meta: MetadataRecord) -> EvaluationResult:
    """Evidence-grounded maturity score without calling an LLM."""
    if principle not in RUBRIC:
        raise KeyError(principle)
    f = meta.fields
    d = meta.deterministic
    evidence: list[dict[str, str]] = []
    score = 0
    conf = 0.55
    recs: list[str] = []

    if principle == "F1":
        if d.get("identifier_present"):
            score = 2
            evidence.append(_ev("pid", str(f.get("pid") or f.get("doi"))))
            if d.get("https_landing"):
                score = 3
            conf = 0.85
        else:
            recs.append("Mint a resolvable DOI/Handle and embed it in landing metadata.")
            conf = 0.7
    elif principle == "F2":
        rich = sum(1 for k in ("title", "creator", "abstract", "keywords", "spatial_extent") if f.get(k))
        score = 1 if f.get("title") else 0
        if rich >= 3:
            score = 2
        if rich >= 4:
            score = 3
        evidence.extend(_ev(k, str(f[k])) for k in ("title", "abstract") if f.get(k))
        conf = 0.8 if score >= 2 else 0.6
        if score < 3:
            recs.append("Add abstract, keywords, and spatial/temporal coverage (ISO 19115).")
    elif principle == "F3":
        if f.get("distribution") or f.get("download_url") or f.get("access_url"):
            score = 2
            evidence.append(_ev("distribution", str(f.get("distribution") or f.get("download_url"))))
            if f.get("api_access") or f.get("wms") or f.get("stac"):
                score = 3
            conf = 0.75
        else:
            recs.append("Expose machine-readable distribution / access URLs in metadata.")
    elif principle == "F4":
        if f.get("datacite") or f.get("schema_org") or f.get("catalog_index"):
            score = 2
            evidence.append(_ev("index", "catalog or Schema.org present"))
            if (f.get("datacite") or f.get("catalog_index")) and f.get("schema_org"):
                score = 3
            conf = 0.7
        else:
            score = 1 if f.get("title") else 0
            recs.append("Register with DataCite / expose Schema.org JSON-LD.")
    elif principle == "A1.1":
        if f.get("protocol") in {"https", "http", "ftp"} or d.get("https_landing"):
            score = 3 if f.get("api_access") or f.get("open_access") else 2
            evidence.append(_ev("protocol", str(f.get("protocol") or "https")))
            conf = 0.85
        else:
            recs.append("Publish over open standard protocols (HTTPS / OGC / STAC).")
    elif principle == "A1.2":
        if f.get("open_access"):
            score = 3
            evidence.append(_ev("access", "open_access"))
            conf = 0.8
        elif f.get("auth") or f.get("oauth") or f.get("api_key"):
            score = 2
            evidence.append(_ev("auth", str(f.get("auth") or "api_key")))
            conf = 0.7
        else:
            score = 1 if d.get("https_landing") else 0
            recs.append("Document authentication clearly (OAuth / API key / open).")
    elif principle == "A2":
        if d.get("preservation_hint"):
            score = 2
            evidence.append(_ev("preservation", str(f.get("preservation_policy"))))
            conf = 0.55
        else:
            score = 1 if d.get("identifier_present") else 0
            conf = 0.45
            recs.append("Publish an explicit metadata preservation / tombstone policy.")
    elif principle == "I1":
        if f.get("json_ld") or f.get("rdf") or f.get("dcat"):
            score = 2
            evidence.append(_ev("kr", "JSON-LD/RDF/DCAT"))
            if f.get("geosparql") or d.get("epsg_or_crs"):
                score = 3
            conf = 0.65
        else:
            score = 1 if f.get("schema_org") else 0
            conf = 0.5
            recs.append("Emit JSON-LD / DCAT with geospatial semantics (GeoSPARQL).")
    elif principle == "I2":
        regs = sum(1 for k in ("vocabulary_registry", "gcmd", "cf_conventions", "sweet") if f.get(k))
        score = min(3, regs)
        if score == 0 and (f.get("epsg") or f.get("iso19115")):
            score = 1  # community vocab without generic registry
            conf = 0.4
            recs.append("Register geospatial vocabularies in LOV/BARTOC or document governance.")
        else:
            conf = 0.55 if score else 0.35
            if score < 2:
                recs.append("Ground terms in FAIR vocabulary registries (GCMD/CF/SWEET).")
    elif principle == "I3":
        if f.get("related_identifier") or f.get("derived_from") or f.get("cites"):
            score = 2
            evidence.append(_ev("relation", str(f.get("related_identifier") or f.get("cites"))))
            if f.get("prov_o") or f.get("datacite_relation"):
                score = 3
            conf = 0.6
        else:
            recs.append("Add typed DataCite/PROV relations to related datasets.")
            conf = 0.45
    elif principle == "R1.1":
        if f.get("spdx") or (f.get("license") and "CC" in str(f.get("license")).upper()):
            score = 3 if f.get("spdx") else 2
            evidence.append(_ev("license", str(f.get("license") or f.get("spdx"))))
            conf = 0.85
        elif f.get("license"):
            score = 1
            evidence.append(_ev("license", str(f["license"])))
            conf = 0.6
            recs.append("Use a standard SPDX/CC license identifier in machine-readable form.")
        else:
            recs.append("Add a clear machine-readable usage license.")
    elif principle == "R1.2":
        bits = sum(1 for k in ("provenance", "orcid", "lineage", "version", "methodology") if f.get(k))
        score = min(3, bits)
        if bits:
            evidence.append(_ev("provenance", str(f.get("provenance") or f.get("lineage") or "present")))
        conf = 0.65 if bits >= 2 else 0.5
        if bits < 2:
            recs.append("Provide structured lineage (ISO/PROV-O) and creator ORCID.")
    elif principle == "R1.3":
        bits = sum(
            1
            for k in ("iso19115", "cf_netcdf", "stac", "epsg", "crs")
            if f.get(k) or (k == "epsg" and d.get("epsg_or_crs"))
        )
        score = min(3, max(1, bits) if bits else 0)
        if bits:
            evidence.append(_ev("community", f"{bits} geospatial standards signals"))
            conf = 0.7
        else:
            conf = 0.4
            recs.append("Adopt ISO 19115 / CF-NetCDF / STAC and declare CRS (EPSG).")
    else:
        raise KeyError(principle)

    rationale = (
        f"{principle} ({RUBRIC[principle].name}): maturity {score}/3 "
        f"from deterministic + structured fields."
    )
    return EvaluationResult(
        principle=principle,
        score=_clip_score(score),
        confidence=float(conf),
        evidence=evidence,
        rationale=rationale,
        recommendations=recs,
    )


def evaluate_all(meta: MetadataRecord) -> dict[str, EvaluationResult]:
    return {p: evaluate_subprinciple(p, meta) for p in SUB_PRINCIPLES}

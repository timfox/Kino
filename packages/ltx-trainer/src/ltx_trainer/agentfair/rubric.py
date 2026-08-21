"""0–3 maturity rubric for the 13 FAIR sub-principles (Table 2)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.agentfair.config import SUB_PRINCIPLES


@dataclass(frozen=True)
class SubPrincipleSpec:
    principle_id: str
    dimension: str
    name: str
    evidence_hints: tuple[str, ...]
    geo_indicators: tuple[str, ...]


RUBRIC: dict[str, SubPrincipleSpec] = {
    "F1": SubPrincipleSpec(
        "F1",
        "F",
        "Globally unique persistent identifier",
        ("pid", "doi", "handle", "content_negotiation"),
        ("DOI", "NASA CMR", "STAC IDs"),
    ),
    "F2": SubPrincipleSpec(
        "F2",
        "F",
        "Rich metadata description",
        ("title", "creator", "abstract", "keywords", "spatial_extent"),
        ("ISO 19115", "spatial extent"),
    ),
    "F3": SubPrincipleSpec(
        "F3",
        "F",
        "Metadata includes data identifier",
        ("distribution", "download_url", "access_url"),
        ("OPeNDAP", "WMS/WFS", "STAC"),
    ),
    "F4": SubPrincipleSpec(
        "F4",
        "F",
        "Registered in searchable resource",
        ("datacite", "schema_org", "catalog_index"),
        ("CKAN API", "CSW", "STAC"),
    ),
    "A1.1": SubPrincipleSpec(
        "A1.1",
        "A",
        "Standardized retrieval protocol",
        ("protocol", "https", "api_access"),
        ("OGC", "STAC API", "THREDDS"),
    ),
    "A1.2": SubPrincipleSpec(
        "A1.2",
        "A",
        "Authentication or authorization support",
        ("auth", "oauth", "api_key", "open_access"),
        ("Earthdata Login", "ESA Hubs"),
    ),
    "A2": SubPrincipleSpec(
        "A2",
        "A",
        "Metadata preserved without data",
        ("preservation_policy", "coretrustseal", "tombstone"),
        ("PANGAEA", "SDI catalogs"),
    ),
    "I1": SubPrincipleSpec(
        "I1",
        "I",
        "Formal knowledge representation",
        ("json_ld", "rdf", "dcat", "schema_org"),
        ("GeoSPARQL", "DCAT", "Schema.org"),
    ),
    "I2": SubPrincipleSpec(
        "I2",
        "I",
        "FAIR vocabularies used",
        ("vocabulary_registry", "gcmd", "cf_conventions"),
        ("GCMD", "CF conventions", "SWEET"),
    ),
    "I3": SubPrincipleSpec(
        "I3",
        "I",
        "Qualified references to other data",
        ("related_identifier", "derived_from", "cites"),
        ("DataCite relations", "PROV-O"),
    ),
    "R1.1": SubPrincipleSpec(
        "R1.1",
        "R",
        "Clear data usage license",
        ("license", "spdx", "creative_commons"),
        ("OGL", "CC-BY", "ODbL"),
    ),
    "R1.2": SubPrincipleSpec(
        "R1.2",
        "R",
        "Detailed provenance",
        ("provenance", "orcid", "lineage", "version"),
        ("ISO 19115 lineage", "PROV-O"),
    ),
    "R1.3": SubPrincipleSpec(
        "R1.3",
        "R",
        "Domain community standards",
        ("iso19115", "cf_netcdf", "stac", "epsg"),
        ("ISO 19139", "CF-NetCDF", "STAC"),
    ),
}


def assert_rubric_complete() -> None:
    missing = set(SUB_PRINCIPLES) - set(RUBRIC)
    if missing:
        raise AssertionError(f"rubric missing: {sorted(missing)}")


def maturity_label(score: int) -> str:
    return {0: "non-compliant", 1: "partial", 2: "substantial", 3: "full"}.get(
        int(score), "invalid"
    )

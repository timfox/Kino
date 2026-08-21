"""Hybrid metadata extraction (Stage 1) — deterministic signals from a record."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


@dataclass
class MetadataRecord:
    """Unified metadata record M with provenance pointers."""

    url: str
    fields: dict[str, Any] = field(default_factory=dict)
    snippets: list[dict[str, str]] = field(default_factory=list)
    deterministic: dict[str, bool] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.fields.get(key, default)


_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
_EPSG_RE = re.compile(r"EPSG[:\s]?\d{3,6}", re.I)


def deterministic_checks(url: str, fields: dict[str, Any]) -> dict[str, bool]:
    """Paper §4.4 Findability-style hard checks (subset for CPU stub)."""
    text_blob = " ".join(str(v) for v in fields.values() if v is not None)
    pid = bool(fields.get("pid") or fields.get("doi") or _DOI_RE.search(text_blob))
    https = urlparse(url).scheme.lower() == "https" or bool(fields.get("https"))
    schema = bool(fields.get("schema_org") or fields.get("json_ld"))
    license_ok = bool(fields.get("license") or fields.get("spdx"))
    return {
        "identifier_present": pid,
        "https_landing": https,
        "schema_org_or_jsonld": schema,
        "license_present": license_ok,
        "epsg_or_crs": bool(fields.get("epsg") or fields.get("crs") or _EPSG_RE.search(text_blob)),
        "ogc_or_stac": bool(
            fields.get("wms")
            or fields.get("wfs")
            or fields.get("stac")
            or any(k in text_blob.upper() for k in ("WMS", "WFS", "STAC"))
        ),
        "preservation_hint": bool(fields.get("preservation_policy") or fields.get("coretrustseal")),
    }


def extract_metadata(url: str, raw: dict[str, Any] | None = None, *, repository: str = "") -> MetadataRecord:
    """Merge structured fields + lightweight enrichment (no network)."""
    fields = dict(raw or {})
    if repository and "repository" not in fields:
        fields["repository"] = repository
    if url and "landing_url" not in fields:
        fields["landing_url"] = url
    # Normalize common aliases
    if "doi" in fields and "pid" not in fields:
        fields["pid"] = fields["doi"]
    if fields.get("json_ld") and not fields.get("schema_org"):
        fields["schema_org"] = True
    det = deterministic_checks(url, fields)
    snippets: list[dict[str, str]] = []
    for key in ("title", "abstract", "license", "pid", "doi"):
        if fields.get(key):
            snippets.append({"source": key, "snippet": str(fields[key])[:240]})
    return MetadataRecord(url=url, fields=fields, snippets=snippets, deterministic=det)


def demo_records() -> list[MetadataRecord]:
    """Three paper case-study style fixtures (Table 4 proxies)."""
    return [
        extract_metadata(
            "https://zenodo.org/records/demo-crater-lake",
            {
                "title": "Crater Lake bathymetry",
                "creator": "Demo Lab",
                "abstract": "High-resolution bathymetric survey",
                "keywords": ["bathymetry", "lake"],
                "doi": "10.5281/zenodo.0000001",
                "pid": "10.5281/zenodo.0000001",
                "https": True,
                "schema_org": True,
                "json_ld": True,
                "datacite": True,
                "distribution": "https://zenodo.org/api/records/0000001/files",
                "download_url": "https://zenodo.org/api/files/demo.nc",
                "protocol": "https",
                "api_access": True,
                "open_access": True,
                "license": "CC-BY-4.0",
                "spdx": "CC-BY-4.0",
                "provenance": "survey 2024; processed with GDAL",
                "orcid": "0000-0002-0000-0000",
                "version": "1.0",
                "iso19115": True,
                "epsg": "EPSG:4326",
                "cf_netcdf": True,
                "related_identifier": "10.5281/zenodo.0000002",
                "preservation_policy": "zenodo_tombstone",
                "repository": "Zenodo",
            },
            repository="Zenodo",
        ),
        extract_metadata(
            "https://aurin.org.au/demo-osm-pois",
            {
                "title": "OSM POIs",
                "creator": "AURIN",
                "https": True,
                "protocol": "https",
                "license": "ODbL",
                "provenance": "OpenStreetMap extract",
                "repository": "AURIN",
            },
            repository="AURIN",
        ),
        extract_metadata(
            "https://earthdata.nasa.gov/demo-gdis",
            {
                "title": "GDIS Disasters",
                "creator": "NASA",
                "abstract": "Global disaster inventory",
                "doi": "10.5067/DEMO/GDIS",
                "pid": "10.5067/DEMO/GDIS",
                "https": True,
                "schema_org": True,
                "datacite": True,
                "distribution": "https://earthdata.nasa.gov/demo",
                "protocol": "https",
                "auth": "Earthdata Login",
                "api_key": True,
                "license": "NASA Open",
                "epsg": "EPSG:4326",
                "stac": False,
                "repository": "Earthdata",
            },
            repository="Earthdata",
        ),
    ]

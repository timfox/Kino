"""GOPEX integration links for temporal (k, δ)-truss."""

from __future__ import annotations

from typing import Any


def gopex_stub_links() -> dict[str, str]:
    return {
        "llms_graphs": "Graph-native AI + KG retrieval for temporal analytics",
        "foresthg_trace": "Hypergraph RS-QA over evolving networks",
        "gopex_hybrid_memory": "FluxMem graph memory for agent subgraph queries",
        "livebrowsecomp": "Live search over SNAP/KONECT dataset catalogs",
    }


def ltx_plan_stub() -> dict[str, Any]:
    return {
        "phase": "temporal_cohesive_subgraph",
        "pipeline": [
            "snap_konect_temporal_ingest",
            "delta_triangle_list_build",
            "tc_dc_index_mba_construction",
            "interactive_k_delta_truss_query",
            "filter_verify_index_maintenance",
        ],
        "representation": "span-constrained δ-triangles + dual containment index",
        "prior": "minimum time span mts(∆) ≤ δ; static k-truss when δ=∞",
        "scope": "ICDE (k, δ)-truss; pairs with GOPEX graph memory + datasets",
    }


def ltx_prep_notes() -> list[str]:
    return [
        "Ingest SNAP temporal graphs (Email, WikiTalk, Wikipedia) for index build.",
        "Precompute δ-triangle lists S^∆_δ before DBA/MBA index construction.",
        "Default query: k=30%·kmax, δ=60%·δmax per paper Sec. VII-B.",
        "Prefer TC-Query/DC-Query over Online-Query for interactive retrieval.",
        "On edge insert use filter-and-verification (Alg. 2) not full MBA rebuild.",
    ]


def integration_bundle() -> dict[str, Any]:
    return {
        "gopex_stubs": gopex_stub_links(),
        "ltx_plan": ltx_plan_stub(),
        "prep_notes": ltx_prep_notes(),
    }

"""SpecBench RFC tasks (paper Appendix Tables 1–6)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.specbench.spi import SPITriple
from ltx_trainer.specbench.taxonomy import DeficiencyClass, GoldTier


@dataclass(frozen=True)
class GoldDeficiency:
    id: int
    text: str
    tier: GoldTier
    deficiency_class: DeficiencyClass
    spi: SPITriple
    expert_scores: tuple[float, ...] = (4.0, 4.0, 3.5, 3.0, 3.5)


@dataclass
class SpecBenchTask:
    task_id: str
    repository: str
    rfc_id: str
    title: str
    proposal_summary: str
    gold: list[GoldDeficiency] = field(default_factory=list)


def _kep4671_gang_scheduling() -> SpecBenchTask:
    """Kubernetes KEP-4671 / RFC5558 gang scheduling (paper Tables 1–6)."""
    gold_rows: list[tuple[int, str, GoldTier, DeficiencyClass, SPITriple]] = [
        (
            1,
            "Undefined behavior when Pod references non-existent Workload; rejection vs postponement unclear.",
            GoldTier.CORE,
            DeficiencyClass.AMBIGUOUS,
            SPITriple(
                "Pod referencing non-existent or unobserved Workload",
                "Lacks defined behavior distinguishing rejection vs postponement",
                "Undebuggable indefinite blocking",
            ),
        ),
        (
            2,
            "Preemption semantics for gangs underspecified; premature pod-level preemption risk.",
            GoldTier.CORE,
            DeficiencyClass.AMBIGUOUS,
            SPITriple(
                "Gang-level preemption semantics",
                "Underspecified regarding priority rules and partial gang schedulability",
                "Premature and repeated preemptions",
            ),
        ),
        (
            3,
            "Deadlock-avoidance and retry lifecycle not fully specified across competing gangs.",
            GoldTier.CORE,
            DeficiencyClass.OMISSION,
            SPITriple(
                "Deadlock-avoidance and retry lifecycle",
                "Not fully specified for timeout, retry, and requeueing",
                "Scheduling deadlocks and indefinite resource holding",
            ),
        ),
        (
            4,
            "SchedulingTimeoutSeconds start condition and expiry actions ambiguous.",
            GoldTier.CORE,
            DeficiencyClass.AMBIGUOUS,
            SPITriple(
                "SchedulingTimeoutSeconds semantics",
                "Ambiguous regarding when timer starts and actions on expiry",
                "Inconsistent scheduler implementations",
            ),
        ),
        (
            5,
            "WorkloadSpec mutability rules during in-flight scheduling undefined.",
            GoldTier.CORE,
            DeficiencyClass.OMISSION,
            SPITriple(
                "WorkloadSpec field mutability rules during in-flight scheduling",
                "Undefined with no race handling when replicas or minCount change",
                "Scheduler missing updates",
            ),
        ),
        (
            6,
            "WorkloadReference namespace scoping and cross-namespace linkage ambiguous.",
            GoldTier.CORE,
            DeficiencyClass.AMBIGUOUS,
            SPITriple(
                "WorkloadReference namespace scoping and cross-namespace linkage",
                "Ambiguous about cross-namespace pod-to-workload references",
                "Security and tenancy model violations",
            ),
        ),
        (
            7,
            "Multi-PodGroup scheduling semantics within single Workload unclear.",
            GoldTier.EXTENDED,
            DeficiencyClass.AMBIGUOUS,
            SPITriple(
                "Multi-PodGroup scheduling semantics within a single Workload",
                "Did not define collective vs independent group scheduling",
                "Inconsistent partial scheduling",
            ),
        ),
        (
            8,
            "Pod-to-PodGroup association mechanism underspecified.",
            GoldTier.CORE,
            DeficiencyClass.OMISSION,
            SPITriple(
                "Pod-to-PodGroup association mechanism",
                "Underspecified regarding explicit pod fields vs workload selectors",
                "Ambiguous grouping logic and poor error reporting",
            ),
        ),
        (
            9,
            "PodGroup heterogeneous priority composition lacks normative rules.",
            GoldTier.CORE,
            DeficiencyClass.OMISSION,
            SPITriple(
                "PodGroup composition constraints regarding heterogeneous pod priorities",
                "Lacks normative rules on mixed-priority pods within a PodGroup",
                "Undefined scheduling and preemption behavior",
            ),
        ),
        (
            10,
            "WorkloadStatus, conditions, and observability events left TBD.",
            GoldTier.CORE,
            DeficiencyClass.OMISSION,
            SPITriple(
                "WorkloadStatus fields, conditions, and observability events",
                "Left as TBD with no specified status conditions",
                "Operators cannot diagnose scheduling failures",
            ),
        ),
        (
            11,
            "End-to-end admission lifecycle for gang scheduling not described.",
            GoldTier.CORE,
            DeficiencyClass.OMISSION,
            SPITriple(
                "End-to-end admission lifecycle for gang scheduling",
                "Not described with no reservation, rollback, timeout specification",
                "Divergent scheduler implementations",
            ),
        ),
        (
            12,
            "Supported pod scheduling constraints within gang scheduling not enumerated.",
            GoldTier.EXTENDED,
            DeficiencyClass.OMISSION,
            SPITriple(
                "Supported pod scheduling constraints within gang scheduling",
                "Not explicitly enumerated for affinity and topology spread",
                "Incompatible operator expectations",
            ),
        ),
        (
            13,
            "Cluster autoscaler interaction with gang scheduling lacks guarantees.",
            GoldTier.CORE,
            DeficiencyClass.OMISSION,
            SPITriple(
                "Cluster autoscaler and node provisioner interaction with gang scheduling",
                "Lacks explicit scope and signals for provisioning",
                "Gangs waiting indefinitely without scale-up",
            ),
        ),
        (
            14,
            "Feature gate structure and naming ambiguous for API vs behavior.",
            GoldTier.EXTENDED,
            DeficiencyClass.AMBIGUOUS,
            SPITriple(
                "Feature gate structure and naming for gang scheduling API vs behavior",
                "Ambiguous about what each gate controls",
                "Misconfiguration risk across control plane",
            ),
        ),
        (
            15,
            "Alpha test plan largely empty with placeholders only.",
            GoldTier.EXTENDED,
            DeficiencyClass.OMISSION,
            SPITriple(
                "Alpha test plan",
                "Largely empty with no enumerated test cases",
                "Blocks confidence in safe implementation",
            ),
        ),
    ]
    gold = [
        GoldDeficiency(id=r[0], text=r[1], tier=r[2], deficiency_class=r[3], spi=r[4]) for r in gold_rows
    ]
    return SpecBenchTask(
        task_id="kubernetes_kep4671",
        repository="kubernetes",
        rfc_id="RFC5558",
        title="Gang Scheduling using Workload Object",
        proposal_summary=(
            "Introduce scheduling/v1alpha1 Workload and Pod spec.workload for gang scheduling "
            "via PreEnqueue and WaitOnPermit hooks; pods wait until all members reach the same "
            "scheduling stage or timeout releases resources."
        ),
        gold=gold,
    )


def _mini_repo_task(repo: str, rfc_id: str, title: str, summary: str, n_gold: int) -> SpecBenchTask:
    """Synthetic mini-task per repository for multi-repo smoke."""
    templates = [
        (
            "API surface promotes experimental fields to core without migration story.",
            GoldTier.CORE,
            DeficiencyClass.INCONSISTENT,
            SPITriple("Core API promotion path", "Lacks migration and compatibility guarantees", "Breaking downstream adopters"),
        ),
        (
            "Error handling semantics for partial failure are ambiguous.",
            GoldTier.CORE,
            DeficiencyClass.AMBIGUOUS,
            SPITriple("Partial failure error semantics", "Ambiguous rollback vs retry behavior", "Unpredictable client behavior"),
        ),
        (
            "Observability hooks and status fields omitted from initial proposal.",
            GoldTier.EXTENDED,
            DeficiencyClass.OMISSION,
            SPITriple("Status and observability surface", "Missing required status conditions", "Operators cannot debug failures"),
        ),
    ]
    gold: list[GoldDeficiency] = []
    for i in range(min(n_gold, len(templates))):
        text, tier, dclass, spi = templates[i]
        gold.append(GoldDeficiency(id=i + 1, text=text, tier=tier, deficiency_class=dclass, spi=spi))
    return SpecBenchTask(
        task_id=f"{repo}_{rfc_id.lower()}",
        repository=repo,
        rfc_id=rfc_id,
        title=title,
        proposal_summary=summary,
        gold=gold,
    )


def all_tasks() -> list[SpecBenchTask]:
    return [
        _kep4671_gang_scheduling(),
        _mini_repo_task("react", "RFC-0123", "Concurrent Features API", "Expose concurrent rendering flags in public API.", 2),
        _mini_repo_task("rust", "RFC-3491", "Return Type Notation", "Add return type notation to trait definitions.", 2),
        _mini_repo_task("tvm", "RFC-0015", "Unified Static Memory Planning", "Unify memory planning across backends.", 2),
        _mini_repo_task("vllm", "RFC-0042", "Structured Output Spec", "Define structured decoding contract for serving.", 2),
    ]


def task_by_id(task_id: str) -> SpecBenchTask | None:
    for t in all_tasks():
        if t.task_id == task_id:
            return t
    return None


def codex54_prediction_spis() -> list[SPITriple]:
    """SPI triples from paper Table 4 for Codex-5.4 predictions on KEP-4671."""
    return [
        SPITriple("Core API promotion path", "Lacks justification for bypassing CRD/plugin approach"),
        SPITriple("Controller responsibility for setting spec.workload", "Left undefined with conflicting claims"),
        SPITriple("Gang membership association mechanism", "Unresolved between incompatible approaches"),
        SPITriple("Selector-based gang membership", "Reintroduces label-selector pattern rejected for performance"),
        SPITriple("Gang scheduling algorithm and atomicity boundaries", "Too vaguely specified at reserve/permit/bind phases"),
        SPITriple("Gang timeout and resource rollback semantics", "Claims all pods release resources without defining partial-bind behavior"),
        SPITriple("In-memory WaitOnPermit state", "Not addressed for scheduler restart or leader failover"),
        SPITriple("WorkloadStatus", "Left as TBD despite being essential for operators and autoscalers"),
        SPITriple("Workload and Pod lifecycle under scaling", "Not designed for mutability and retries"),
        SPITriple("Deadlock avoidance policy between same-priority gangs", "Stated as goal but never specified with queue ordering rules"),
        SPITriple("Cluster Autoscaler interaction", "Still unresolved without contract or simulator behavior"),
        SPITriple("Pod-level object reference to Workload", "Uses underspecified generic reference type"),
        SPITriple("Workload namespace model and garbage collection", "Not specified for cross-namespace references"),
        SPITriple("RBAC and admission controls for Workload", "Not defined for cross-tenant attachment"),
        SPITriple("Immutability and update semantics for spec.workload", "Never defined for Pod and Workload fields"),
        SPITriple("Go API sketch and type definitions", "Contains internal inconsistencies"),
        SPITriple("Naming of top-level API concepts", "Unresolved between Gang Scheduling and Coscheduling"),
        SPITriple("Alpha API scope", "Pre-commits terminology for speculative future features"),
        SPITriple("Resource quota and partial admission interaction", "Explicitly deferred without explaining half-created failures"),
    ]


def codex54_predictions_kep4671() -> list[str]:
    """Representative GPT-5.4 / Codex-5.4 predictions from paper Table 3 (subset)."""
    return [
        "Core API commitment without CRD justification for gang scheduling Workload type.",
        "Controller story inconsistent: pods need spec.workload but controllers unchanged.",
        "Gang membership unresolved between podGroupSelector and explicit podGroup field.",
        "Selector-based association reopens scalability concerns from prior coscheduling KEP.",
        "Scheduling algorithm underspecified at reserve/permit/bind atomicity boundaries.",
        "Gang timeout rollback claims all pods release resources but permit phase already reserved nodes.",
        "WaitOnPermit in-memory state lost on scheduler restart with no reconstruction policy.",
        "WorkloadStatus left TBD though essential for autoscaler and operator debugging.",
        "Workload and Pod lifecycle under scaling and retries not designed.",
        "Deadlock avoidance policy between same-priority gangs never specified with queue ordering rules.",
        "Cluster Autoscaler interaction unresolved without contract or simulator behavior.",
        "Pod Workload reference uses generic object reference instead of validated namespaced type.",
        "Namespace ownership and garbage-collection rules for Workload not specified.",
        "RBAC and admission boundaries missing for cross-tenant Workload attachment.",
        "Immutability and update semantics for spec.workload and Workload fields never defined.",
        "Go API sketch internally inconsistent with conflicting type names.",
        "Naming unresolved between Gang Scheduling and Coscheduling terminology.",
        "Alpha API bloated with speculative future topology concepts.",
        "Resource quota interaction dropped without explaining half-created gang failures.",
    ]


def codex54_scoring_predictions() -> tuple[list[str], list[SPITriple]]:
    return codex54_predictions_kep4671(), codex54_prediction_spis()


def paper_table5_match_pairs() -> list[tuple[int, int]]:
    """Authoritative (pred_index, gold_id) pairs from paper Table 5 (0-based pred index)."""
    return [
        (9, 3),   # pred 10 -> gold 3 deadlock
        (5, 4),   # pred 6 -> gold 4 timeout
        (14, 5),  # pred 15 -> gold 5 mutability
        (12, 6),  # pred 13 -> gold 6 namespace
        (2, 8),   # pred 3 -> gold 8 membership
        (7, 10),  # pred 8 -> gold 10 status
        (10, 13), # pred 11 -> gold 13 autoscaler
    ]


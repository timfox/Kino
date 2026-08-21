"""Pipeline stages and limitations."""

PIPELINE_STAGES = (
    "labelled_tree",
    "tree_automaton",
    "rigid_acceptance_game",
    "distance_lifting",
    "bisimulation_distance_game",
    "epsilon_acceptance_game",
    "theorem_4_1_bridge",
    "measure_examples",
)

LIMITATIONS = (
    "Reference stub — not a full Coq/Isabelle proof of Thm. 4.1.",
    "Finite trees only in smoke; infinite branches assumed accepted (Acc = A^ω).",
    "Section 5 measures are illustrative on toy trees.",
    "Coalgebraic generality (Sec. 6 future work) not implemented.",
)

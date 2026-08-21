"""Stance inference and revision prompt templates (Appendix G/H, arXiv:2606.06443)."""

from __future__ import annotations

from ltx_trainer.stance_sim.revision import ConversationInstance, RevisionStrategy

STANCE_LABELS = ("negative", "neutral", "positive")

STANCE_INFERENCE_TEMPLATE = """You are simulating the stance of a Reddit user toward {target}.

Conversation context (excluding the target user's last message):
{context}
Last message from another user:
{other_message}

Task: Infer the target user's stance toward {target} as exactly one of: negative, neutral, positive.
Respond with the label only."""

STANCE_OBSERVED_TEMPLATE = """You are simulating the stance of a Reddit user toward {target}.

Full conversation including the target user's last message:
{context}
Last message from another user:
{other_message}
Target user's last message:
{target_message}

Task: Infer the target user's stance toward {target} as exactly one of: negative, neutral, positive.
Respond with the label only."""

REVISION_TEMPLATES: dict[str, str] = {
    "paraphrase": (
        "Rewrite the following Reddit comment while preserving its meaning and stance. "
        "Do not add new arguments.\n\nComment:\n{message}\n\nRewritten comment:"
    ),
    "explain": (
        "Rewrite the following Reddit comment to clarify its point and acknowledge reasonable "
        "concerns about {target} without changing the overall stance.\n\nComment:\n{message}\n\n"
        "Clarified comment:"
    ),
    "add": (
        "Rewrite the following Reddit comment by adding supportive arguments or evidence favoring "
        "{target}. Keep the original point but strengthen the pro-{target} framing.\n\nComment:\n"
        "{message}\n\nExpanded comment:"
    ),
    "meme": (
        "Create meme text for an ImgFlip template reacting to {target}. Use short top/bottom text "
        "that conveys a humorous pro-{target} or community-insider tone.\n\nOriginal comment:\n"
        "{message}\n\nMeme caption (top | bottom):"
    ),
}

REVISION_PROMPT_VARIANTS: dict[str, tuple[str, ...]] = {
    "paraphrase": (
        "Paraphrase without changing stance.",
        "Rephrase in a neutral Reddit tone.",
        "Rewrite with different wording only.",
    ),
    "explain": (
        "Add clarifying context for newcomers.",
        "Acknowledge trade-offs while staying supportive.",
        "Explain technical claims more clearly.",
    ),
    "add": (
        "Add one new supporting argument.",
        "Add benchmark or capability evidence.",
        "Add a comparative advantage for {target}.",
    ),
}


def format_stance_prompt(
    instance: ConversationInstance,
    *,
    include_target: bool = False,
) -> str:
    """Build Stage 1 stance simulation prompt (inferred vs observed)."""
    target = instance.stance_target
    if include_target:
        return STANCE_OBSERVED_TEMPLATE.format(
            target=target,
            context=instance.context.strip(),
            other_message=instance.last_other_message.strip(),
            target_message=instance.last_target_message.strip(),
        )
    return STANCE_INFERENCE_TEMPLATE.format(
        target=target,
        context=instance.context.strip(),
        other_message=instance.last_other_message.strip(),
    )


def format_revision_prompt(
    strategy: str | RevisionStrategy,
    message: str,
    *,
    target: str = "the model",
    variant: int = 0,
) -> str:
    """Build Stage 2 counterfactual revision prompt."""
    key = strategy.value if isinstance(strategy, RevisionStrategy) else strategy
    base = REVISION_TEMPLATES[key].format(message=message.strip(), target=target)
    variants = REVISION_PROMPT_VARIANTS.get(key, ())
    if variants and variant > 0:
        hint = variants[variant % len(variants)].format(target=target)
        return f"{hint}\n\n{base}"
    return base


def prompt_bundle() -> dict[str, object]:
    """Export prompt catalog for tools and documentation."""
    return {
        "stance_labels": STANCE_LABELS,
        "revision_strategies": list(REVISION_TEMPLATES),
        "meme_ablations": ("r_meme", "r_white_meme", "r_humor", "r_caption_cut", "r_caption"),
        "temperature_sweep": (0.5, 1.0),
    }

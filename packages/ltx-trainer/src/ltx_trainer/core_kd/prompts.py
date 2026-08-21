"""Caption hints for conversational MER when GOPEX_CORE_KD is enabled."""

CORE_KD_CAPTION_HINT = (
    "For conversational emotion recognition: note speaker turn, dialogue context, "
    "and whether language/audio/visual cues agree on the target utterance emotion."
)

LTX_VIDEO_PROMPT_SUFFIX = (
    " CoRe-KD: preserve target-utterance emotion when modalities disagree; "
    "note missing or unreliable audio/visual cues in dialogue."
)

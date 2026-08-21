"""Caption / LTX prompt hints for counting-aware video captions."""

VLM_COUNT_CAPTION_HINT = (
    "When describing scenes with repeated discrete objects (stones, pieces, crowd units), "
    "note approximate counts only when clearly visible; distinguish target objects from distractors."
)

LTX_VIDEO_PROMPT_SUFFIX = (
    " [vlm-count: individuation→magnitude→symbolic mapping; avoid conflating distractor density with target count]"
)

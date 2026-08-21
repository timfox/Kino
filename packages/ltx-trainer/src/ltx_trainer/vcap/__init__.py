"""VCap: witness-adjudicator hypergeometric rewards for visual captioning (arXiv:2605.28023)."""

from ltx_trainer.vcap.captioning import (
    attach_vcap_caption_meta,
    attach_vcap_to_preprocess_meta,
    dense_caption_instruction,
    merge_preprocess_extra,
    prune_manifest_low_reward,
    qa_manifest_directory,
    score_caption_pair,
    summarize_qa_reports,
    tag_precomputed_projects,
    vcap_caption_enabled,
    vcap_epoch,
    vcap_preprocess_extra,
    vcap_user_prompt_lines,
)
from ltx_trainer.vcap.config import VCapConfig
from ltx_trainer.vcap.hypergeometric import p_collision, p_recall
from ltx_trainer.vcap.pipeline import (
    evaluation_demo,
    evaluation_smoke,
    framework_card,
    knowledge_card,
)
from ltx_trainer.vcap.selfimprove import (
    backup_witness_pool,
    batch_backup_projects,
    e2_workflow_card,
    recaption_command,
)
from ltx_trainer.vcap.rewards import grpo_advantages, mock_judge_scores, sentence_reward

__all__ = [
    "VCapConfig",
    "backup_witness_pool",
    "batch_backup_projects",
    "e2_workflow_card",
    "recaption_command",
    "attach_vcap_caption_meta",
    "attach_vcap_to_preprocess_meta",
    "dense_caption_instruction",
    "merge_preprocess_extra",
    "prune_manifest_low_reward",
    "qa_manifest_directory",
    "summarize_qa_reports",
    "tag_precomputed_projects",
    "vcap_caption_enabled",
    "vcap_epoch",
    "vcap_user_prompt_lines",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "grpo_advantages",
    "knowledge_card",
    "mock_judge_scores",
    "p_collision",
    "p_recall",
    "score_caption_pair",
    "sentence_reward",
    "vcap_preprocess_extra",
]

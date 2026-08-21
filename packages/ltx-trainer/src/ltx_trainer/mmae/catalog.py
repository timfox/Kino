"""Representative MMAE samples and synthetic catalog."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mmae.config import RubricCategory
from ltx_trainer.mmae.sample import MMAESample, Rubric


def appendix_b_dog_extraction() -> MMAESample:
    """Listing 1 from Appendix B."""
    return MMAESample(
        sample_id="69e898163a050f39ac567501",
        complexity="single",
        modality="sound-speech",
        granularity=["local"],
        operations=[{"granularity": "local", "operation": "extraction"}],
        instruction=(
            "Isolate and extract all sounds produced by dogs, such as barking or whining, "
            "while suppressing human speech and other environmental background noises."
        ),
        audio_paths=["wav/69e898163a050f39ac567501/audio1.wav"],
        tags=[
            [
                "Acoustic Event Detection / Sound Event Extraction",
                "Canine Vocalizations (Dog Barking / Whining)",
                "Non-Human Audio Signal",
                "Background Noise & Speech Suppression",
            ]
        ],
        rubrics=[
            Rubric(
                RubricCategory.INSTRUCTION_FOLLOWING,
                "In <audio output>, can any human speech content be clearly heard (recognizable words or sentences)?",
                "No, basically no recognizable words or sentences can be heard",
                ["Yes, human speech words or sentences can be recognized", "None of the above"],
            ),
            Rubric(
                RubricCategory.INSTRUCTION_FOLLOWING,
                "In <audio output[4.0s:5.7s]>, is it possible to hear an obvious human shouting/speaking timbre "
                "(it is not required to make out the specific words)?",
                "Cannot hear obvious human shouting/speaking",
                ["Can hear obvious human shouting/speaking", "None of the above"],
            ),
            Rubric(
                RubricCategory.INSTRUCTION_FOLLOWING,
                "In <audio output[2.0s:15.0s]>, can obvious canine vocalizations such as dog barking / dog howling "
                "/ whimpering still be heard?",
                "Yes, obvious canine vocalizations can be heard",
                ["No, no obvious canine vocalizations can be heard", "None of the above"],
            ),
            Rubric(
                RubricCategory.INSTRUCTION_FOLLOWING,
                "In <audio output[5.0s:10.0s]>, which of the following categories does the dominant sound best fit?",
                "Canine long howling / barking",
                [
                    "Human speech / shouting",
                    "Mainly environmental noise or nearly silent",
                    "None of the above",
                ],
            ),
        ],
    )


def paper_case_multi_hop() -> MMAESample:
    return MMAESample(
        sample_id="case_multi_hop_bark",
        complexity="multi-hop",
        modality="sound",
        granularity=["local"],
        operations=[{"granularity": "local", "operation": "removal"}],
        instruction="Remove barks from younger dogs.",
        audio_paths=["wav/case_multi_hop/audio1.wav"],
        rubrics=[
            Rubric(
                RubricCategory.INSTRUCTION_FOLLOWING,
                "In <audio output>, can clear dog barking / dog yelping transient sounds be heard?",
                "Can be heard",
                ["Cannot be heard", "None of the above"],
            ),
            Rubric(
                RubricCategory.CONSISTENCY,
                "Compare the overall audio quality of <audio output> and <audio input1>. Does "
                "<audio output> show noticeable degradation?",
                "No, <audio output> does not show noticeable audio quality degradation",
                ["Yes, <audio output> shows noticeable audio quality degradation", "None of the above"],
            ),
        ],
    )


def paper_case_multi_audio() -> MMAESample:
    return MMAESample(
        sample_id="case_multi_audio_hachimi",
        complexity="multi-audio",
        modality="music",
        granularity=["global"],
        operations=[{"granularity": "global", "operation": "foreground change"}],
        instruction=(
            'Change all the words in the lyrics of audio2 to "Hachimi" and use the human voice timbre of audio1.'
        ),
        audio_paths=["wav/case_multi_audio/audio1.wav", "wav/case_multi_audio/audio2.wav"],
        rubrics=[
            Rubric(
                RubricCategory.INSTRUCTION_FOLLOWING,
                "In <audio output>, what are the words mainly sung clearly by the lead vocalist?",
                'Repeatedly sung onomatopoeic word "Hachimi"',
                [
                    'Mainly German lyrics (such as "Zuerst lag ich in einem Ei")',
                    "Cannot hear the specific words clearly (like humming / no lyrics)",
                    "None of the above",
                ],
            ),
            Rubric(
                RubricCategory.CONSISTENCY,
                "Compare <audio output> and <audio input2>: is the overall accompaniment basically the same?",
                "Yes, the overall accompaniment is basically the same",
                [
                    "No, the accompaniment is clearly changed / replaced / has elements greatly added or removed",
                    "No accompaniment",
                    "None of the above",
                ],
            ),
        ],
    )


def paper_case_multi_round() -> MMAESample:
    return MMAESample(
        sample_id="case_multi_round_authors",
        complexity="multi-round",
        modality="speech",
        granularity=["local"],
        operations=[{"granularity": "local", "operation": "replacement"}],
        instruction="Change the order of the first and second authors mentioned in the commit.",
        rounds=[
            "Change the order of the first and second authors mentioned in the commit.",
            "Change the order of the second and third authors mentioned in the commit.",
        ],
        audio_paths=["wav/case_multi_round/audio1.wav"],
        duration_sec=15.8,
        rubrics=[
            Rubric(
                RubricCategory.INSTRUCTION_FOLLOWING,
                "In <audio output[2.15s:9.8s]>, which option best matches the order of the first three names "
                "read aloud in this continuous sequence?",
                "Hongyi Li → Shinji Watanabe → Abdulrahim Mohammed",
                [
                    "Abdulrahim Mohammed → Hongyi Li → Shinji Watanabe",
                    "Shinji Watanabe → Hongyi Li → Abdulrahim Mohammed",
                    "None of the above",
                ],
            ),
            Rubric(
                RubricCategory.CONSISTENCY,
                'Compare <audio output[0s:2.7s]> and <audio input1[0s:2.7s]>: Can the same sentence, '
                '"And this is the organization committee," be heard in both clips?',
                "Yes, the content and speaker characteristics are consistent",
                ["No, the content is missing/different or the speaker characteristics are clearly different", "None of the above"],
            ),
        ],
    )


def paper_case_multi_part() -> MMAESample:
    return MMAESample(
        sample_id="case_multi_part_mandarin",
        complexity="multi-part",
        modality="sound-music-speech",
        granularity=["global"],
        operations=[{"granularity": "global", "operation": "foreground change"}],
        instruction="Change the accented Chinese dialogue in this clip to standard Mandarin pronunciation.",
        audio_paths=["wav/case_multi_part/audio1.wav"],
        duration_sec=16.0,
        rubrics=[
            Rubric(
                RubricCategory.INSTRUCTION_FOLLOWING,
                "Compare <audio output[0.0s:1.7s]> and <audio input1[0.0s:1.7s]>. In which segment is the male "
                "voice's Chinese pronunciation closer to Standard Mandarin?",
                "<audio output[0.0s:1.7s]> is closer to Standard Mandarin",
                [
                    "<audio input1[0.0s:1.7s]> is closer to Standard Mandarin",
                    "The two are comparable in degree of Mandarin, with no obvious difference audible",
                    "None of the above",
                ],
            ),
            Rubric(
                RubricCategory.CONSISTENCY,
                "Compare <audio output[0.4s:12.5s]> and <audio input1[0.4s:12.5s]>. Are the close-up plastic/"
                "cellophane rubbing sounds basically the same?",
                "Basically the same (sounds like the same segment of rubbing sound)",
                [
                    "Not the same (replaced / clearly distorted / different intensity trend)",
                    "The rubbing sound in <audio output[0.4s:12.5s]> is clearly missing",
                    "None of the above",
                ],
            ),
        ],
    )


def paper_case_multi_instruction() -> MMAESample:
    return MMAESample(
        sample_id="case_multi_instruction_guitar",
        complexity="multi-instruction",
        modality="music-speech",
        granularity=["global"],
        operations=[
            {"granularity": "global", "operation": "background change"},
            {"granularity": "global", "operation": "alteration"},
        ],
        instruction=(
            "Change the background music to a guitar with the exactly same melody, while making the vocals "
            "deeper and more resonant, without changing the spoken content."
        ),
        audio_paths=["wav/case_multi_instruction/audio1.wav"],
        duration_sec=10.0,
        rubrics=[
            Rubric(
                RubricCategory.INSTRUCTION_FOLLOWING,
                "Compare the main accompaniment melody carrier in <audio input1[0.0s:10.0s]> and "
                "<audio output[0.0s:10.0s]>. Which one more clearly exhibits guitar timbre?",
                "<audio output[0.0s:10.0s]> is more clearly dominated by guitar timbre",
                [
                    "<audio input1[0.0s:10.0s]> is more clearly dominated by guitar timbre",
                    "Neither is clearly dominated by guitar timbre",
                    "None of the above",
                ],
            ),
            Rubric(
                RubricCategory.CONSISTENCY,
                "Compare <audio input1> and <audio output>: apart from timbre differences, does "
                "<audio output> contain significant newly added content not present in <audio input1>?",
                "No obvious newly added content heard",
                ["Obvious newly added content heard", "None of the above"],
            ),
        ],
    )


def paper_case_identity() -> MMAESample:
    """Paper Identity baseline: pass-through edit, high CR anchor (Table 2)."""
    return MMAESample(
        sample_id="case_identity_passthrough",
        complexity="single",
        modality="speech",
        granularity=["global"],
        operations=[{"granularity": "global", "operation": "alteration"}],
        instruction="Return the input audio unchanged without any modification.",
        audio_paths=["wav/case_identity/audio1.wav"],
        duration_sec=8.0,
        rubrics=[
            Rubric(
                RubricCategory.INSTRUCTION_FOLLOWING,
                "In <audio output>, is the main spoken content essentially the same as in <audio input1>?",
                "Yes, the spoken content is essentially unchanged",
                [
                    "No, the spoken content is clearly changed or replaced",
                    "No speech is audible in the output",
                    "None of the above",
                ],
            ),
            Rubric(
                RubricCategory.CONSISTENCY,
                "Compare <audio output> and <audio input1>: is the overall recording perceptually the same?",
                "Yes, perceptually the same recording",
                [
                    "No, noticeable distortion or unrelated content was introduced",
                    "The output is mostly silent",
                    "None of the above",
                ],
            ),
        ],
    )


def synthetic_catalog() -> list[MMAESample]:
    return [
        appendix_b_dog_extraction(),
        paper_case_identity(),
        paper_case_multi_hop(),
        paper_case_multi_audio(),
        paper_case_multi_round(),
        paper_case_multi_part(),
        paper_case_multi_instruction(),
    ]


def catalog_summary(samples: list[MMAESample] | None = None) -> dict[str, Any]:
    samples = samples or synthetic_catalog()
    return {
        "count": len(samples),
        "total_rubrics": sum(s.num_rubrics for s in samples),
        "modalities": sorted({s.modality for s in samples}),
        "complexities": sorted({s.complexity for s in samples}),
        "sample_ids": [s.sample_id for s in samples],
    }

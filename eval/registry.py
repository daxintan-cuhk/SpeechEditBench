"""
Task registry for benchmark-level evaluation orchestration.

The registry centralizes:
- task id
- default samples path
- whether the task is currently enabled for `--task all`
- evaluator callable (run_evaluation)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


RunEvaluation = Callable[..., dict]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    run_evaluation: RunEvaluation
    default_samples_file: Path
    enabled: bool = True


def get_task_registry(repo_root: Path) -> dict[str, TaskSpec]:
    """Build task registry lazily to avoid importing all evaluators on startup."""
    from eval.tasks.acoustic_editing import run_evaluation as acoustic_eval
    from eval.tasks.compositional_editing import run_evaluation as compositional_eval
    from eval.tasks.content_editing import run_evaluation as content_eval
    from eval.tasks.emotion_editing import run_evaluation as emotion_eval
    from eval.tasks.paralinguistic_editing import run_evaluation as paralinguistic_eval
    from eval.tasks.prosody_editing import run_evaluation as prosody_eval
    from eval.tasks.speaker_editing import run_evaluation as speaker_eval
    from eval.tasks.style_editing import run_evaluation as style_eval

    data_root = repo_root / "data"
    return {
        "acoustic_editing": TaskSpec(
            task_id="acoustic_editing",
            run_evaluation=acoustic_eval,
            default_samples_file=data_root / "acoustic_editing" / "samples.jsonl",
        ),
        "content_editing": TaskSpec(
            task_id="content_editing",
            run_evaluation=content_eval,
            default_samples_file=data_root / "content_editing" / "samples.jsonl",
        ),
        "compositional_editing": TaskSpec(
            task_id="compositional_editing",
            run_evaluation=compositional_eval,
            default_samples_file=data_root / "compositional_editing" / "samples.jsonl",
        ),
        "emotion_editing": TaskSpec(
            task_id="emotion_editing",
            run_evaluation=emotion_eval,
            default_samples_file=data_root / "emotion_editing" / "samples.jsonl",
        ),
        "paralinguistic_editing": TaskSpec(
            task_id="paralinguistic_editing",
            run_evaluation=paralinguistic_eval,
            default_samples_file=data_root / "paralinguistic_editing" / "samples.jsonl",
        ),
        "prosody_editing": TaskSpec(
            task_id="prosody_editing",
            run_evaluation=prosody_eval,
            default_samples_file=data_root / "prosody_editing" / "samples.jsonl",
        ),
        "speaker_editing": TaskSpec(
            task_id="speaker_editing",
            run_evaluation=speaker_eval,
            default_samples_file=data_root / "speaker_editing" / "samples.jsonl",
        ),
        "style_editing": TaskSpec(
            task_id="style_editing",
            run_evaluation=style_eval,
            default_samples_file=data_root / "style_editing" / "samples.jsonl",
        ),
    }

from __future__ import annotations

from pathlib import Path

import run_qwen25_omni_micro5 as micro


REPO_ROOT = Path(__file__).resolve().parents[3]

RESULT_DIR = (
    REPO_ROOT
    / "local_eval/emotion_judge/results/"
      "qwen25_omni_gptq_int4/micro5_prompt_v2"
)


def build_prompt_v2(row: dict) -> str:
    language = row["language"]
    transcript = row.get("transcript", "")
    labels = ", ".join(row["allowed_labels"])

    if language == "zh":
        return f"""
你是一名严格的语音情感分类评测器。

转写文本仅用于帮助你对齐语音内容。不得根据文本所描述的事件、词语含义、
讽刺内容或人物关系推断情感。必须主要根据声音本身判断，包括：

- 音高及音高变化
- 音量和能量
- 语速与停顿
- 声音紧张度、气声和音质
- 韵律、重音和情感唤醒程度

类别定义：

angry：
真正的愤怒、敌意或攻击性。声音通常紧张、强硬、有持续责备或对抗感。

fearful：
害怕、焦虑或受到威胁。声音可能紧张、颤抖、犹豫或气息不稳。

happy：
愉快、满足或温暖的积极情绪，通常不像高度兴奋那样强烈。

neutral：
没有明确的主导情感，声音平稳、普通。

playfulness：
调侃、开玩笑、淘气或轻松的讽刺。可能有夸张语调，但整体具有轻松、
戏谑或“带笑”的感觉，而不是真正的敌意。

sad：
悲伤、失落或低落。通常能量低、语速偏慢、声音压抑。

surprise：
突然的惊讶或意外。通常表现为短暂而明显的音高、能量或节奏突变，
而不是持续的责备和敌意。

关键区分规则：

- playfulness 与 angry：
  轻松、调侃、夸张但没有真实敌意时选择 playfulness；
  持续紧张、攻击和真正责备时才选择 angry。
- surprise 与 angry：
  突然的意外和惊讶选择 surprise；
  持续的敌意或对抗选择 angry。
- neutral 与 playfulness：
  存在明显的调侃、戏谑韵律时不能选择 neutral。

转写文本：
{transcript}

可选标签：
{labels}

只输出以下 JSON，不得输出解释、Markdown、置信度或其他文字：

{{"predicted_emotion":"<label>"}}
""".strip()

    return f"""
You are a strict speech-emotion classification evaluator.

The transcript is provided only to align the spoken content. Do not use the
meaning of the words, the described situation, or the topic as evidence for
the emotion. Base the decision primarily on the voice itself:

- pitch and pitch movement
- energy and loudness
- speaking rate and pauses
- voice tension, breathiness, and voice quality
- rhythm, emphasis, valence, and emotional arousal

Label definitions:

angry:
Genuine hostility, confrontation, or outward aggression. The voice is usually
tense, forceful, attacking, or persistently blaming.

excited:
Strong positive high-arousal enthusiasm or anticipation. The voice is
energetic, animated, bright, and usually faster or more dynamic than happy.

fearful:
Fear, anxiety, or perceived threat. The voice may sound tense, shaky,
hesitant, breathy, or unstable.

frustrated:
Exasperation caused by difficulty, blockage, helplessness, or repeated
failure. The voice is strained or dissatisfied, but is less directly hostile
and attacking than angry.

happy:
Positive pleasure, warmth, or satisfaction. It is usually calmer and less
highly aroused than excited.

neutral:
No clearly dominant emotion. The delivery is ordinary and relatively stable.

sad:
Sorrow, loss, discouragement, or low mood. The voice is usually subdued,
slow, low-energy, or withdrawn.

surprise:
Sudden astonishment or startle, often marked by an abrupt pitch, energy, or
timing change. It is not sustained hostility.

Critical contrast rules:

- excited versus happy:
  Select excited only when positive emotion is clearly high-arousal,
  energetic, and animated. Otherwise select happy.
- frustrated versus sad:
  Select frustrated when tension, blockage, helpless exasperation, or strained
  dissatisfaction dominates. Select sad when subdued sorrow and low energy
  dominate.
- angry versus frustrated:
  Select angry when outward attack, hostility, or confrontation dominates.
  Select frustrated when the dissatisfaction is strained or helpless rather
  than attacking.
- surprise versus angry:
  Select surprise for abrupt astonishment; select angry only for sustained
  hostility.

Transcript:
{transcript}

Allowed labels:
{labels}

Return only the following JSON. Do not output an explanation, confidence,
Markdown, or any additional text:

{{"predicted_emotion":"<label>"}}
""".strip()


def main() -> None:
    # prepare_inputs() 会在运行时调用 core.build_prompt，
    # 因而可以在不修改原始 v1 脚本的情况下替换 Prompt。
    micro.core.build_prompt = build_prompt_v2

    micro.RESULT_DIR = RESULT_DIR
    micro.RESULT_JSONL = RESULT_DIR / "predictions.jsonl"
    micro.SUMMARY_JSON = RESULT_DIR / "summary.json"

    print("Prompt version      : v2_contrastive_label_definitions")
    print("Output directory    :", RESULT_DIR)

    micro.main()


if __name__ == "__main__":
    main()

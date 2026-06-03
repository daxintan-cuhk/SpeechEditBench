"""
副语言事件分类体系（Paralinguistic Taxonomy）。

The public evaluator imports these event labels, descriptions, and thresholds
from ``eval.resources.paralinguistic_taxonomy``. Keep this file in sync with
``eval/resources/paralinguistic_annotation_prompt.txt``.
"""

# ── 全部副语言事件标签（顺序固定，作为评分维度的规范顺序）─────────────────────

ALL_EVENTS = [
    "breath",   # 呼吸声
    "laugh",    # 笑声
    "cough",    # 咳嗽声
    "sigh",     # 叹气声
]

# ── Gemini 音频检测阈值 ────────────────────────────────────────────────────────

# event_scores 中分值 >= DETECT_THRESHOLD → 事件明显存在 → 可作为 remove 候选
DETECT_THRESHOLD = 2

# event_scores 中分值 <= ABSENT_THRESHOLD → 事件基本不存在 → 可作为 add 候选
ABSENT_THRESHOLD = 1

# ── 事件自然语言描述（供 GPT-4o 指令生成使用）────────────────────────────────

EVENT_DESCRIPTIONS = {
    "breath": {
        "en": "audible breathing sounds (inhales, exhales, breath noise)",
        "zh": "可听见的呼吸声（吸气声、呼气声、气息噪声）",
    },
    "laugh": {
        "en": "laughter or laughing sounds (giggles, chuckles, audible laughing)",
        "zh": "笑声（咯咯笑、轻笑、明显的笑声）",
    },
    "cough": {
        "en": "audible coughing sounds (single coughs or repeated cough bursts)",
        "zh": "可听见的咳嗽声（单次或连续咳嗽）",
    },
    "sigh": {
        "en": "audible sighing sounds (long exhales expressing relief, fatigue, or emotion)",
        "zh": "可听见的叹气声（带情绪的长呼气）",
    },
}

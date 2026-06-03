"""
风格分类体系的单一数据源（Style Taxonomy）。

The public evaluator imports these labels and descriptions from
``eval.resources.style_taxonomy``. Keep the label list and descriptions in sync
with any style-judge prompt changes.
"""

# ── 全部风格标签（顺序固定，作为评分维度的规范顺序）────────────────────────────

ALL_STYLES = [
    "public-broadcast",
    "intimate",
    "dramatic",
    "restrained-flat",
    "storytelling",
    "conversational",
]

# ── Gemini 音频标注阈值 ────────────────────────────────────────────────────────

# primary_style 判断：各维中最高分 ≥ PRIMARY_MIN_SCORE 才认为有显著主风格
PRIMARY_MIN_SCORE = 3

# 高置信准入阈值（用于 Step1 入库筛选）
# - confidence: Gemini 返回的整体置信度
# - margin: primary_score - second_best_score（统一要求 >= STYLE_MARGIN_MIN）
STYLE_CONFIDENCE_THRESHOLDS = {
    "public-broadcast": 0.70,
    "intimate": 0.70,
    "dramatic": 0.70,
    "restrained-flat": 0.70,
    "storytelling": 0.72,
    "conversational": 0.78,
}
STYLE_MARGIN_MIN = 1

# ── 风格自然语言描述（供 GPT-4o 文本过滤 prompt 和编辑指令生成使用）─────────────

STYLE_DESCRIPTIONS = {
    "public-broadcast": {
        "en": "public broadcast style (clear projection, authoritative, addressing a broad audience like a news anchor)",
        "zh": "播报式（清晰投射、面向广众的正式播报风格，如新闻主播）",
    },
    "intimate": {
        "en": "intimate style (soft, personal, close and directed at one listener, warm vocal quality)",
        "zh": "私密近讲式（柔和、个人化、面向单人的亲近感，温暖音质）",
    },
    "dramatic": {
        "en": "dramatic style (theatrical, expressive, wide prosodic range, performed and heightened delivery)",
        "zh": "戏剧化/表演化（夸张的韵律起伏、高张力的表演感）",
    },
    "restrained-flat": {
        "en": "restrained flat style (controlled, understated, minimal prosodic variation, deliberately subdued)",
        "zh": "克制平直（低起伏、收敛的克制表达风格）",
    },
    "storytelling": {
        "en": "storytelling style (narrative pacing, scene-guiding delivery that draws the listener into a story)",
        "zh": "叙事引导式（有故事感、引领听众的叙述节奏）",
    },
    "conversational": {
        "en": "conversational style (natural, spontaneous everyday speech, relaxed and casual register)",
        "zh": "口语会话式（自然随意的日常口语，轻松随意的风格）",
    },
}

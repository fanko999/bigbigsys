"""
用户发言性质识别与画像服务 - 按角色隔离
"""
import json
import re
import time
from typing import Any, Dict, List

from role_context import ensure_role_storage

LABEL_FACTUAL = "factual"
LABEL_UNCERTAIN = "uncertain"
LABEL_HUMOROUS = "humorous"
LABEL_EXAGGERATED = "exaggerated"
LABEL_TEST_BEHAVIOR = "test_behavior"
LABEL_EMOTIONAL = "emotional"


def _profile_file():
    return ensure_role_storage()["memory"] / "user_profile.json"


def load_user_profile() -> Dict[str, Any]:
    file = _profile_file()
    if file.exists():
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {
        "counts": {},
        "last_labels": [],
        "style_summary": "",
        "updated_at": "",
    }


def save_user_profile(data: Dict[str, Any]) -> None:
    data["updated_at"] = str(int(time.time()))
    with open(_profile_file(), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def classify_utterance(text: str) -> Dict[str, Any]:
    content = (text or "").strip()
    lowered = content.lower()
    reasons: List[str] = []
    label = LABEL_FACTUAL
    confidence = "medium"

    if not content:
        return {"label": LABEL_UNCERTAIN, "confidence": "low", "reasons": ["空内容"]}

    if any(marker in lowered for marker in ["测试", "你还记得", "你知道吗", "你是不是", "你能不能", "幻觉", "模型"]):
        label = LABEL_TEST_BEHAVIOR
        confidence = "high"
        reasons.append("出现明显测试模型/记忆的表达")

    emotional_markers = ["难受", "生气", "崩溃", "烦", "开心", "伤心", "委屈", "焦虑", "害怕", "紧张"]
    if any(marker in content for marker in emotional_markers):
        label = LABEL_EMOTIONAL if label == LABEL_FACTUAL else label
        reasons.append("以情绪表达为主")

    humorous_markers = ["哈哈", "嘿嘿", "笑死", "逗你", "开玩笑", "吹牛", "狗头", "LOL", "233"]
    if any(marker.lower() in lowered for marker in humorous_markers):
        label = LABEL_HUMOROUS if label not in (LABEL_TEST_BEHAVIOR,) else label
        confidence = "high"
        reasons.append("带有明显玩笑/调侃标记")

    uncertain_markers = ["可能", "大概", "好像", "也许", "估计", "差不多", "记不清"]
    if any(marker in content for marker in uncertain_markers):
        label = LABEL_UNCERTAIN if label == LABEL_FACTUAL else label
        reasons.append("包含不确定表达")

    exaggerated_patterns = [
        r"\d+\s*个?亿",
        r"\d+\s*万亿",
        r"随便赚",
        r"秒赚",
        r"宇宙第一",
        r"天下无敌",
        r"一天赚",
        r"闭眼赚",
    ]
    if any(re.search(pattern, content, re.IGNORECASE) for pattern in exaggerated_patterns):
        label = LABEL_EXAGGERATED if label not in (LABEL_TEST_BEHAVIOR,) else label
        confidence = "high"
        reasons.append("包含明显夸张/吹牛表达")

    if not reasons:
        reasons.append("更像普通事实陈述")
        confidence = "medium"

    return {"label": label, "confidence": confidence, "reasons": reasons[:3]}


def should_promote_to_fact(label: str) -> bool:
    return label in (LABEL_FACTUAL, LABEL_UNCERTAIN)


def get_memory_importance_for_label(label: str) -> float:
    if label == LABEL_FACTUAL:
        return 0.55
    if label == LABEL_UNCERTAIN:
        return 0.4
    if label == LABEL_EMOTIONAL:
        return 0.45
    if label == LABEL_HUMOROUS:
        return 0.3
    if label == LABEL_EXAGGERATED:
        return 0.2
    if label == LABEL_TEST_BEHAVIOR:
        return 0.25
    return 0.35


def update_user_profile(classification: Dict[str, Any], text: str) -> Dict[str, Any]:
    profile = load_user_profile()
    counts = profile.setdefault("counts", {})
    label = classification.get("label", LABEL_FACTUAL)
    counts[label] = counts.get(label, 0) + 1

    last_labels = profile.setdefault("last_labels", [])
    last_labels.append({
        "label": label,
        "confidence": classification.get("confidence", "low"),
        "preview": (text or "")[:60],
        "timestamp": str(int(time.time())),
    })
    profile["last_labels"] = last_labels[-20:]
    profile["style_summary"] = build_user_style_summary(profile)
    save_user_profile(profile)
    return profile


def build_user_style_summary(profile: Dict[str, Any]) -> str:
    counts = profile.get("counts", {})
    if not counts:
        return "暂无明显表达风格特征"

    factual = counts.get(LABEL_FACTUAL, 0)
    exaggerated = counts.get(LABEL_EXAGGERATED, 0)
    humorous = counts.get(LABEL_HUMOROUS, 0)
    testing = counts.get(LABEL_TEST_BEHAVIOR, 0)
    emotional = counts.get(LABEL_EMOTIONAL, 0)
    uncertain = counts.get(LABEL_UNCERTAIN, 0)
    total = max(sum(counts.values()), 1)

    parts: List[str] = []
    if factual / total >= 0.55:
        parts.append("整体上以认真事实陈述为主")
    if exaggerated + humorous >= 2:
        if exaggerated >= humorous:
            parts.append("偶尔会用夸张或吹牛方式开玩笑")
        else:
            parts.append("偶尔会用调侃玩笑的方式表达")
    if testing >= 1:
        parts.append("会主动测试系统的记忆和判断边界")
    if emotional >= 1:
        parts.append("情绪表达比较直接")
    if uncertain >= 2:
        parts.append("有时会用保留和试探口吻描述信息")

    return "；".join(parts[:4]) or "暂无明显表达风格特征"


def get_user_profile_context_text() -> str:
    profile = load_user_profile()
    summary = profile.get("style_summary") or "暂无明显表达风格特征"
    counts = profile.get("counts", {})
    top_counts = []
    for label, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:4]:
        top_counts.append(f"{label}:{count}")
    details = "，".join(top_counts) if top_counts else "暂无统计"
    return f"【用户表达画像】\n- 概要: {summary}\n- 统计: {details}"

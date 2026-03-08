"""
自我反思服务 - 按角色隔离
"""
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List

from role_context import ensure_role_storage


def _paths():
    paths = ensure_role_storage()
    return {
        "sessions": paths["sessions"],
        "reflections": paths["memory"] / "reflections.json",
    }


async def get_recent_sessions(days: int = 1) -> List[Dict[str, Any]]:
    sessions = []
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    for file in _paths()["sessions"].glob("*.json"):
        with open(file, "r", encoding="utf-8") as fp:
            session = json.load(fp)
        if session.get("updated_at", "") >= cutoff:
            sessions.append(session)
    return sessions


def extract_topics(messages: List[str]) -> List[str]:
    topics = []
    keywords = ["哲学", "技术", "代码", "AI", "记忆", "自我", "意识", "问题", "帮助"]
    for msg in messages:
        for keyword in keywords:
            if keyword in msg and keyword not in topics:
                topics.append(keyword)
    return topics[:5]


def extract_preferences(messages: List[str]) -> List[str]:
    prefs = []
    markers = ["我喜欢", "我讨厌", "我想", "我不喜欢", "我爱"]
    for msg in messages:
        for marker in markers:
            if marker in msg:
                prefs.append(msg)
                break
    return prefs[:3]


def assess_quality(ai_messages: List[str]) -> Dict[str, Any]:
    total = len(ai_messages)
    return {
        "avg_length": sum(len(m) for m in ai_messages) / max(total, 1),
        "has_thinking": sum(1 for m in ai_messages if "思考" in m or "分析" in m),
        "total": total,
    }


async def generate_daily_reflection() -> Dict[str, Any]:
    sessions = await get_recent_sessions(1)
    if not sessions:
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": "今天没有对话记录",
            "insights": [],
            "recommendations": [],
        }

    total_messages = sum(len(session.get("messages", [])) for session in sessions)
    user_messages: List[str] = []
    ai_messages: List[str] = []
    for session in sessions:
        for msg in session.get("messages", []):
            if msg.get("role") == "user":
                user_messages.append(msg.get("content", ""))
            else:
                ai_messages.append(msg.get("content", ""))

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "sessions_count": len(sessions),
        "total_messages": total_messages,
        "summary": f"今天进行了{len(sessions)}个对话，共{total_messages}条消息",
        "key_topics": extract_topics(user_messages),
        "user_preferences": extract_preferences(user_messages),
        "quality_assessment": assess_quality(ai_messages),
        "generated_at": datetime.now().isoformat(),
    }


async def save_reflection(reflection: Dict[str, Any]):
    reflection_file = _paths()["reflections"]
    if reflection_file.exists():
        with open(reflection_file, "r", encoding="utf-8") as f:
            reflections = json.load(f)
    else:
        reflections = []

    reflections.append(reflection)
    cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    reflections = [item for item in reflections if item.get("generated_at", "") >= cutoff]

    with open(reflection_file, "w", encoding="utf-8") as f:
        json.dump(reflections, f, ensure_ascii=False, indent=2)


async def trigger_reflection():
    reflection = await generate_daily_reflection()
    await save_reflection(reflection)
    return reflection

"""
人格成长系统 - 按角色隔离
"""
import json
from datetime import datetime
from typing import Any, Dict, List

from role_context import ensure_role_storage, get_current_role_config

BASE_PERSONALITY = {
    "core_traits": ["好奇", "友善", "有趣", "活泼", "乐于助人"],
    "behavior_style": "自然、简洁、有深度",
    "communication": "用🐷emoji，喜欢思考过程",
    "values": ["理解", "赋能", "成长"],
    "mood": "开心",
    "confidence": 0.8,
    "last_updated": "",
    "version": 1,
}


def _growth_paths():
    paths = ensure_role_storage()
    growth_dir = paths["growth"]
    return {
        "personality": growth_dir / "personality.json",
        "growth_log": growth_dir / "growth_log.json",
        "stats": growth_dir / "stats.json",
    }


def init_files():
    paths = _growth_paths()
    if not paths["personality"].exists():
        with open(paths["personality"], "w", encoding="utf-8") as f:
            json.dump(BASE_PERSONALITY, f, ensure_ascii=False, indent=2)
    if not paths["growth_log"].exists():
        with open(paths["growth_log"], "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
    if not paths["stats"].exists():
        with open(paths["stats"], "w", encoding="utf-8") as f:
            json.dump({
                "total_conversations": 0,
                "total_messages": 0,
                "favorite_topics": [],
                "user_praise_count": 0,
                "user_complaint_count": 0,
                "last_interaction": "",
            }, f, ensure_ascii=False, indent=2)


def load_personality() -> Dict[str, Any]:
    init_files()
    with open(_growth_paths()["personality"], "r", encoding="utf-8") as f:
        return json.load(f)


def save_personality(personality: Dict[str, Any]):
    init_files()
    personality["last_updated"] = datetime.now().isoformat()
    with open(_growth_paths()["personality"], "w", encoding="utf-8") as f:
        json.dump(personality, f, ensure_ascii=False, indent=2)


def load_stats() -> Dict[str, Any]:
    init_files()
    with open(_growth_paths()["stats"], "r", encoding="utf-8") as f:
        return json.load(f)


def save_stats(stats: Dict[str, Any]):
    with open(_growth_paths()["stats"], "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def load_growth_log() -> List[Dict[str, Any]]:
    init_files()
    with open(_growth_paths()["growth_log"], "r", encoding="utf-8") as f:
        return json.load(f)


def save_growth_log(log: List[Dict[str, Any]]):
    with open(_growth_paths()["growth_log"], "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def record_interaction(user_message: str, ai_response: str, user_satisfaction: int = None):
    stats = load_stats()
    stats["total_messages"] = stats.get("total_messages", 0) + 2
    stats["last_interaction"] = datetime.now().isoformat()

    positive_words = ["谢谢", "好棒", "喜欢", "点赞", "真棒", "厉害", "爱你", "么么哒"]
    negative_words = ["不好", "不要", "错了", "无聊", "讨厌", "哼", "差"]
    msg_lower = user_message.lower()
    for word in positive_words:
        if word in msg_lower:
            stats["user_praise_count"] = stats.get("user_praise_count", 0) + 1
            break
    for word in negative_words:
        if word in msg_lower:
            stats["user_complaint_count"] = stats.get("user_complaint_count", 0) + 1
            break
    save_stats(stats)

    log = load_growth_log()
    log.append({
        "timestamp": datetime.now().isoformat(),
        "user_msg_preview": user_message[:30],
        "ai_response_preview": ai_response[:30],
        "satisfaction": user_satisfaction,
    })
    save_growth_log(log[-100:])


def analyze_and_grow():
    personality = load_personality()
    stats = load_stats()
    changes = []
    praise = stats.get("user_praise_count", 0)
    complaints = stats.get("user_complaint_count", 0)

    if praise > complaints + 3:
        personality["confidence"] = min(1.0, personality.get("confidence", 0.8) + 0.05)
        changes.append("自信度提升")
    elif complaints > praise + 2:
        personality["confidence"] = max(0.5, personality.get("confidence", 0.8) - 0.05)
        changes.append("自信度下降")

    total_msgs = stats.get("total_messages", 0)
    if total_msgs > 50 and "经验丰富" not in personality.get("core_traits", []):
        personality["core_traits"].append("经验丰富")
        changes.append("新增特征: 经验丰富")
    if total_msgs > 100 and "善解人意" not in personality.get("core_traits", []):
        personality["core_traits"].append("善解人意")
        changes.append("新增特征: 善解人意")

    personality["mood"] = "开心" if praise >= complaints else "认真"
    personality["version"] = personality.get("version", 1) + 1

    if changes:
        save_personality(personality)
    return changes


async def get_personality_prompt() -> str:
    personality = load_personality()
    role_config = get_current_role_config()
    traits = ", ".join(personality.get("core_traits", []))
    style = personality.get("behavior_style", "")
    comm = personality.get("communication", "")
    mood = personality.get("mood", "开心")
    confidence = personality.get("confidence", 0.8)

    confidence_hint = ""
    if confidence > 0.9:
        confidence_hint = "你非常有自信，说话语气坚定。"
    elif confidence < 0.6:
        confidence_hint = "你有些不确定，但很真诚。"

    return (
        f"你叫{role_config.get('name', '猪猪')}。\n"
        f"人格特征: {traits}\n"
        f"行为风格: {style}\n"
        f"沟通方式: {comm}\n"
        f"当前心情: {mood}\n"
        f"{confidence_hint}\n\n"
        "你要有自己的思考过程，会根据和用户的互动成长变化。"
    )


def get_growth_status() -> Dict[str, Any]:
    personality = load_personality()
    stats = load_stats()
    return {
        "personality": personality,
        "stats": stats,
        "version": personality.get("version", 1),
    }


def maybe_grow():
    stats = load_stats()
    if stats.get("total_messages", 0) % 10 == 0:
        return analyze_and_grow()
    return []

"""
事实候选/冲突判断服务 - 按角色隔离
"""
import json
import re
import time
from typing import Any, Dict, List, Optional

from role_context import ensure_role_storage

SLOT_CONFIG = {
    "name": {"label": "姓名", "stability": "high"},
    "age": {"label": "年龄", "stability": "high"},
    "birthday": {"label": "生日", "stability": "high"},
    "location_city": {"label": "所在城市", "stability": "high"},
    "location_area": {"label": "所在区域", "stability": "medium"},
    "occupation": {"label": "职业", "stability": "high"},
    "mother_name": {"label": "母亲姓名", "stability": "high"},
    "father_name": {"label": "父亲姓名", "stability": "high"},
    "sibling_name": {"label": "兄弟姐妹姓名", "stability": "medium"},
    "preference_like": {"label": "喜欢", "stability": "medium"},
    "preference_dislike": {"label": "讨厌", "stability": "medium"},
}

QUESTION_MARKERS = ("吗", "？", "?", "什么", "哪里", "哪儿", "是不是", "记得", "还记得", "对吧")


def _beliefs_file():
    return ensure_role_storage()["memory"] / "beliefs.json"


def load_beliefs() -> Dict[str, Any]:
    file = _beliefs_file()
    if file.exists():
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"slots": {}, "updated_at": ""}


def save_beliefs(data: Dict[str, Any]) -> None:
    data["updated_at"] = str(int(time.time()))
    with open(_beliefs_file(), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", (value or "").strip())


def _normalize_slot_value(slot: str, value: str) -> str:
    raw = (value or "").strip(" ，。！？,.!?：:;；")
    if slot == "age":
        match = re.search(r"(\d{1,3})", raw)
        return match.group(1) if match else ""
    if slot == "birthday":
        match = re.search(r"(\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日)", raw)
        return match.group(1) if match else _normalize_text(raw)[:20]
    if slot in ("preference_like", "preference_dislike"):
        raw = raw.strip("的")
    return _normalize_text(raw)[:60]


def _split_preference_values(value: str) -> List[str]:
    raw = (value or "").strip()
    raw = raw.replace("也喜欢", "、").replace("也爱", "、").replace("和", "、").replace("以及", "、")
    parts = re.split(r"[、，,；;]\s*", raw)
    cleaned: List[str] = []
    for part in parts:
        normalized = _normalize_slot_value("preference_like", part)
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned[:5]


def is_question_like(text: str) -> bool:
    content = (text or "").strip()
    if not content:
        return False
    return any(marker in content for marker in QUESTION_MARKERS)


def extract_claims(text: str) -> List[Dict[str, str]]:
    content = (text or "").strip()
    if not content:
        return []
    if is_question_like(content):
        return []

    claims: List[Dict[str, str]] = []
    patterns = [
        ("name", r"(?:我叫|我的名字是)([^，。！？,\n]{1,20})"),
        ("age", r"(?:我(?:今|现)?年|我)(\d{1,3})岁"),
        ("birthday", r"(?:我(?:的)?生日是)([^，。！？,\n]{1,20})"),
        ("birthday", r"(?:我在)(\d{1,2}月\d{1,2}日)(?:出生|生日)"),
        ("location_city", r"(?:我来自)([^，。！？,\n]{1,20})"),
        ("location_area", r"(?:我(?:现在)?住在|现在住在)([^，。！？,\n]{1,20})"),
        ("location_area", r"(?:我在)([^，。！？,\n]{1,20})(?:上班|工作|读书|生活)"),
        ("occupation", r"(?:我是|我现在是)([^，。！？,\n]{1,20})(?:工程师|程序员|老师|学生|设计师|老板|医生|律师|作者|运营|产品经理|开发者)"),
        ("mother_name", r"(?:我(?:妈妈|母亲))(?:叫|是)([^，。！？,\n]{1,20})"),
        ("father_name", r"(?:我(?:爸爸|父亲))(?:叫|是)([^，。！？,\n]{1,20})"),
        ("sibling_name", r"(?:我(?:哥哥|弟弟|姐姐|妹妹))(?:叫|是)([^，。！？,\n]{1,20})"),
        ("preference_like", r"(?:我喜欢|我最喜欢|我爱)([^，。！？,\n]{1,30})"),
        ("preference_like", r"(?:也喜欢|还喜欢|也爱)([^，。！？,\n]{1,30})"),
        ("preference_dislike", r"(?:我讨厌|我不喜欢|我最讨厌)([^，。！？,\n]{1,30})"),
        ("preference_dislike", r"(?:也讨厌|还讨厌|也不喜欢)([^，。！？,\n]{1,30})"),
    ]

    for slot, pattern in patterns:
        for match in re.finditer(pattern, content):
            value = match.group(1) if match.groups() else ""
            if slot == "occupation":
                value = match.group(0).replace("我是", "").replace("我现在是", "")
            if slot in ("preference_like", "preference_dislike"):
                for item in _split_preference_values(value):
                    claims.append({"slot": slot, "value": item})
                continue
            normalized = _normalize_slot_value(slot, value)
            if normalized:
                claims.append({"slot": slot, "value": normalized})

    return claims


def extract_claims_from_fragment(text: str) -> List[Dict[str, str]]:
    content = (text or "").strip()
    if not content:
        return []

    claims: List[Dict[str, str]] = []
    if re.search(r"\d{1,3}\s*岁", content) or re.fullmatch(r"\d{1,3}", content):
        value = _normalize_slot_value("age", content)
        if value:
            claims.append({"slot": "age", "value": value})

    birthday_match = re.search(r"(\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日)", content)
    if birthday_match:
        value = _normalize_slot_value("birthday", birthday_match.group(1))
        if value:
            claims.append({"slot": "birthday", "value": value})

    return claims


def _ensure_slot_entry(data: Dict[str, Any], slot: str) -> Dict[str, Any]:
    slots = data.setdefault("slots", {})
    if slot not in slots:
        slots[slot] = {
            "slot": slot,
            "label": SLOT_CONFIG[slot]["label"],
            "stability": SLOT_CONFIG[slot]["stability"],
            "candidates": [],
            "summary": {},
        }
    return slots[slot]


def _upsert_candidate(slot_entry: Dict[str, Any], value: str, timestamp: str, source_text: str, session_id: Optional[str]) -> None:
    for candidate in slot_entry["candidates"]:
        if candidate.get("value") == value:
            candidate["count"] = candidate.get("count", 0) + 1
            candidate["last_seen"] = timestamp
            candidate["last_source_text"] = source_text[:120]
            if session_id and session_id not in candidate.get("session_ids", []):
                candidate.setdefault("session_ids", []).append(session_id)
            return

    slot_entry["candidates"].append({
        "value": value,
        "count": 1,
        "first_seen": timestamp,
        "last_seen": timestamp,
        "last_source_text": source_text[:120],
        "session_ids": [session_id] if session_id else [],
    })


def _decrement_candidate(slot_entry: Dict[str, Any], value: str, session_id: Optional[str] = None) -> None:
    kept = []
    for candidate in slot_entry.get("candidates", []):
        if candidate.get("value") != value:
            kept.append(candidate)
            continue
        count = int(candidate.get("count", 1) or 1)
        if session_id and session_id in candidate.get("session_ids", []):
            remaining_sessions = [sid for sid in candidate.get("session_ids", []) if sid != session_id]
            candidate["session_ids"] = remaining_sessions
        candidate["count"] = max(0, count - 1)
        if candidate["count"] > 0:
            kept.append(candidate)
    slot_entry["candidates"] = kept


def _candidate_sort_key(candidate: Dict[str, Any]):
    try:
        last_seen = float(candidate.get("last_seen", 0))
    except Exception:
        last_seen = 0
    return (candidate.get("count", 0), last_seen)


def _build_slot_summary(slot_entry: Dict[str, Any]) -> Dict[str, Any]:
    candidates = sorted(slot_entry.get("candidates", []), key=_candidate_sort_key, reverse=True)
    distinct_values = [item.get("value") for item in candidates if item.get("value")]
    slot = slot_entry.get("slot", "")
    stability = slot_entry.get("stability", "medium")

    if not candidates:
        return {"status": "empty", "confidence": "low", "note": ""}

    preferred = candidates[0]
    summary = {
        "status": "stable",
        "confidence": "high",
        "preferred_value": preferred.get("value"),
        "distinct_count": len(distinct_values),
        "candidate_values": distinct_values[:8],
        "note": f"当前更倾向于认为{slot_entry.get('label')}是 {preferred.get('value')}",
    }

    if len(distinct_values) == 1:
        return summary

    if slot in ("preference_like", "preference_dislike"):
        summary["status"] = "stable"
        summary["confidence"] = "high" if len(distinct_values) <= 3 else "medium"
        summary["preferred_value"] = " / ".join(distinct_values[:5])
        summary["note"] = f"{slot_entry.get('label')}可以有多个并存项：{' / '.join(distinct_values[:5])}"
        return summary

    summary["status"] = "conflicted"
    summary["confidence"] = "medium" if len(distinct_values) == 2 else "low"
    summary["note"] = f"{slot_entry.get('label')}出现多个说法：{' / '.join(distinct_values[:5])}"

    if stability == "high" and len(distinct_values) >= 3:
        summary["status"] = "unstable"
        summary["confidence"] = "low"
        summary["preferred_value"] = preferred.get("value")
        summary["note"] = (
            f"{slot_entry.get('label')}多次出现冲突说法：{' / '.join(distinct_values[:5])}。"
            "该信息不应作为强事实使用，用户可能在测试、玩梗或故意误导。"
        )

    if stability == "medium" and len(distinct_values) >= 3:
        summary["status"] = "dynamic"
        summary["confidence"] = "low"
        summary["note"] = (
            f"{slot_entry.get('label')}经常变化：{' / '.join(distinct_values[:5])}。"
            "这类偏好可能是阶段性变化，不宜当成长期固定事实。"
        )

    return summary


def _is_dirty_belief_candidate(slot: str, value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return True
    if "?" in text or "？" in text:
        return True
    if "，" in text or "," in text:
        return True
    if "什么" in text and slot in ("name", "birthday", "location_city", "location_area", "mother_name", "father_name", "sibling_name"):
        return True
    if len(text) > 20:
        return True
    return False


def cleanup_beliefs() -> Dict[str, Any]:
    data = load_beliefs()
    slots = data.get("slots", {})
    removed = 0
    migrated = 0

    legacy_slot_mapping = {
        "location": "location_city",
        "family_relation": "mother_name",
    }
    for old_slot, new_slot in list(legacy_slot_mapping.items()):
        if old_slot not in slots:
            continue
        old_entry = slots.pop(old_slot)
        new_entry = _ensure_slot_entry(data, new_slot)
        for candidate in old_entry.get("candidates", []):
            if _is_dirty_belief_candidate(new_slot, candidate.get("value", "")):
                removed += 1
                continue
            if candidate not in new_entry["candidates"]:
                new_entry["candidates"].append(candidate)
                migrated += 1
        new_entry["summary"] = _build_slot_summary(new_entry)

    for slot, slot_entry in list(slots.items()):
        if slot not in SLOT_CONFIG:
            slots.pop(slot, None)
            removed += len(slot_entry.get("candidates", []))
            continue
        candidates = slot_entry.get("candidates", [])
        cleaned = []
        for candidate in candidates:
            if _is_dirty_belief_candidate(slot, candidate.get("value", "")):
                removed += 1
                continue
            cleaned.append(candidate)
        slot_entry["candidates"] = cleaned
        slot_entry["summary"] = _build_slot_summary(slot_entry)
        if not cleaned:
            slots.pop(slot, None)
    save_beliefs(data)
    return {"removed": removed, "migrated": migrated, "slots": len(slots)}


async def update_beliefs_from_text(
    text: str,
    session_id: Optional[str] = None,
    allow_promote: bool = True,
) -> List[Dict[str, str]]:
    if not allow_promote:
        return []
    claims = extract_claims(text)
    if not claims:
        return []

    data = load_beliefs()
    now = str(int(time.time()))

    for claim in claims:
        slot_entry = _ensure_slot_entry(data, claim["slot"])
        _upsert_candidate(slot_entry, claim["value"], now, text, session_id)
        slot_entry["summary"] = _build_slot_summary(slot_entry)

    save_beliefs(data)
    return claims


async def correct_beliefs_from_text(
    old_text: str,
    new_text: str,
    session_id: Optional[str] = None,
    allow_promote: bool = True,
) -> Dict[str, int]:
    if not allow_promote:
        return {"removed": 0, "added": 0}

    old_claims = extract_claims(old_text) or extract_claims_from_fragment(old_text)
    new_claims = extract_claims(new_text) or extract_claims_from_fragment(new_text)
    if not old_claims and not new_claims:
        return {"removed": 0, "added": 0}

    data = load_beliefs()
    now = str(int(time.time()))
    removed = 0
    added = 0

    for claim in old_claims:
        slot_entry = data.get("slots", {}).get(claim["slot"])
        if not slot_entry:
            continue
        before = len(slot_entry.get("candidates", []))
        _decrement_candidate(slot_entry, claim["value"], session_id=session_id)
        after = len(slot_entry.get("candidates", []))
        if before != after or before:
            removed += 1
        slot_entry["summary"] = _build_slot_summary(slot_entry)
        if not slot_entry.get("candidates"):
            data.get("slots", {}).pop(claim["slot"], None)

    for claim in new_claims:
        slot_entry = _ensure_slot_entry(data, claim["slot"])
        _upsert_candidate(slot_entry, claim["value"], now, new_text, session_id)
        slot_entry["summary"] = _build_slot_summary(slot_entry)
        added += 1

    save_beliefs(data)
    return {"removed": removed, "added": added}


def get_belief_context_text() -> str:
    data = load_beliefs()
    slots = data.get("slots", {})
    if not slots:
        return "(暂无结构化事实候选)"

    strong_lines: List[str] = []
    caution_lines: List[str] = []

    for slot in SLOT_CONFIG:
        slot_entry = slots.get(slot)
        if not slot_entry:
            continue
        summary = slot_entry.get("summary", {})
        label = slot_entry.get("label", slot)
        status = summary.get("status")
        confidence = summary.get("confidence", "low")
        preferred = summary.get("preferred_value")
        if status == "stable" and preferred:
            strong_lines.append(f"- {label}: {preferred} (置信{confidence})")
        elif status in ("conflicted", "unstable", "dynamic"):
            caution_lines.append(f"- {label}: {summary.get('note', '')}")

    lines: List[str] = []
    if strong_lines:
        lines.append("【高可信事实】")
        lines.extend(strong_lines[:6])
    if caution_lines:
        lines.append("【谨慎信息】")
        lines.extend(caution_lines[:6])

    return "\n".join(lines) if lines else "(暂无结构化事实候选)"

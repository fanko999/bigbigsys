"""
完整的记忆系统 - 保持原有检索/评分逻辑，改为按角色隔离存储
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from role_context import ensure_role_storage, get_current_role_config, get_role_paths

MEMORY_TYPES = {
    "user_preference": {"weight": 0.9, "desc": "用户偏好", "half_life": 180},
    "user_info": {"weight": 0.9, "desc": "用户信息", "half_life": 365},
    "project": {"weight": 0.8, "desc": "项目信息", "half_life": 90},
    "goal": {"weight": 0.7, "desc": "目标计划", "half_life": 60},
    "context": {"weight": 0.5, "desc": "工作上下文", "half_life": 7},
    "reflection": {"weight": 0.6, "desc": "反思总结", "half_life": 30},
}


def _memory_paths() -> Dict[str, Path]:
    paths = ensure_role_storage()
    memory_dir = paths["memory"]
    memory_file = memory_dir / "memories.json"
    if not memory_file.exists():
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
    return {
        "dir": memory_dir,
        "file": memory_file,
        "index": memory_dir / "memory.index",
        "ids": memory_dir / "memory_ids.json",
    }


def load_memories() -> List[Dict[str, Any]]:
    memory_file = _memory_paths()["file"]
    try:
        with open(memory_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_memories(memories: List[Dict[str, Any]]) -> None:
    memory_file = _memory_paths()["file"]
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)


def get_half_life_time_weight(created_at: float, memory_type: str) -> float:
    now = time.time()
    days_ago = (now - created_at) / 86400
    half_life = MEMORY_TYPES.get(memory_type, {}).get("half_life", 30)
    time_weight = 2 ** (-days_ago / half_life)
    return max(0.1, min(1.0, time_weight))


def get_embedding(text: str) -> List[float]:
    if not text:
        return []

    role_config = get_current_role_config()
    url = f"{role_config.get('embedding_host', 'http://127.0.0.1:11435').rstrip('/')}/api/embeddings"
    payload = {
        "model": role_config.get("embedding_model", "bge-large"),
        "prompt": text[:8000],
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json().get("embedding", [])
    except Exception as e:
        print(f"向量获取失败: {e}")
    return []


def should_save(content: str) -> bool:
    if len(content) < 5:
        return False
    important_keywords = ["我叫", "我姓", "fk", "喜欢", "讨厌", "爱好", "项目", "任务", "目标", "计划", "最爱"]
    content_lower = content.lower()
    if any(keyword in content_lower for keyword in important_keywords):
        return True
    return len(content) > 10


def get_memory_type(content: str) -> tuple[str, float]:
    if any(kw in content for kw in ["我喜欢", "我讨厌", "我爱", "我的爱好"]):
        return ("user_preference", 0.9)
    if any(kw in content for kw in ["我叫", "我姓", "我来自"]):
        return ("user_info", 0.9)
    if any(kw in content for kw in ["项目", "代码", "开发", "任务", "工作"]):
        return ("project", 0.8)
    if any(kw in content for kw in ["目标", "计划", "想", "要", "打算"]):
        return ("goal", 0.7)
    return ("context", 0.5)


async def search_memories(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not query:
        return []

    paths = _memory_paths()
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []

    try:
        import faiss

        if paths["index"].exists() and paths["ids"].exists():
            index = faiss.read_index(str(paths["index"]))
            with open(paths["ids"], "r", encoding="utf-8") as f:
                memory_ids = json.load(f)
            if index.ntotal > 0:
                v = np.array([query_embedding], dtype="float32")
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
                search_k = min(top_k * 4, index.ntotal)
                scores, indices = index.search(v, search_k)
                all_memories = load_memories()
                memory_map = {m.get("id"): m for m in all_memories}

                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= len(memory_ids):
                        continue
                    mem_id = memory_ids[idx]
                    mem = memory_map.get(mem_id)
                    if not mem:
                        continue
                    importance = mem.get("importance", 0.5)
                    mem_type = mem.get("type", "context")
                    type_weight = MEMORY_TYPES.get(mem_type, {}).get("weight", 0.5)
                    created = _coerce_timestamp(mem.get("created_at", 0))
                    time_weight = get_half_life_time_weight(created, mem_type)
                    hit_count = mem.get("hit_count", 0)
                    hit_weight = min(0.2, hit_count * 0.02)
                    final_score = score * 0.4 + importance * 0.2 + type_weight * 0.2 + time_weight * 0.1 + hit_weight * 0.1
                    results.append({
                        "id": mem.get("id"),
                        "content": mem.get("content"),
                        "type": mem_type,
                        "importance": importance,
                        "score": float(final_score),
                        "vector_score": float(score),
                        "created_at": mem.get("created_at"),
                    })
                results.sort(key=lambda item: item["score"], reverse=True)
                return results[:top_k]
    except Exception as e:
        print(f"[Faiss] 索引搜索失败: {e}")

    results = []
    memories = load_memories()
    for mem in memories:
        if not mem.get("embedding"):
            continue
        try:
            vector_score = float(np.dot(query_embedding, mem["embedding"]))
            importance = mem.get("importance", 0.5)
            mem_type = mem.get("type", "context")
            type_weight = MEMORY_TYPES.get(mem_type, {}).get("weight", 0.5)
            created = _coerce_timestamp(mem.get("created_at", 0))
            time_weight = get_half_life_time_weight(created, mem_type)
            hit_count = mem.get("hit_count", 0)
            hit_weight = min(0.2, hit_count * 0.02)
            final_score = vector_score * 0.4 + importance * 0.2 + type_weight * 0.2 + time_weight * 0.1 + hit_weight * 0.1
            results.append({
                "id": mem.get("id"),
                "content": mem.get("content"),
                "type": mem_type,
                "importance": importance,
                "score": float(final_score),
                "vector_score": float(vector_score),
                "created_at": mem.get("created_at"),
            })
        except Exception:
            continue

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


async def search_memories_multi_hop(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    first_hop = await search_memories(query, top_k=3)
    if not first_hop:
        return []

    keywords: List[str] = []
    for mem in first_hop:
        content = mem.get("content", "")
        if len(content) > 10:
            words = content.replace("我", "").replace("的", "").replace("是", "").replace("喜欢", "").replace("叫", "").split()
            keywords.extend(words[:2])

    second_hop: List[Dict[str, Any]] = []
    if keywords:
        second_hop = await search_memories(" ".join(keywords[:5]), top_k=3)

    combined = {m["id"]: m for m in first_hop}
    for mem in second_hop:
        combined.setdefault(mem["id"], mem)

    results = list(combined.values())
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


async def add_memory(
    content: str,
    memory_type: str = "context",
    importance: float = 0.5,
    tags: Optional[List[str]] = None,
    auto_detect: bool = True,
    force: bool = False,
):
    if not force and not should_save(content):
        return None

    if auto_detect:
        memory_type, importance = get_memory_type(content)

    content = content[:300]
    embedding = get_embedding(content)
    if not embedding:
        return None

    now = str(int(time.time()))
    memories = load_memories()
    for mem in memories:
        if mem.get("content", "")[:30] == content[:30]:
            mem["content"] = content
            mem["last_hit"] = now
            mem["hit_count"] = mem.get("hit_count", 0) + 1
            mem["updated_at"] = now
            save_memories(memories)
            return mem

    memory = {
        "id": f"mem_{int(time.time() * 1000)}",
        "content": content,
        "type": memory_type,
        "importance": importance,
        "embedding": embedding,
        "tags": tags or [],
        "created_at": now,
        "last_hit": now,
        "hit_count": 0,
    }
    memories.append(memory)
    save_memories(memories)
    return memory


async def get_all_memories(limit: int = 100) -> List[Dict[str, Any]]:
    memories = load_memories()
    memories.sort(key=lambda item: item.get("importance", 0), reverse=True)
    return memories[:limit]


async def delete_memory(memory_id: str):
    memories = [mem for mem in load_memories() if mem.get("id") != memory_id]
    save_memories(memories)


async def correct_memory(old_content: str, new_content: str) -> bool:
    memories = load_memories()
    for mem in memories:
        if old_content in mem.get("content", ""):
            mem["content"] = new_content[:300]
            embedding = get_embedding(new_content)
            if embedding:
                mem["embedding"] = embedding
            mem["updated_at"] = str(int(time.time()))
            mem["corrected"] = True
            save_memories(memories)
            return True
    return False


async def clear_memories():
    save_memories([])


async def get_memory_stats() -> Dict[str, Any]:
    memories = load_memories()
    stats: Dict[str, Any] = {"total": len(memories), "by_type": {}, "avg_importance": 0}
    total_importance = 0.0
    for mem in memories:
        mem_type = mem.get("type", "unknown")
        stats["by_type"][mem_type] = stats["by_type"].get(mem_type, 0) + 1
        total_importance += mem.get("importance", 0)
    if memories:
        stats["avg_importance"] = total_importance / len(memories)
    return stats


async def check_critical_memories() -> Optional[str]:
    memories = load_memories()
    if not memories:
        return None

    now = time.time()
    warnings = []
    for mem in memories:
        importance = mem.get("importance", 0.5)
        if importance < 0.7:
            continue
        created = _coerce_timestamp(mem.get("created_at", 0))
        mem_type = mem.get("type", "context")
        half_life = MEMORY_TYPES.get(mem_type, {}).get("half_life", 30)
        days_ago = (now - created) / 86400
        current_weight = 2 ** (-days_ago / half_life)
        if current_weight < 0.15 and mem.get("hit_count", 0) < 2:
            warnings.append({
                "content": mem.get("content", "")[:30],
                "days_ago": int(days_ago),
            })

    if not warnings:
        return None

    lines = ["我想提醒你一些事："]
    for warning in warnings[:3]:
        lines.append(f"- {warning['content']}... (已过{warning['days_ago']}天)")
    lines.append("")
    lines.append("这些记忆对我很重要，不想忘记...")
    return "\n".join(lines)


def _coerce_timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return 0.0

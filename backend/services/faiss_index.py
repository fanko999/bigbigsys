"""
Faiss向量索引模块 - 按角色隔离
"""
import json
from pathlib import Path

import faiss
import numpy as np

from role_context import ensure_role_storage


def _paths():
    paths = ensure_role_storage()
    return {
        "memory_file": paths["memory"] / "memories.json",
        "index_file": paths["memory"] / "memory.index",
        "ids_file": paths["memory"] / "memory_ids.json",
    }


def load_index():
    dimension = 1024
    paths = _paths()
    if paths["index_file"].exists() and paths["ids_file"].exists():
        try:
            index = faiss.read_index(str(paths["index_file"]))
            with open(paths["ids_file"], "r", encoding="utf-8") as f:
                memory_ids = json.load(f)
            return index, memory_ids
        except Exception as e:
            print(f"[Faiss] 加载失败: {e}")
    return faiss.IndexFlatIP(dimension), []


def save_index(index, memory_ids):
    paths = _paths()
    faiss.write_index(index, str(paths["index_file"]))
    with open(paths["ids_file"], "w", encoding="utf-8") as f:
        json.dump(memory_ids, f, ensure_ascii=False)


def build_index():
    dimension = 1024
    paths = _paths()
    if not paths["memory_file"].exists():
        return

    with open(paths["memory_file"], "r", encoding="utf-8") as f:
        memories = json.load(f)
    if not memories:
        return

    index = faiss.IndexFlatIP(dimension)
    memory_ids = []
    vectors = []
    for mem in memories:
        vec = mem.get("embedding", [])
        if len(vec) != dimension:
            continue
        v = np.array(vec, dtype="float32")
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        vectors.append(v)
        memory_ids.append(mem.get("id"))

    if vectors:
        index.add(np.array(vectors))
        save_index(index, memory_ids)


def search(query_vector, top_k=5, memories=None):
    index, memory_ids = load_index()
    if index.ntotal == 0:
        return []

    v = np.array(query_vector, dtype="float32")
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    v = v.reshape(1, -1)

    scores, indices = index.search(v, min(top_k * 2, index.ntotal))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(memory_ids):
            continue
        mem_id = memory_ids[idx]
        if memories:
            for mem in memories:
                if mem.get("id") == mem_id:
                    results.append({"id": mem_id, "content": mem.get("content"), "score": float(score)})
                    break
        else:
            results.append({"id": mem_id, "score": float(score)})
    return results[:top_k]

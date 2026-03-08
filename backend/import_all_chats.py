"""
全量导入聊天记录到向量库
"""
import os
import json
import time
import requests
import numpy as np
from pathlib import Path

# 路径
SESSIONS_DIR = Path(__file__).parent.parent / "data" / "sessions"
MEMORY_FILE = Path(__file__).parent.parent / "data" / "memory" / "memories.json"

OLLAMA_HOST = "http://127.0.0.1:11435"
MODEL = "bge-large"

def get_embedding(text):
    """获取向量"""
    if not text:
        return []
    url = f"{OLLAMA_HOST}/api/embeddings"
    payload = {"model": MODEL, "prompt": text[:8000]}
    try:
        r = requests.post(url, json=payload, timeout=120)
        if r.status_code == 200:
            return r.json().get("embedding", [])
    except Exception as e:
        print(f"向量获取失败: {e}")
    return []

def get_memory_type(content):
    """自动判断记忆类型"""
    if "我叫" in content or "我姓" in content:
        return ("user_info", 0.9)
    if "喜欢" in content or "讨厌" in content or "我爱" in content:
        return ("user_preference", 0.9)
    if "项目" in content or "代码" in content or "开发" in content:
        return ("project", 0.8)
    if "目标" in content or "计划" in content:
        return ("goal", 0.7)
    return ("context", 0.5)

def import_all():
    """全量导入"""
    print("=" * 50)
    print("开始全量导入聊天记录")
    print("=" * 50)
    
    # 读取现有记忆
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except:
        existing = []
    
    existing_contents = {m.get("content", "")[:30] for m in existing}
    print(f"现有记忆: {len(existing)}条")
    
    # 获取所有会话
    session_files = list(SESSIONS_DIR.glob("*.json"))
    print(f"会话文件: {len(session_files)}个")
    
    new_count = 0
    skip_count = 0
    
    for session_file in session_files:
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                session = json.load(f)
            
            for msg in session.get("messages", []):
                if msg.get("role") != "user":
                    continue
                
                content = msg.get("content", "").strip()
                if len(content) < 5:
                    continue
                
                # 检查是否已存在
                if content[:30] in existing_contents:
                    skip_count += 1
                    continue
                
                # 获取向量
                embedding = get_embedding(content)
                if not embedding:
                    continue
                
                # 判断类型
                mem_type, importance = get_memory_type(content)
                
                # 创建记忆
                now = str(int(time.time()))
                memory = {
                    "id": f"mem_{now}_{new_count}",
                    "content": content[:300],
                    "type": mem_type,
                    "importance": importance,
                    "embedding": embedding,
                    "tags": [],
                    "created_at": now,
                    "last_hit": now,
                    "hit_count": 0
                }
                
                existing.append(memory)
                existing_contents.add(content[:30])
                new_count += 1
                
                if new_count % 50 == 0:
                    print(f"已处理: {new_count}条新记忆...")
                    
        except Exception as e:
            print(f"处理会话失败: {session_file.name} - {e}")
    
    # 保存
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    
    print("=" * 50)
    print(f"导入完成!")
    print(f"新增记忆: {new_count}条")
    print(f"跳过重复: {skip_count}条")
    print(f"总记忆: {len(existing)}条")
    print("=" * 50)

if __name__ == "__main__":
    import_all()

# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"C:\Users\Administrator\.openclaw\workspace\ai-web-chat\backend")
import asyncio
from services.memory_service import get_embedding, add_memory, search_memories

async def test():
    # 测试向量获取
    print("测试1: 获取向量")
    emb = await get_embedding("你好，我是小猪仔")
    print(f"向量维度: {len(emb)}")
    
    # 添加测试记忆
    print("\n测试2: 添加记忆")
    await add_memory(
        content="用户叫小猪仔，他喜欢和AI讨论哲学问题",
        memory_type="user_preference",
        importance=0.8,
        tags=["用户", "偏好"]
    )
    print("记忆已添加")
    
    # 检索
    print("\n测试3: 检索记忆")
    results = await search_memories("用户信息")
    print(f"检索到 {len(results)} 条记忆")
    for r in results:
        print(f"  - {r['content'][:50]}... (相似度: {r['score']:.3f})")

asyncio.run(test())

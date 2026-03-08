# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"C:\Users\Administrator\.openclaw\workspace\ai-web-chat\backend")
import asyncio
from services.memory_service import search_memories, get_all_memories

async def test():
    # 检索测试
    print("=== 检索测试 ===")
    results = await search_memories("用户叫什么名字", top_k=5)
    print(f"找到 {len(results)} 条记忆")
    for r in results:
        print(f"  - {r['content'][:50]}... (相似度: {r['score']:.3f})")
    
    print("\n=== 全部记忆 ===")
    all_mems = await get_all_memories()
    print(f"共 {len(all_mems)} 条记忆")
    for m in all_mems[:5]:
        print(f"  - {m.get('content', '')[:50]}...")

asyncio.run(test())

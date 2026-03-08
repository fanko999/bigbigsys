# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import asyncio
from services.memory_service import search_memories

async def test():
    print("=== 测试检索: 你叫什么 ===")
    results = await search_memories("你叫什么", top_k=5)
    print(f"找到 {len(results)} 条")
    for r in results:
        print(f"  - {r.get('content', '')[:40]} (score: {r.get('score', 0):.3f})")
    
    print("\n=== 测试检索: 喜欢吃什么 ===")
    results = await search_memories("喜欢吃什么", top_k=5)
    print(f"找到 {len(results)} 条")
    for r in results:
        print(f"  - {r.get('content', '')[:40]} (score: {r.get('score', 0):.3f})")

asyncio.run(test())

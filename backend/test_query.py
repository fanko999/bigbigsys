# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import asyncio
import sys
sys.path.insert(0, r"C:\Users\Administrator\.openclaw\workspace\ai-web-chat\backend")

from services.memory_service import search_memories

async def test():
    r = await search_memories("你还记得我叫什么", top_k=5)
    print(f"Found: {len(r)}")
    for m in r:
        print(f"  {m['score']:.3f} - {m['content'][:40]}")

asyncio.run(test())

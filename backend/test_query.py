# -*- coding: utf-8 -*-
import sys
import io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import asyncio
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.memory_service import search_memories

async def test():
    r = await search_memories("你还记得我叫什么", top_k=5)
    print(f"Found: {len(r)}")
    for m in r:
        print(f"  {m['score']:.3f} - {m['content'][:40]}")

asyncio.run(test())

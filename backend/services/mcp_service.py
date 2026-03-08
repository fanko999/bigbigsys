"""
MiniMax MCP 客户端 - 支持看图和搜索
"""
import asyncio
import os
import json
from typing import List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MiniMaxMCP:
    """MiniMax MCP 客户端"""
    
    def __init__(self):
        self.api_key = os.environ.get("MINIMAX_API_KEY", "")
        self.session = None
        self.initialized = False
    
    async def connect(self):
        """连接MCP服务"""
        if self.initialized:
            return
        
        server_params = StdioServerParameters(
            command="uvx",
            args=["minimax-coding-plan-mcp", "-y"],
            env={
                **os.environ,
                "MINIMAX_API_KEY": self.api_key,
                "MINIMAX_API_HOST": "https://api.minimaxi.com"
            }
        )
        
        self.read, self.write = await stdio_client(server_params).__aenter__()
        self.session = ClientSession(self.read, self.write)
        await self.session.initialize()
        self.initialized = True
        print("[MCP] 连接成功!")
    
    async def understand_image(self, image_path: str, prompt: str = "请描述这张图片") -> str:
        """理解图片"""
        await self.connect()
        
        result = await self.session.call_tool(
            "understand_image",
            arguments={
                "prompt": prompt,
                "image_source": image_path
            }
        )
        
        for content in result.content:
            if content.type == "text":
                return content.text
        return "无法理解图片"
    
    async def web_search(self, query: str) -> str:
        """网络搜索"""
        await self.connect()
        
        result = await self.session.call_tool(
            "web_search",
            arguments={"query": query}
        )
        
        for content in result.content:
            if content.type == "text":
                return content.text
        return "搜索失败"
    
    async def close(self):
        """关闭连接"""
        if self.session:
            await self.session.close()
            self.initialized = False

# 全局实例
_mcp_client = None

async def get_mcp_client() -> MiniMaxMCP:
    """获取MCP客户端"""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MiniMaxMCP()
    return _mcp_client

async def understand_image(image_path: str, prompt: str = "请描述这张图片") -> str:
    """理解图片的便捷函数"""
    client = await get_mcp_client()
    return await client.understand_image(image_path, prompt)

async def web_search(query: str) -> str:
    """网络搜索的便捷函数"""
    client = await get_mcp_client()
    return await client.web_search(query)

if __name__ == "__main__":
    async def test():
        print("=== 测试MiniMax MCP ===")
        
        # 测试看图
        image_path = r"C:\Users\Administrator\.openclaw\media\inbound\file_1---eb4eb128-7c13-4024-b3ca-da84c6f55651.jpg"
        print(f"测试图片: {image_path}")
        
        result = await understand_image(image_path, "请描述这张图片")
        print(f"\n图片理解结果:\n{result}")
    
    asyncio.run(test())

# -*- coding: utf-8 -*-
import subprocess
import os
import json

UVX_PATH = os.environ.get("UVX_PATH", "uvx")
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_API_HOST = os.environ.get("MINIMAX_API_HOST", "https://api.minimaxi.com")

def run_mcp(query: str, image_path: str = None, api_key: str = "", api_host: str = "") -> str:
    env = os.environ.copy()
    env["MINIMAX_API_KEY"] = api_key or MINIMAX_API_KEY
    env["MINIMAX_API_HOST"] = api_host or MINIMAX_API_HOST
    
    if image_path:
        # 使用正确的路径格式
        image_path = image_path.replace("\\", "/")
        input_data = (
            b'{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"pig","version":"1.0"}}}\n'
            b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"understand_image","arguments":{"prompt":"'
            + query.encode() + b'","image_source":"'
            + image_path.encode() + b'"}}}\n'
        )
    else:
        input_data = (
            b'{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"pig","version":"1.0"}}}\n'
            b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"web_search","arguments":{"query":"'
            + query.encode() + b'"}}}\n'
        )
    
    try:
        result = subprocess.run(
            [UVX_PATH, "minimax-coding-plan-mcp", "-y"],
            input=input_data,
            capture_output=True,
            env=env,
            timeout=120
        )
    except FileNotFoundError:
        return "处理失败: 未找到 uvx，请先安装 uvx 或设置 UVX_PATH"
    
    if result.stdout:
        for line in result.stdout.split(b"\n"):
            if b'"text"' in line:
                try:
                    data = json.loads(line)
                    if "result" in data:
                        content = data["result"].get("content", [])
                        for c in content:
                            if c.get("type") == "text":
                                return c.get("text", "")
                except:
                    pass
    
    return "处理失败"

def web_search(query: str, api_key: str = "", api_host: str = "") -> str:
    return run_mcp(query, api_key=api_key, api_host=api_host)

def understand_image(prompt: str, image_source: str, api_key: str = "", api_host: str = "") -> str:
    return run_mcp(prompt, image_source, api_key=api_key, api_host=api_host)

if __name__ == "__main__":
    print("=== Test MCP ===")
    
    # Test search
    print("\n1. Search:")
    result = web_search("AI trends")
    print(result[:500])
    
    # Test image
    print("\n2. Image:")
    img = os.environ.get("TEST_IMAGE_PATH", "")
    if img:
        result = understand_image("describe this image", img)
        print(result[:500])
    else:
        print("Set TEST_IMAGE_PATH to test image understanding.")

"""
LLM服务 - Ollama调用
支持文本聊天和多模态(图片)
"""
import json
import base64
import aiohttp
from typing import List, Dict, Any, Optional
import config

async def chat_with_ollama(
    message: str,
    model: str,
    ollama_host: str,
    history: List[Dict[str, Any]] = None,
    system_prompt: str = None,
    images: List[str] = None  # 图片base64列表
) -> str:
    """
    调用Ollama API进行聊天
    
    Args:
        message: 用户消息
        model: 模型名称
        ollama_host: Ollama主机地址
        history: 历史消息列表
        system_prompt: 系统提示词
        images: 图片base64列表
    
    Returns:
        AI回复内容
    """
    # 构建消息列表
    messages = []
    
    # 添加系统提示词
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    elif config.SYSTEM_PROMPT:
        messages.append({
            "role": "system",
            "content": config.SYSTEM_PROMPT
        })
    
    # 添加历史消息
    if history:
        for msg in history:
            if msg.get("role") in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
    
    # 添加当前消息（支持多模态）
    if images:
        # 多模态消息
        user_msg = {
            "role": "user",
            "content": message,
            "images": images  # base64编码的图片列表
        }
    else:
        user_msg = {
            "role": "user",
            "content": message
        }
    messages.append(user_msg)
    
    # 构建请求
    url = f"{ollama_host}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("message", {}).get("content", "无回复")
                else:
                    error_text = await response.text()
                    return f"错误: {response.status} - {error_text}"
    except Exception as e:
        return f"调用失败: {str(e)}"

async def chat_stream_with_ollama(
    message: str,
    model: str,
    ollama_host: str,
    history: List[Dict[str, Any]] = None,
    system_prompt: str = None,
    images: List[str] = None,
):
    """
    流式调用Ollama API
    
    Yields:
        生成的文本片段
    """
    # 构建消息列表
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    elif config.SYSTEM_PROMPT:
        messages.append({"role": "system", "content": config.SYSTEM_PROMPT})
    
    if history:
        for msg in history:
            if msg.get("role") in ["user", "assistant"]:
                messages.append({"role": msg["role"], "content": msg["content"]})
    
    if images:
        messages.append({"role": "user", "content": message, "images": images})
    else:
        messages.append({"role": "user", "content": message})
    
    url = f"{ollama_host}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                content = data.get("message", {}).get("content", "")
                                if content:
                                    yield content
                            except:
                                continue
                else:
                    yield f"错误: {response.status}"
    except Exception as e:
        yield f"调用失败: {str(e)}"

async def check_ollama_status(ollama_host: str) -> Dict[str, Any]:
    """检查Ollama状态"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ollama_host}/api/tags", timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    models = await response.json()
                    return {
                        "status": "online",
                        "models": models.get("models", []),
                        "host": ollama_host
                    }
                else:
                    return {"status": "error", "host": ollama_host, "error": f"HTTP {response.status}"}
    except Exception as e:
        return {"status": "offline", "host": ollama_host, "error": str(e)}

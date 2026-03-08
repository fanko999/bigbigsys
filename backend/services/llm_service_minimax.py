"""
MiniMax LLM 服务 - 使用MiniMax API (支持看图)
"""
import json
import base64
import requests
from typing import List, Dict, Any, Optional

# 使用M2.5极速版，更快
DEFAULT_MODEL = "MiniMax-M2.5-highspeed"

def encode_image_to_base64(image_path: str) -> str:
    """将图片转换为base64"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"图片编码失败: {e}")
        return None

def build_message_with_images(message: str, images: List[str] = None) -> List[Dict]:
    """构建支持图片的消息"""
    if not images:
        return [{"type": "text", "text": message}]
    
    # MiniMax支持多图，需要用特定格式
    content = []
    
    # 添加文本
    content.append({
        "type": "text",
        "text": message
    })
    
    # 添加图片
    for img_path in images:
        # 支持URL或本地路径
        if img_path.startswith("http"):
            # 如果是URL，直接用URL
            content.append({
                "type": "image_url",
                "image_url": {"url": img_path}
            })
        else:
            # 本地图片转base64
            b64 = encode_image_to_base64(img_path)
            if b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
    
    return content

def chat_with_minimax(
    message: str,
    model: str = DEFAULT_MODEL,
    history: List[Dict] = None,
    system_prompt: str = None,
    images: List[str] = None,
    api_key: str = "",
    base_url: str = "https://api.minimaxi.com/v1",
    temperature: float = 0.7,
) -> str:
    """调用MiniMax API进行聊天(支持图片)"""
    
    # 构建消息列表
    messages = []
    
    # 添加系统提示
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # 添加历史记录
    if history:
        for msg in history[-10:]:  # 最近10条
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
    
    # 添加当前消息(支持图片)
    if images:
        # 多模态消息
        user_content = build_message_with_images(message, images)
        messages.append({
            "role": "user",
            "content": user_content
        })
    else:
        messages.append({
            "role": "user",
            "content": message
        })
    
    # API调用
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": temperature,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
        return f"API错误: {response.status_code} - {response.text[:200]}"
    except Exception as e:
        return f"调用失败: {str(e)}"

async def chat_with_minimax_async(
    message: str,
    model: str = DEFAULT_MODEL,
    history: List[Dict] = None,
    system_prompt: str = None,
    images: List[str] = None,
    api_key: str = "",
    base_url: str = "https://api.minimaxi.com/v1",
    temperature: float = 0.7,
) -> str:
    """异步调用MiniMax API(支持图片)"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        chat_with_minimax,
        message,
        model,
        history,
        system_prompt,
        images,
        api_key,
        base_url,
        temperature,
    )

def stream_chat_with_minimax(
    message: str,
    model: str = DEFAULT_MODEL,
    history: List[Dict] = None,
    system_prompt: str = None,
    images: List[str] = None,
    api_key: str = "",
    base_url: str = "https://api.minimaxi.com/v1",
    temperature: float = 0.7,
):
    """流式调用MiniMax API(支持图片)"""
    
    # 构建消息列表
    messages = []
    
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    if history:
        for msg in history[-10:]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
    
    # 支持图片
    if images:
        user_content = build_message_with_images(message, images)
        messages.append({
            "role": "user",
            "content": user_content
        })
    else:
        messages.append({
            "role": "user",
            "content": message
        })
    
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": temperature,
        "stream": True
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120, stream=True)
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            json_data = json.loads(data)
                            if "choices" in json_data and len(json_data["choices"]) > 0:
                                delta = json_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except:
                            continue
        else:
            yield f"API错误: {response.status_code}"
    except Exception as e:
        yield f"调用失败: {str(e)}"

if __name__ == "__main__":
    # 测试看图功能
    print("=== 测试MiniMax看图 ===")
    
    # 测试文本
    response = chat_with_minimax("你好", api_key="")
    print(f"文本回复: {response[:100]}...")

import json
from typing import Any, Dict, List, Optional

import aiohttp

from services.llm_service import chat_stream_with_ollama, chat_with_ollama
from services.llm_service_minimax import chat_with_minimax_async


async def chat_with_api_compatible_async(
    message: str,
    model: str,
    base_url: str,
    api_key: str,
    history: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
    images: Optional[List[str]] = None,
    temperature: float = 0.7,
) -> str:
    if not base_url or not api_key or not model:
        return "API模型配置不完整"

    base = base_url.rstrip("/")
    url = f"{base}/chat/completions"

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        for msg in history[-10:]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

    if images:
        content: List[Dict[str, Any]] = [{"type": "text", "text": message}]
        for image in images:
            if image.startswith("http") or image.startswith("data:"):
                content.append({"type": "image_url", "image_url": {"url": image}})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": message})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status != 200:
                    return f"API错误: {response.status} - {await response.text()}"
                data = await response.json()
                choices = data.get("choices", [])
                if not choices:
                    return "API未返回内容"
                return choices[0].get("message", {}).get("content", "API未返回内容")
    except Exception as e:
        return f"API调用失败: {e}"


async def generate_chat_response(
    role_config: Dict[str, Any],
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
    images: Optional[List[str]] = None,
    model_override: Optional[str] = None,
) -> str:
    provider = (role_config.get("provider") or "api").lower()
    model = model_override or role_config.get("chat_model") or ""
    temperature = float(role_config.get("temperature", 0.7) or 0.7)

    if provider == "ollama":
        return await chat_with_ollama(
            message=message,
            model=model,
            ollama_host=role_config.get("chat_ollama_host", "http://127.0.0.1:11434"),
            history=history,
            system_prompt=system_prompt,
            images=images,
        )

    if provider == "minimax":
        return await chat_with_minimax_async(
            message=message,
            model=model,
            history=history,
            system_prompt=system_prompt,
            images=images,
            api_key=role_config.get("api_key", ""),
            base_url=role_config.get("api_base_url", "https://api.minimaxi.com/v1"),
            temperature=temperature,
        )

    return await chat_with_api_compatible_async(
        message=message,
        model=model,
        base_url=role_config.get("api_base_url", ""),
        api_key=role_config.get("api_key", ""),
        history=history,
        system_prompt=system_prompt,
        images=images,
        temperature=temperature,
    )


async def stream_chat_with_api_compatible(
    message: str,
    model: str,
    base_url: str,
    api_key: str,
    history: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
    images: Optional[List[str]] = None,
    temperature: float = 0.7,
):
    if not base_url or not api_key or not model:
        yield "API模型配置不完整"
        return

    base = base_url.rstrip("/")
    url = f"{base}/chat/completions"

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        for msg in history[-10:]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

    if images:
        content: List[Dict[str, Any]] = [{"type": "text", "text": message}]
        for image in images:
            if image.startswith("http") or image.startswith("data:"):
                content.append({"type": "image_url", "image_url": {"url": image}})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": message})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status != 200:
                    yield f"API错误: {response.status} - {await response.text()}"
                    return

                async for raw_line in response.content:
                    if not raw_line:
                        continue
                    for line in raw_line.decode("utf-8", errors="ignore").splitlines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            parsed = json.loads(data)
                        except Exception:
                            continue
                        choices = parsed.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
    except Exception as e:
        yield f"API调用失败: {e}"


async def stream_chat_response(
    role_config: Dict[str, Any],
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
    images: Optional[List[str]] = None,
    model_override: Optional[str] = None,
):
    provider = (role_config.get("provider") or "api").lower()
    model = model_override or role_config.get("chat_model") or ""
    temperature = float(role_config.get("temperature", 0.7) or 0.7)

    if provider == "ollama":
        async for chunk in chat_stream_with_ollama(
            message=message,
            model=model,
            ollama_host=role_config.get("chat_ollama_host", "http://127.0.0.1:11434"),
            history=history,
            system_prompt=system_prompt,
            images=images,
        ):
            yield chunk
        return

    async for chunk in stream_chat_with_api_compatible(
        message=message,
        model=model,
        base_url=role_config.get("api_base_url", "https://api.minimaxi.com/v1"),
        api_key=role_config.get("api_key", ""),
        history=history,
        system_prompt=system_prompt,
        images=images,
        temperature=temperature,
    ):
        yield chunk

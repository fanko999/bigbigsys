"""
AI Web Chat 后端入口 - 支持多角色独立记忆/会话系统
"""
import base64
import json
import re
import shutil
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
from role_context import (
    DEFAULT_ROLE_ID,
    ensure_role_storage,
    get_role_paths,
    list_roles,
    load_role_config,
    normalize_role_id,
    role_scope,
    slugify_role_id,
    upsert_role,
)
from services.llm_router import generate_chat_response, stream_chat_response
from services.llm_service import check_ollama_status

app = FastAPI(title="AI Web Chat API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.SERVER["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str
    model: Optional[str] = None
    has_image: Optional[bool] = False


class Session(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[Message] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    role_id: Optional[str] = DEFAULT_ROLE_ID
    model: Optional[str] = None
    images: Optional[List[str]] = None


class SessionCreateRequest(BaseModel):
    role_id: Optional[str] = DEFAULT_ROLE_ID
    title: Optional[str] = "新对话"


class RolePayload(BaseModel):
    id: Optional[str] = None
    name: str
    avatar: Optional[str] = "🤖"
    description: Optional[str] = ""
    system_prompt: Optional[str] = None
    provider: Optional[str] = None
    chat_model: Optional[str] = None
    vision_model: Optional[str] = None
    embedding_model: Optional[str] = None
    chat_ollama_host: Optional[str] = None
    embedding_host: Optional[str] = None
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    analysis_enabled: Optional[bool] = None
    memory_top_k: Optional[int] = None
    history_limit: Optional[int] = None
    temperature: Optional[float] = None


class RoleImportPayload(BaseModel):
    role: Dict[str, Any]
    data: Dict[str, Any] = Field(default_factory=dict)
    overwrite: bool = False


class MemoryUpdatePayload(BaseModel):
    content: str
    type: Optional[str] = None
    importance: Optional[float] = None
    tags: Optional[List[str]] = None


def generate_id() -> str:
    return str(uuid.uuid4())[:8]


def _sessions_dir(role_id: Optional[str]) -> Path:
    return ensure_role_storage(role_id)["sessions"]


def get_sessions(role_id: Optional[str]) -> List[Dict[str, Any]]:
    sessions = []
    for file in _sessions_dir(role_id).glob("*.json"):
        with open(file, "r", encoding="utf-8") as fp:
            sessions.append(json.load(fp))
    return sorted(sessions, key=lambda item: item.get("updated_at", ""), reverse=True)


def get_session(session_id: str, role_id: Optional[str]) -> Optional[Dict[str, Any]]:
    session_file = _sessions_dir(role_id) / f"{session_id}.json"
    if not session_file.exists():
        return None
    with open(session_file, "r", encoding="utf-8") as fp:
        return json.load(fp)


def save_session(session: Dict[str, Any], role_id: Optional[str]) -> None:
    session_file = _sessions_dir(role_id) / f"{session['id']}.json"
    with open(session_file, "w", encoding="utf-8") as fp:
        json.dump(session, fp, ensure_ascii=False, indent=2)


def create_empty_session(role_id: Optional[str], title: str = "新对话") -> Dict[str, Any]:
    now = datetime.now().isoformat()
    session = {
        "id": generate_id(),
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    save_session(session, role_id)
    return session


def maybe_update_session_title(session: Dict[str, Any], user_message: str) -> None:
    if session.get("title") not in ("新对话", "小猪对话"):
        return
    cleaned = user_message.strip().replace("\n", " ")
    if cleaned:
        session["title"] = cleaned[:28]


def build_history_text(messages: List[Dict[str, Any]], limit: int) -> str:
    lines = []
    recent_msgs = messages[-limit:]
    for idx, msg in enumerate(recent_msgs, 1):
        role = "用户" if msg.get("role") == "user" else "AI"
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        # Filter out earlier hallucinated assistant replies about "empty messages"
        # so they do not keep poisoning later turns.
        if msg.get("role") == "assistant" and "空消息" in content:
            continue
        lines.append(f"{idx}. {role}: {content[:180]}")
    return "\n".join(lines)


def build_analysis_prompt(message: str, history_text: str) -> str:
    return f"""你是一个对话分析助手。请分析以下对话信息，输出结构化的分析结果。

一、用户当前消息：
{message}

二、当前对话历史信息（最近20条）：
{history_text or "(暂无历史)"}

请分析以下内容（只输出分析结果，不要有其他内容）：
1. 用户当前消息与历史对话的逻辑关系是什么？（如：延续话题、切换话题、问答、无关）
2. 用户的目的/意图是什么？（如：闲聊、问问题、分享、寻求帮助、执行任务等）
3. 用户状态推测是什么？请保持保守，并给出置信度（低/中/高）
4. 如果有关联的历史，列出具体是哪些对话与当前消息有关联

输出格式：
关系类型：[延续/切换/问答/无关]
用户目的：[闲聊/问问题/分享/寻求帮助/执行任务/其他]
用户状态推测：[可能偏急/可能偏放松/开心/平静/烦恼/工作中/不确定]
置信度：[低/中/高]
关联历史：[例如 1,3,5 或 无]
说明：[一句简短解释]
"""


def build_structured_system_prompt(
    role_config: Dict[str, Any],
    user_message: str,
    analysis_result: str,
    history_text: str,
    memory_items: List[Dict[str, Any]],
) -> str:
    parts = [role_config.get("system_prompt", config.SYSTEM_PROMPT).strip()]
    parts.append(f"一、【用户当前消息】\n{user_message}")
    parts.append(f"二、【分析现状结果】\n{analysis_result or '未分析'}")
    parts.append(f"三、【当前对话历史信息】\n{history_text or '(暂无历史)'}")

    if memory_items:
        memory_lines = []
        for idx, mem in enumerate(memory_items, 1):
            time_info = mem.get("created_at", "")
            memory_lines.append(f"{idx}. [{mem.get('score', 0):.2f}] {mem.get('content', '')} (time={time_info})")
        parts.append("四、【相关长期记忆】\n" + "\n".join(memory_lines))
    else:
        parts.append("四、【相关长期记忆】\n(暂无命中)")

    parts.append(
        "【回答规则】\n"
        "1. 优先回答当前消息。\n"
        "2. 分析结果只作为内部参考，不向用户展示。\n"
        "3. 用户状态和置信度只影响语气，不影响事实判断。\n"
        "4. 如与历史有关联，可引用对话历史中的具体信息。\n"
        "5. 必要时再引用长期记忆，不要强行硬套。\n"
        "6. 回答保持自然、简洁、清楚。\n"
        "7. 只有当【用户当前消息】在去掉空格和换行后确实为空时，才可以说用户发了空消息。\n"
        "8. 如果【用户当前消息】里有实际文字内容，严禁回答用户发了空消息、没说话、空白消息。"
    )
    return "\n\n".join(parts)


def sanitize_assistant_output(text: str) -> str:
    if not text:
        return ""

    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```think[\s\S]*?```", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def sanitize_assistant_output_partial(text: str) -> str:
    if not text:
        return ""

    cleaned = re.sub(r"<think>[\s\S]*?(</think>|$)", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```think[\s\S]*?(```|$)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def detect_correction(message: str) -> Optional[Dict[str, str]]:
    patterns = [
        r"其实我(?:不|没)(?:是|喜欢|想|要)",
        r"记错了[，,]+(?:是|我)",
        r"纠正一下[，,]+",
        r"不是(?:的|吧|啦|哈)",
        r"(?:不|没)(?:对|是|吧)",
        r"重新说(?:一下|)",
        r"错了[，,]+(?:是|我)",
    ]
    if not any(re.search(pattern, message) for pattern in patterns):
        return None

    new_match = re.search(r"[是到为](.+?)(?:。|$)", message)
    if not new_match:
        return {"old": message, "new": message}

    old_match = re.search(r"(?:不是|不喜欢|没说)(.+?)[，,]", message)
    return {
        "old": old_match.group(1).strip() if old_match else message,
        "new": new_match.group(1).strip(),
    }


def save_image_inputs(images: Optional[List[str]], role_id: Optional[str]) -> Optional[List[str]]:
    if not images:
        return None
    prepared = []
    temp_dir = ensure_role_storage(role_id)["temp"]
    for idx, image in enumerate(images):
        if image.startswith("http") or image.startswith("data:"):
            prepared.append(image)
            continue
        if len(image) > 100:
            temp_path = temp_dir / f"upload_{int(datetime.now().timestamp())}_{idx}.jpg"
            with open(temp_path, "wb") as f:
                f.write(base64.b64decode(image))
            prepared.append(str(temp_path))
        else:
            prepared.append(image)
    return prepared


def rebuild_memory_index_safe():
    try:
        from services.faiss_index import build_index

        build_index()
    except Exception as e:
        print(f"[Faiss] 跳过索引重建: {e}")


def export_role_bundle(role_id: str) -> Dict[str, Any]:
    role_id = slugify_role_id(role_id)
    paths = ensure_role_storage(role_id)
    bundle = {
        "version": "1.0",
        "exported_at": datetime.now().isoformat(),
        "role": load_role_config(role_id),
        "data": {
            "sessions": [],
            "memories": [],
            "growth": {},
        },
    }

    for file in paths["sessions"].glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            bundle["data"]["sessions"].append(json.load(f))

    memory_file = paths["memory"] / "memories.json"
    if memory_file.exists():
        with open(memory_file, "r", encoding="utf-8") as f:
            bundle["data"]["memories"] = json.load(f)

    growth = {}
    for name in ("personality.json", "growth_log.json", "stats.json"):
        file = paths["growth"] / name
        if file.exists():
            with open(file, "r", encoding="utf-8") as f:
                growth[name] = json.load(f)
    reflections_file = paths["memory"] / "reflections.json"
    if reflections_file.exists():
        with open(reflections_file, "r", encoding="utf-8") as f:
            growth["reflections.json"] = json.load(f)
    bundle["data"]["growth"] = growth
    return bundle


def import_role_bundle(payload: RoleImportPayload) -> Dict[str, Any]:
    role_config = dict(payload.role)
    role_id = normalize_role_id(
        role_config.get("id"),
        fallback_name=role_config.get("name"),
        allow_default=role_config.get("id") == DEFAULT_ROLE_ID,
    )
    role_paths = get_role_paths(role_id)
    if role_id == DEFAULT_ROLE_ID and not payload.overwrite:
        raise HTTPException(status_code=400, detail="默认角色只能覆盖导入")
    if role_id != DEFAULT_ROLE_ID and role_paths["root"].exists() and not payload.overwrite:
        existing_config = role_paths["config"]
        if existing_config.exists():
            raise HTTPException(status_code=409, detail="角色已存在，请开启覆盖导入或换一个名字")

    role_config["id"] = role_id
    saved_role = upsert_role(role_config, role_id)
    paths = ensure_role_storage(role_id)

    if payload.overwrite:
        for file in paths["sessions"].glob("*.json"):
            file.unlink()

    for session in payload.data.get("sessions", []):
        save_session(session, role_id)

    memories = payload.data.get("memories")
    if memories is not None:
        memory_file = paths["memory"] / "memories.json"
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(memories, f, ensure_ascii=False, indent=2)

    growth = payload.data.get("growth", {})
    for name, data in growth.items():
        target_dir = paths["growth"] if name != "reflections.json" else paths["memory"]
        with open(target_dir / name, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    with role_scope(role_id):
        rebuild_memory_index_safe()
    return saved_role


@app.get("/")
async def root():
    return {"message": "AI Web Chat API", "version": "2.0.0"}


@app.get("/api/roles")
async def api_list_roles():
    return {"roles": list_roles(), "default_role_id": DEFAULT_ROLE_ID}


@app.post("/api/roles")
async def api_create_role(payload: RolePayload):
    role_data = payload.model_dump(exclude_none=True)
    role = upsert_role(role_data)
    return role


@app.get("/api/roles/{role_id}")
async def api_get_role(role_id: str):
    return load_role_config(role_id)


@app.put("/api/roles/{role_id}")
async def api_update_role(role_id: str, payload: RolePayload):
    role_data = payload.model_dump(exclude_none=True)
    role = upsert_role(role_data, role_id)
    return role


@app.delete("/api/roles/{role_id}")
async def api_delete_role(role_id: str):
    role_id = slugify_role_id(role_id)
    if role_id == DEFAULT_ROLE_ID:
        raise HTTPException(status_code=400, detail="默认角色不能删除")

    paths = get_role_paths(role_id)
    if not paths["root"].exists():
        raise HTTPException(status_code=404, detail="角色不存在")

    shutil.rmtree(paths["root"], ignore_errors=True)
    roles = [item for item in list_roles() if item.get("id") not in (DEFAULT_ROLE_ID, role_id)]
    role_index = []
    for item in roles:
        role_index.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "avatar": item.get("avatar", "🤖"),
            "description": item.get("description", ""),
            "provider": item.get("provider", "api"),
        })
    role_index_path = Path(__file__).parent.parent / "data" / "roles.json"
    with open(role_index_path, "w", encoding="utf-8") as f:
        json.dump(role_index, f, ensure_ascii=False, indent=2)
    return {"status": "success"}


@app.get("/api/roles/{role_id}/export")
async def api_export_role(role_id: str):
    return export_role_bundle(role_id)


@app.post("/api/roles/import")
async def api_import_role(payload: RoleImportPayload):
    role = import_role_bundle(payload)
    return {"status": "success", "role": role}


@app.get("/api/config")
async def get_config(role_id: str = Query(DEFAULT_ROLE_ID)):
    role_config = load_role_config(role_id)
    return {
        **role_config,
        "ollama_hosts": config.OLLAMA_HOSTS,
        "default_ollama_host": config.DEFAULT_OLLAMA_HOST,
    }


@app.post("/api/config")
async def set_config(config_data: Dict[str, Any], role_id: str = Query(DEFAULT_ROLE_ID)):
    updated = upsert_role(config_data, role_id)
    return {"status": "success", "config": updated}


@app.get("/api/models")
async def get_models(host: Optional[str] = Query(None)):
    if host:
        return await check_ollama_status(host)

    result = {}
    for name, ollama_host in config.OLLAMA_HOSTS.items():
        result[name] = await check_ollama_status(ollama_host)
    return result


@app.get("/api/sessions")
async def list_sessions(role_id: str = Query(DEFAULT_ROLE_ID)):
    return get_sessions(role_id)


@app.post("/api/sessions")
async def create_session(request: SessionCreateRequest):
    return create_empty_session(request.role_id, request.title or "新对话")


@app.get("/api/sessions/{session_id}")
async def get_session_messages(session_id: str, role_id: str = Query(DEFAULT_ROLE_ID)):
    session = get_session(session_id, role_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, role_id: str = Query(DEFAULT_ROLE_ID)):
    session_file = _sessions_dir(role_id) / f"{session_id}.json"
    if not session_file.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    session_file.unlink()
    return {"message": "Session deleted"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    role_id = slugify_role_id(request.role_id or DEFAULT_ROLE_ID)
    with role_scope(role_id) as role_config:
        from services.memory_service import add_memory, correct_memory, search_memories
        from services.personality_service import maybe_grow, record_interaction

        session = get_session(request.session_id, role_id) if request.session_id else None
        if not session:
            session = create_empty_session(role_id)

        maybe_update_session_title(session, request.message)
        user_message = {
            "id": generate_id(),
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat(),
            "model": request.model or role_config.get("chat_model"),
            "has_image": bool(request.images),
        }
        session["messages"].append(user_message)

        history_limit = int(role_config.get("history_limit", 20) or 20)
        recent_history = session["messages"][:-1][-history_limit:]
        history_text = build_history_text(recent_history, history_limit)

        memory_top_k = int(role_config.get("memory_top_k", 20) or 20)
        memories = await search_memories(request.message, top_k=memory_top_k)
        filtered_memories = [mem for mem in memories if mem.get("score", 0) > 0.25 and "？" not in mem.get("content", "")]
        filtered_memories = filtered_memories[:memory_top_k]

        analysis_result = "未启用分析"
        if role_config.get("analysis_enabled", True):
            analysis_prompt = build_analysis_prompt(request.message, history_text)
            analysis_result = await generate_chat_response(
                role_config=role_config,
                message=analysis_prompt,
                history=[],
                system_prompt="你是一个专业的对话分析助手，只输出结构化分析结果。",
                images=None,
                model_override=request.model or role_config.get("chat_model"),
            )

        system_prompt = build_structured_system_prompt(
            role_config=role_config,
            user_message=request.message,
            analysis_result=analysis_result,
            history_text=history_text,
            memory_items=filtered_memories,
        )
        prepared_images = save_image_inputs(request.images, role_id)
        response_text = await generate_chat_response(
            role_config=role_config,
            # Keep the actual user turn in the final generation call.
            # Analysis already enriches the system prompt; clearing the user
            # message here causes the model to think the latest turn is empty.
            message=request.message,
            history=[],
            system_prompt=system_prompt,
            images=prepared_images,
            model_override=request.model or role_config.get("chat_model"),
        )
        response_text = sanitize_assistant_output(response_text)

        correction = detect_correction(request.message)
        if correction:
            corrected = await correct_memory(correction["old"], correction["new"])
            if corrected:
                response_text += "\n\n[已纠正记忆]"
            else:
                await add_memory(request.message, force=True)
        elif len(request.message) > 3:
            await add_memory(request.message, force=True)
        rebuild_memory_index_safe()

        assistant_message = {
            "id": generate_id(),
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat(),
            "model": request.model or role_config.get("chat_model"),
        }
        session["messages"].append(assistant_message)
        session["updated_at"] = datetime.now().isoformat()
        save_session(session, role_id)

        record_interaction(request.message, response_text)
        maybe_grow()

        return {"session": session, "message": assistant_message, "role_id": role_id}


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    role_id = slugify_role_id(request.role_id or DEFAULT_ROLE_ID)

    async def event_generator():
        with role_scope(role_id) as role_config:
            from services.memory_service import add_memory, correct_memory, search_memories
            from services.personality_service import maybe_grow, record_interaction

            session = get_session(request.session_id, role_id) if request.session_id else None
            if not session:
                session = create_empty_session(role_id)

            maybe_update_session_title(session, request.message)
            user_message = {
                "id": generate_id(),
                "role": "user",
                "content": request.message,
                "timestamp": datetime.now().isoformat(),
                "model": request.model or role_config.get("chat_model"),
                "has_image": bool(request.images),
            }
            session["messages"].append(user_message)

            history_limit = int(role_config.get("history_limit", 20) or 20)
            recent_history = session["messages"][:-1][-history_limit:]
            history_text = build_history_text(recent_history, history_limit)

            memory_top_k = int(role_config.get("memory_top_k", 20) or 20)
            memories = await search_memories(request.message, top_k=memory_top_k)
            filtered_memories = [mem for mem in memories if mem.get("score", 0) > 0.25 and "？" not in mem.get("content", "")]
            filtered_memories = filtered_memories[:memory_top_k]

            analysis_result = "未启用分析"
            if role_config.get("analysis_enabled", True):
                analysis_prompt = build_analysis_prompt(request.message, history_text)
                analysis_result = await generate_chat_response(
                    role_config=role_config,
                    message=analysis_prompt,
                    history=[],
                    system_prompt="你是一个专业的对话分析助手，只输出结构化分析结果。",
                    images=None,
                    model_override=request.model or role_config.get("chat_model"),
                )

            system_prompt = build_structured_system_prompt(
                role_config=role_config,
                user_message=request.message,
                analysis_result=analysis_result,
                history_text=history_text,
                memory_items=filtered_memories,
            )
            prepared_images = save_image_inputs(request.images, role_id)
            assistant_message = {
                "id": generate_id(),
                "role": "assistant",
                "content": "",
                "timestamp": datetime.now().isoformat(),
                "model": request.model or role_config.get("chat_model"),
            }

            yield sse_event("start", {
                "session_id": session["id"],
                "role_id": role_id,
                "message": assistant_message,
            })

            raw_response = ""
            visible_response = ""

            try:
                async for chunk in stream_chat_response(
                    role_config=role_config,
                    message=request.message,
                    history=[],
                    system_prompt=system_prompt,
                    images=prepared_images,
                    model_override=request.model or role_config.get("chat_model"),
                ):
                    raw_response += chunk
                    sanitized = sanitize_assistant_output_partial(raw_response)
                    if not sanitized.startswith(visible_response):
                        delta = sanitized
                    else:
                        delta = sanitized[len(visible_response):]
                    visible_response = sanitized
                    if delta:
                        yield sse_event("chunk", {"content": delta})
            except asyncio.CancelledError:
                pass
            except Exception as e:
                yield sse_event("error", {"message": str(e)})

            final_response = sanitize_assistant_output(raw_response)
            if not final_response and visible_response:
                final_response = visible_response

            correction = detect_correction(request.message)
            if correction:
                corrected = await correct_memory(correction["old"], correction["new"])
                if corrected:
                    final_response += "\n\n[已纠正记忆]"
                else:
                    await add_memory(request.message, force=True)
            elif len(request.message) > 3:
                await add_memory(request.message, force=True)
            rebuild_memory_index_safe()

            assistant_message["content"] = final_response
            assistant_message["timestamp"] = datetime.now().isoformat()
            session["messages"].append(assistant_message)
            session["updated_at"] = datetime.now().isoformat()
            save_session(session, role_id)

            record_interaction(request.message, final_response)
            maybe_grow()

            yield sse_event("end", {
                "session": session,
                "message": assistant_message,
                "role_id": role_id,
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/roles/{role_id}/memories")
async def api_list_role_memories(
    role_id: str,
    query: str = "",
    limit: int = 100,
):
    with role_scope(role_id):
        from services.memory_service import get_all_memories, search_memories

        if query:
            return {"items": await search_memories(query, top_k=limit)}
        return {"items": await get_all_memories(limit=limit)}


@app.put("/api/roles/{role_id}/memories/{memory_id}")
async def api_update_role_memory(
    role_id: str,
    memory_id: str,
    payload: MemoryUpdatePayload,
):
    with role_scope(role_id):
        from services.memory_service import get_embedding, load_memories, save_memories

        memories = load_memories()
        target = next((item for item in memories if item.get("id") == memory_id), None)
        if not target:
            raise HTTPException(status_code=404, detail="记忆不存在")

        target["content"] = payload.content[:300]
        if payload.type is not None:
            target["type"] = payload.type
        if payload.importance is not None:
            target["importance"] = payload.importance
        if payload.tags is not None:
            target["tags"] = payload.tags

        embedding = get_embedding(target["content"])
        if embedding:
            target["embedding"] = embedding
        target["updated_at"] = str(int(datetime.now().timestamp()))
        save_memories(memories)
        rebuild_memory_index_safe()
        return {"status": "success", "memory": target}


@app.delete("/api/roles/{role_id}/memories/{memory_id}")
async def api_delete_role_memory(role_id: str, memory_id: str):
    with role_scope(role_id):
        from services.memory_service import delete_memory

        await delete_memory(memory_id)
        rebuild_memory_index_safe()
        return {"status": "success"}


@app.post("/api/pig/send")
async def pig_send_message(request: ChatRequest):
    return await chat(request)


@app.get("/api/pig/memory/read")
async def pig_read_memory(
    query: str = "",
    limit: int = 10,
    role_id: str = Query(DEFAULT_ROLE_ID),
):
    with role_scope(role_id):
        from services.memory_service import get_all_memories, search_memories

        if query:
            results = await search_memories(query, top_k=limit)
            return {"type": "search", "results": results}
        memories = await get_all_memories(limit=limit)
        return {"type": "all", "memories": memories}


@app.post("/api/pig/memory/write")
async def pig_write_memory(
    content: str,
    memory_type: str = "general",
    importance: float = 0.5,
    role_id: str = Query(DEFAULT_ROLE_ID),
):
    with role_scope(role_id):
        from services.memory_service import add_memory

        memory = await add_memory(content=content, memory_type=memory_type, importance=importance)
        rebuild_memory_index_safe()
        return {"status": "success", "memory": memory}


@app.post("/api/pig/trigger_reflection")
async def pig_trigger_reflection(role_id: str = Query(DEFAULT_ROLE_ID)):
    with role_scope(role_id):
        from services.reflection_service import trigger_reflection

        reflection = await trigger_reflection()
        return reflection


@app.get("/api/pig/personality")
async def pig_get_personality(role_id: str = Query(DEFAULT_ROLE_ID)):
    with role_scope(role_id):
        from services.personality_service import get_personality_prompt, load_personality

        personality = load_personality()
        prompt = await get_personality_prompt()
        return {"personality": personality, "prompt": prompt}


@app.get("/api/pig/reflections")
async def pig_get_reflections(days: int = 7, role_id: str = Query(DEFAULT_ROLE_ID)):
    with role_scope(role_id):
        reflection_file = ensure_role_storage(role_id)["memory"] / "reflections.json"
        if not reflection_file.exists():
            return {"reflections": []}
        with open(reflection_file, "r", encoding="utf-8") as f:
            reflections = json.load(f)
        return {"reflections": reflections[-days:]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.SERVER["host"], port=config.SERVER["port"])

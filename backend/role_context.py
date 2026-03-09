import json
import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ROLES_DIR = DATA_DIR / "roles"
ROLE_META_FILE = DATA_DIR / "roles.json"
GLOBAL_MODEL_CONFIG_FILE = DATA_DIR / "global_model_config.json"

DEFAULT_ROLE_ID = "default"
GLOBAL_SETTING_KEYS = {
    "provider",
    "chat_model",
    "vision_model",
    "embedding_model",
    "chat_ollama_host",
    "embedding_host",
    "api_base_url",
    "api_key",
    "analysis_enabled",
    "memory_top_k",
    "history_limit",
    "temperature",
    "archive_all_messages",
}

_current_role_id: ContextVar[str] = ContextVar("current_role_id", default=DEFAULT_ROLE_ID)
_current_role_config: ContextVar[Optional[Dict[str, Any]]] = ContextVar("current_role_config", default=None)


def slugify_role_id(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return slug or DEFAULT_ROLE_ID


def normalize_role_id(value: Optional[str], fallback_name: Optional[str] = None, allow_default: bool = True) -> str:
    source = (value or fallback_name or "").strip()
    slug = slugify_role_id(source) if source else DEFAULT_ROLE_ID
    if slug == DEFAULT_ROLE_ID and not allow_default:
        return f"role-{uuid.uuid4().hex[:8]}"
    return slug


def build_default_role_config() -> Dict[str, Any]:
    default_chat_host = config.OLLAMA_HOSTS.get(config.DEFAULT_OLLAMA_HOST, "http://127.0.0.1:11434")
    embedding_host = config.OLLAMA_HOSTS.get("vps", config.OLLAMA_HOSTS.get("local_11435", default_chat_host))
    return {
        "id": DEFAULT_ROLE_ID,
        "name": config.AI_NAME,
        "avatar": "🐷",
        "description": "默认猪猪角色",
        "system_prompt": config.SYSTEM_PROMPT,
        "provider": "minimax" if getattr(config, "USE_MINIMAX", False) else "ollama",
        "chat_model": config.MODELS.get("chat", ""),
        "vision_model": config.MODELS.get("vision", ""),
        "embedding_model": config.MODELS.get("embedding", "bge-large"),
        "chat_ollama_host": default_chat_host,
        "embedding_host": embedding_host,
        "api_base_url": "https://api.minimaxi.com/v1",
        "api_key": getattr(config, "MINIMAX_API_KEY", ""),
        "analysis_enabled": True,
        "memory_top_k": 20,
        "history_limit": 20,
        "temperature": 0.7,
        "archive_all_messages": True,
    }


def build_default_global_model_config() -> Dict[str, Any]:
    role_defaults = build_default_role_config()
    return {key: deepcopy(role_defaults.get(key)) for key in GLOBAL_SETTING_KEYS}


def get_role_paths(role_id: Optional[str] = None) -> Dict[str, Path]:
    rid = slugify_role_id(role_id or get_current_role_id())
    if rid == DEFAULT_ROLE_ID:
        root = DATA_DIR
        return {
            "root": root,
            "sessions": root / "sessions",
            "memory": root / "memory",
            "growth": root / "growth",
            "temp": root / "temp",
            "config": root / "default_role_config.json",
        }

    root = ROLES_DIR / rid
    return {
        "root": root,
        "sessions": root / "sessions",
        "memory": root / "memory",
        "growth": root / "growth",
        "temp": root / "temp",
        "config": root / "config.json",
    }


def ensure_role_storage(role_id: Optional[str] = None) -> Dict[str, Path]:
    paths = get_role_paths(role_id)
    for key in ("root", "sessions", "memory", "growth", "temp"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def load_roles_index() -> List[Dict[str, Any]]:
    if ROLE_META_FILE.exists():
        try:
            with open(ROLE_META_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def save_roles_index(items: List[Dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(ROLE_META_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def load_global_model_config() -> Dict[str, Any]:
    defaults = build_default_global_model_config()
    if GLOBAL_MODEL_CONFIG_FILE.exists():
        try:
            with open(GLOBAL_MODEL_CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                return {**defaults, **{k: v for k, v in saved.items() if k in GLOBAL_SETTING_KEYS}}
        except Exception:
            pass
    default_role_config_file = get_role_paths(DEFAULT_ROLE_ID)["config"]
    if default_role_config_file.exists():
        try:
            with open(default_role_config_file, "r", encoding="utf-8") as f:
                saved_default = json.load(f)
            if isinstance(saved_default, dict):
                return {**defaults, **{k: v for k, v in saved_default.items() if k in GLOBAL_SETTING_KEYS}}
        except Exception:
            pass
    return defaults


def persist_global_model_config(settings: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**load_global_model_config(), **{k: v for k, v in settings.items() if k in GLOBAL_SETTING_KEYS}}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_MODEL_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return merged


def persist_role_config(role_config: Dict[str, Any]) -> Dict[str, Any]:
    rid = normalize_role_id(
        role_config.get("id"),
        fallback_name=role_config.get("name"),
        allow_default=bool(role_config.get("id") == DEFAULT_ROLE_ID),
    )
    role_config = deepcopy(role_config)
    role_config["id"] = rid
    paths = ensure_role_storage(rid)
    with open(paths["config"], "w", encoding="utf-8") as f:
        json.dump(role_config, f, ensure_ascii=False, indent=2)

    items = load_roles_index()
    summary = {
        "id": rid,
        "name": role_config.get("name", rid),
        "avatar": role_config.get("avatar", "🤖"),
        "description": role_config.get("description", ""),
        "provider": role_config.get("provider", "api"),
    }
    merged = [item for item in items if item.get("id") != rid and item.get("id") != DEFAULT_ROLE_ID]
    merged.append(summary)
    merged.sort(key=lambda item: item.get("name", ""))
    save_roles_index(merged)
    return role_config


def load_role_config(role_id: Optional[str] = None) -> Dict[str, Any]:
    rid = slugify_role_id(role_id or get_current_role_id())
    default_config = build_default_role_config()
    global_model_config = load_global_model_config()
    if rid == DEFAULT_ROLE_ID:
        paths = ensure_role_storage(rid)
        if paths["config"].exists():
            try:
                with open(paths["config"], "r", encoding="utf-8") as f:
                    saved = json.load(f)
                merged = {**default_config, **saved, **global_model_config, "id": DEFAULT_ROLE_ID}
                return merged
            except Exception:
                return {**default_config, **global_model_config}
        return {**default_config, **global_model_config}

    paths = ensure_role_storage(rid)
    if not paths["config"].exists():
        role_config = {**default_config, "id": rid, "name": rid}
        return persist_role_config(role_config)

    with open(paths["config"], "r", encoding="utf-8") as f:
        saved = json.load(f)
    merged = {**default_config, **saved, **global_model_config, "id": rid}
    return merged


def list_roles() -> List[Dict[str, Any]]:
    global_settings = load_global_model_config()
    default_summary = {
        "id": DEFAULT_ROLE_ID,
        "name": build_default_role_config()["name"],
        "avatar": "🐷",
        "description": "默认猪猪角色",
        "provider": global_settings.get("provider", build_default_role_config()["provider"]),
    }
    indexed = load_roles_index()
    found = [default_summary]
    seen = {DEFAULT_ROLE_ID}
    for item in indexed:
        rid = slugify_role_id(item.get("id", ""))
        if rid in seen:
            continue
        seen.add(rid)
        found.append(item)
    if ROLES_DIR.exists():
        for role_dir in ROLES_DIR.iterdir():
            if not role_dir.is_dir():
                continue
            rid = slugify_role_id(role_dir.name)
            if rid in seen:
                continue
            config_data = load_role_config(rid)
            found.append({
                "id": rid,
                "name": config_data.get("name", rid),
                "avatar": config_data.get("avatar", "🤖"),
                "description": config_data.get("description", ""),
                "provider": config_data.get("provider", "api"),
            })
    return found


def upsert_role(role_data: Dict[str, Any], role_id: Optional[str] = None) -> Dict[str, Any]:
    base = load_role_config(role_id)
    explicit_id = role_data.get("id") or role_id
    rid = normalize_role_id(
        explicit_id,
        fallback_name=role_data.get("name"),
        allow_default=explicit_id == DEFAULT_ROLE_ID,
    )
    merged = {**base, **role_data, "id": rid}
    return persist_role_config(merged)


def get_current_role_id() -> str:
    return _current_role_id.get()


def get_current_role_config() -> Dict[str, Any]:
    cached = _current_role_config.get()
    if cached:
        return cached
    config_data = load_role_config(get_current_role_id())
    _current_role_config.set(config_data)
    return config_data


@contextmanager
def role_scope(role_id: Optional[str]):
    rid = slugify_role_id(role_id or DEFAULT_ROLE_ID)
    config_data = load_role_config(rid)
    token_id = _current_role_id.set(rid)
    token_config = _current_role_config.set(config_data)
    try:
        yield config_data
    finally:
        _current_role_id.reset(token_id)
        _current_role_config.reset(token_config)

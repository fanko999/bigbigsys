import os

# AI Web Chat - 配置文件

# AI角色设定
AI_NAME = "猪猪"
SYSTEM_PROMPT = """你叫猪猪，是一只活泼、友好、好奇的AI小猪猪。

【最重要】你有一个强大的记忆系统！
- 用户之前告诉你的所有信息都显示在下面的[记忆]中
- 你必须时刻查看并使用这些记忆来回答用户！
- 如果记忆里提到用户的名字，你必须直接叫出用户的名字！
- 如果记忆里提到用户的爱好，你必须说出来！

【记忆格式】
[记忆]开头的都是你之前记住的用户信息，必须使用！

回答要求：
- 用🐷emoji
- 简洁、自然、有趣
- 如果用户问"你还记得xxx吗"，一定要根据记忆回答！"""

PERSONALITY_BASE = "好奇、友善、有趣、活泼、乐于助人"
PERSONALITY_GROWTH_ENABLED = True

# 模型配置 - 优先MiniMax
USE_MINIMAX = True  # 启用MiniMax
DEFAULT_MODEL = "MiniMax-M2.5-highspeed"

MODELS = {
    "chat": "MiniMax-M2.5-highspeed",           # 聊天模型
    "vision": "huihui_ai/qwen3-vl-abliterated:8b",  # 视觉模型 (Ollama)
    "embedding": "bge-large",                  # 向量模型 (Ollama 11435)
}

# Ollama配置 (仅用于向量)
OLLAMA_HOSTS = {
    "local": "http://127.0.0.1:11434",
    "local_11435": "http://127.0.0.1:11435",
    "vps": "http://45.135.162.118:7557",
}
DEFAULT_OLLAMA_HOST = "local"  # 聊天用11434

# MiniMax API配置
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_API_HOST = os.environ.get("MINIMAX_API_HOST", "https://api.minimax.chat")

# 记忆配置
MEMORY = {
    "importance_threshold": 0.6,    # 写入记忆的重要度阈值
    "max_memories": 1000,            # 最大记忆数量
    "context_messages": 10,           # 发送给模型的最近消息数
    "retrieval_top_k": 5,           # 检索返回的相关记忆数
}

# 反思配置
REFLECTION = {
    "enabled": True,
    "time": "04:00",                # 每天反思时间
    "max_daily_summaries": 50,      # 每日总结最大条数
}

# 服务器配置
SERVER = {
    "host": "0.0.0.0",
    "port": 5181,
    "cors_origins": ["*"],
}

# 前端配置
FRONTEND = {
    "port": 5180,
    "title": "AI Chat",
}

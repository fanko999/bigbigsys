# 猪猪系统

一个可长期运行的多角色 AI 聊天系统。

它不是单纯的“聊天壳”，而是把以下几件事放到了一起：

- 多角色管理
- 每个角色独立的会话、记忆、成长数据
- 可切换聊天模型和向量模型
- 支持 MiniMax、Ollama、本地或远程 API
- 服务端真正流式输出
- JSON 文件落地，方便备份、迁移、调试

## 这套系统的好处

### 1. 一角色一套独立记忆系统

这不是前端假分组，而是服务端真实隔离。

每个角色都有自己独立的：

- 配置
- 会话
- 长期记忆
- 成长日志
- 向量索引

这意味着你可以同时维护多个不同人格、不同用途的 AI，不会互相串记忆。

### 2. 记忆逻辑和聊天逻辑可长期运行

系统不是只拼 prompt 就完了，当前消息在进入模型前会经过：

1. 读取最近历史对话
2. 检索相关长期记忆
3. 可选做一次对话分析
4. 拼接结构化 system prompt
5. 再调用主模型生成回复

这样做的好处是：

- 当前消息优先
- 历史上下文不容易丢
- 长期记忆可以持续参与回答
- 角色人格和系统 prompt 能稳定生效

### 3. 部署简单，数据透明

数据主要以 JSON 文件存储，目录清晰，方便：

- 本地备份
- 换机器迁移
- 做角色导出导入
- 调试问题时直接看会话和记忆文件

### 4. 模型接入灵活

你可以自由切换：

- MiniMax 作为主聊天模型
- Ollama 作为本地或远程聊天模型
- 远程 OpenAI 兼容 API
- 自定义向量模型和向量服务地址

### 5. 前端可直接运营角色

前端支持：

- 创建角色
- 编辑角色 Prompt
- 角色导出导入
- 记忆查看、编辑、删除
- 切换聊天模型和向量模型
- 折叠式工作台布局
- 真流式输出与停止生成

## 当前架构

```text
frontend/index.html
    ↓
FastAPI backend/main.py
    ↓
角色隔离上下文 role_context.py
    ↓
对话路由 /api/chat /api/chat/stream
    ↓
记忆检索 memory_service.py
    ↓
LLM 路由 llm_router.py
    ├─ MiniMax
    ├─ Ollama
    └─ OpenAI 兼容 API
```

## 目录结构

```text
ai-web-chat/
├─ backend/
│  ├─ main.py
│  ├─ config.py
│  ├─ requirements.txt
│  ├─ role_context.py
│  └─ services/
├─ frontend/
│  └─ index.html
├─ data/
│  ├─ sessions/
│  ├─ memory/
│  ├─ growth/
│  └─ roles/
├─ start_backend.bat
├─ start_frontend.bat
└─ README.md
```

## 环境要求

### 操作系统

- Windows 优先验证
- Linux 也可运行，但启动脚本需自行调整

### Python

- 推荐 Python 3.11

不建议优先用 Python 3.14 跑这套依赖，尤其是涉及部分科学计算或向量库时更容易遇到兼容问题。

### 必备依赖

后端基础依赖见 [backend/requirements.txt](/Users/Administrator/Documents/fank/BV/bigbigsys/ai-web-chat/backend/requirements.txt)：

- `fastapi`
- `uvicorn`
- `aiohttp`
- `pydantic`
- `python-multipart`
- `requests`
- `numpy`

### 可选依赖

- `faiss-cpu`

说明：

- 没装 `faiss-cpu` 时，系统仍然可以运行
- 只是向量索引搜索和索引重建会退化或跳过
- 聊天主链路不会因此直接挂掉

推荐安装方式：

```powershell
python -m pip install faiss-cpu
```

## 模型与服务依赖

你可以按自己的部署方式选择。

### 方案 A：MiniMax 聊天 + Ollama 向量

适合你现在这套“主模型走 MiniMax、向量走远程 Ollama”的方式。

需要：

- 一个可用的 `MiniMax API Key`
- 一个可访问的 Ollama 向量服务
- 至少一个 embedding 模型，例如 `bge-large`

### 方案 B：全部走 Ollama

需要：

- 本地或远程 Ollama 服务
- 一个聊天模型
- 一个向量模型

### 方案 C：OpenAI 兼容 API

需要：

- API Base URL
- API Key
- 对应模型名

## 安装步骤

### 1. 克隆或复制项目

```powershell
cd C:\your\workspace
```

把整个 `ai-web-chat` 项目目录放到目标机器。

### 2. 安装 Python 依赖

```powershell
cd backend
python -m pip install -r requirements.txt
```

如果你希望启用 Faiss：

```powershell
python -m pip install faiss-cpu
```

### 3. 配置环境变量

推荐用环境变量注入密钥，而不是把真实 Key 写进代码。

Windows PowerShell：

```powershell
$env:MINIMAX_API_KEY="你的真实 MiniMax Key"
$env:MINIMAX_API_HOST="https://api.minimax.chat"
```

如果要长期生效，可以写入用户环境变量。

### 4. 检查后端配置

默认配置文件在 [backend/config.py](/Users/Administrator/Documents/fank/BV/bigbigsys/ai-web-chat/backend/config.py)。

你主要需要确认：

- `MODELS["chat"]`
- `MODELS["vision"]`
- `MODELS["embedding"]`
- `OLLAMA_HOSTS`
- `SERVER["port"]`
- `FRONTEND["port"]`

### 5. 启动后端

方式一：

```powershell
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 5181
```

方式二：

直接运行 [start_backend.bat](/Users/Administrator/Documents/fank/BV/bigbigsys/ai-web-chat/start_backend.bat)。

### 6. 启动前端

```powershell
cd ..
python -m http.server 5180
```

或者直接运行 [start_frontend.bat](/Users/Administrator/Documents/fank/BV/bigbigsys/ai-web-chat/start_frontend.bat)。

### 7. 打开网页

- 前端：[http://127.0.0.1:5180](http://127.0.0.1:5180)
- 后端：[http://127.0.0.1:5181](http://127.0.0.1:5181)

## 首次使用建议

首次启动后，建议按这个顺序检查：

1. 后端首页能否打开
2. 前端能否加载角色列表
3. 角色配置里的聊天模型和向量模型是否正确
4. MiniMax Key 是否生效
5. Ollama 地址是否可访问
6. 先新建一个测试角色再开始正式使用

## 角色隔离说明

### 默认角色

默认角色 `default` 使用：

- `data/sessions`
- `data/memory`
- `data/growth`

### 自定义角色

每个新角色使用：

- `data/roles/<role_id>/sessions`
- `data/roles/<role_id>/memory`
- `data/roles/<role_id>/growth`
- `data/roles/<role_id>/config.json`

因此每个角色天然具备独立人格、独立会话、独立记忆，不会与其他角色混用。

## 记忆系统说明

记忆系统当前保留了原有主逻辑，只做了角色隔离和接口整理。

主要流程：

1. 用户消息进入后端
2. 读取最近历史
3. 调用 embedding 服务生成向量
4. 搜索相关记忆
5. 综合向量分数、重要度、类型权重、时间权重、命中次数排序
6. 将命中的长期记忆拼入最终 system prompt

如果安装了 `faiss-cpu`：

- 使用 Faiss 索引加速搜索和重建

如果没有安装：

- 会退回到普通向量比对逻辑
- 日志里会看到 `No module named 'faiss'`

## 流式输出说明

现在系统已经支持真正的服务端流输出：

- 后端接口：`/api/chat/stream`
- 前端会实时读取 `start / chunk / end / error` 事件
- 支持中途停止

这不是前端假装逐字打印，而是服务端真实推流，前端再做平滑渲染。

## API 简表

### 系统

- `GET /`
- `GET /api/models`

### 角色

- `GET /api/roles`
- `POST /api/roles`
- `GET /api/roles/{role_id}`
- `PUT /api/roles/{role_id}`
- `DELETE /api/roles/{role_id}`
- `GET /api/roles/{role_id}/export`
- `POST /api/roles/import`

### 会话

- `GET /api/sessions`
- `POST /api/sessions`
- `GET /api/sessions/{session_id}`
- `DELETE /api/sessions/{session_id}`

### 对话

- `POST /api/chat`
- `POST /api/chat/stream`

### 记忆

- `GET /api/roles/{role_id}/memories`
- `PUT /api/roles/{role_id}/memories/{memory_id}`
- `DELETE /api/roles/{role_id}/memories/{memory_id}`

## 发布到 GitHub 前的建议

这一步很重要。

### 不要提交这些内容

- 真实 API Key
- `data/` 里的真实聊天记录
- `data/` 里的长期记忆
- 本地缓存和 `__pycache__`

本项目已经加入了 [.gitignore](/Users/Administrator/Documents/fank/BV/bigbigsys/ai-web-chat/.gitignore)，默认会忽略这些内容。

### 建议提交这些内容

- `backend/`
- `frontend/`
- `README.md`
- 启动脚本
- 文档说明

## 常见问题

### 1. 日志出现 `No module named 'faiss'`

不是主聊天功能挂了，而是你当前 Python 环境没装 `faiss-cpu`。

### 2. 页面能打开，但模型不回复

优先检查：

- `MiniMax API Key`
- `API Base URL`
- `Ollama` 主机地址
- 模型名是否存在
- 远程模型机是否可访问

### 3. 换到另一台机器后速度不同

通常是下面几项差异造成的：

- 是否安装了 `faiss-cpu`
- 向量服务是否更快
- 网络到 MiniMax 是否更稳定
- 是否是正式模型机在跑 Ollama

## 安全提示

如果你准备公开这个项目，请务必先轮换你以前用过的真实 API Key。

即使我已经把仓库里的明文 Key 改成了环境变量读取，旧 Key 只要曾经出现在公开文件或历史记录里，都应该视为已泄露并立即更换。

## 致谢

这套系统是围绕“猪猪”角色持续迭代出来的一套长期记忆式 AI 聊天服务端 + 前端工程，不是一次性 demo，而是适合持续运行和继续扩展的基础系统。

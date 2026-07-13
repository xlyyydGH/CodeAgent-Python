# CodeAgent Python

CodeAgent Python 是一个 Python-native AI 编程助手项目，定位是“部署一次，浏览器全流程操控”。项目使用 FastAPI 作为主后端、React 作为主界面、Python service 作为代码分析与 tokenizer 辅助服务。

> 本项目基于 MIT 许可的 [ZhikunCode](https://github.com/zhikunqingtao/zhikuncode) 进行 Python-native 架构改写。原项目版权与历史成果归原作者所有；本仓库重点展示 Python Agent Loop、工具调用、权限控制、上下文管理、MCP 与多 Agent 编排的实现与学习过程。

这个仓库适合用于面试展示：重点是把 Agent loop、工具调用、权限审批、上下文预算、实时事件、记忆检索、MCP、验证证据和多 Agent 编排这些核心机制，组织成一套可运行、可讲清楚的 Python-native 架构。

## 目录

- [当前状态](#当前状态)
- [特性亮点](#特性亮点)
- [架构概览](#架构概览)
- [快速开始](#快速开始)
- [LLM 配置](#llm-配置)
- [核心能力](#核心能力)
- [页面与 API](#页面与-api)
- [关键设计](#关键设计)
- [开发与验证](#开发与验证)
- [项目边界](#项目边界)
- [面试讲法](#面试讲法)

## 当前状态

- 主后端位于 `backend-python/`，负责 REST、WebSocket、QueryLoop、工具、权限、MCP、记忆、验证和多 Agent 编排。
- React 前端保留在 `frontend/`，构建产物由 Python 后端直接服务。
- `python-service/` 继续提供 tokenizer、代码分析、复杂度、调用路径等辅助能力。
- `run.py` 会同时启动 Python backend 和 Python analysis service。
- Dockerfile 已改为 Python runtime，并在构建阶段编译 React 前端。
- 当前版本已经覆盖 Agent 编程主链路，但部分深层能力仍在持续完善。详细能力边界见 [docs/python-source-differences.md](docs/python-source-differences.md)。

## 特性亮点

| 特性 | 说明 |
| --- | --- |
| 浏览器端 AI 编程助手 | 部署后通过浏览器访问 React 主界面，支持 Chat、Realtime、Dashboard、Sessions、Tools、Tasks、Settings、Files、Activity、Verify、MCP、Memory 等页面；无前端构建产物时提供 Python fallback UI。 |
| Python-native Agent Loop | 用 FastAPI + Python runtime 承接 Agent 编排层，主请求进入 QueryLoop 状态机，记录 prepare、model_call、streaming、tool_running、waiting_permission、completed、failed、aborted 等阶段。 |
| 多 Agent 协作主链路 | 支持 SubAgent、后台 Agent、Team/Swarm 编排、Coordinator workflow、worker 状态、phase barrier、shared task list、mailbox ack/replay/recover、权限冒泡和批量审批。 |
| 智能上下文管理 | 已恢复 Token 预算、五层 ContextCascade、413/prompt-too-long 恢复、媒体剥离、工具结果摘要、模型真实 usage 回写和自纠错主链路，适合展示长对话和大项目上下文处理设计。 |
| 48+ 工具体系 | ToolRegistry 覆盖文件读写、目录/Glob/Grep、Git、命令执行、PowerShell、REPL、Notebook、LSP、Memory、Agent、Task、Cron、VerifyJourney、TokenCount、ContextStatus、ToolSearch 等能力。 |
| 文件编辑可靠性 | FileEdit/MultiEdit 支持 hash 冲突检测、五策略 fuzzy match、all-or-nothing 批量编辑、同目录临时文件原子写入、snapshot-before-write、unified diff、history snapshot/diff/rewind 和 React History 面板。 |
| 权限与安全控制 | 高风险工具可进入 permission request；命令执行接入风险分级、敏感路径/密钥读取审批、密钥外传阻断、PowerShell 编码执行阻断、反向 shell/管道执行阻断、输出脱敏截断和超时/退出码分类。 |
| 国产大模型与 OpenAI-compatible Provider | 后端支持 XFYun、DashScope、DeepSeek、Moonshot、Zhipu、MiniMax、ZenMux 和通用 OpenAI-compatible，接入 model-aware retry、错误分类、降级链和 usage 标准化。 |
| 实时事件推送 | 提供 SockJS/STOMP 风格 `/ws`，推送 query、tool、permission、verify、swarm 等事件，并支持 user queue、ACK/NACK、REST replay、destination 过滤、订阅生命周期、last-message-id 恢复和轻量背压统计。 |
| MCP 扩展体系 | 支持 MCP server 状态、resources、prompts、capabilities、local invocation、stdio 与 http/streamable_http 真实 JSON-RPC、wrapped tools 动态注册、tools/call、content 提取、缓存降级、1MB 截断、schema validation、auth cache 和 reconnect backoff。 |
| 记忆检索 | MemdirService 支持 episodic、semantic、procedural、team 记忆，检索结合 BM25、CJK token、title boost、分类过滤、rerank boost 和 updatedAt boost。 |
| 运行时验证与证据 | 支持 VerifyJourney、HTTP/API verifier、evidence bundle/blob、signal 计算和前端面板数据，为“Agent 生成结果如何验证”提供展示链路。 |
| 代码分析与可视化数据 | Python analysis service 提供 tokenizer、复杂度、代码路径、代码图、变更影响等能力，供前端复杂度视图、路径追踪和项目分析使用。 |
| Docker 一键部署 | Dockerfile 与 compose 已切到 Python runtime，构建 React 产物后启动 FastAPI backend 与 Python analysis service，暴露 `http://localhost:8080`。 |

> 说明：上表描述的是 Python Edition 当前可展示和已接入的能力。完整 STOMP broker 语义、深层 Bash 沙箱、插件隔离、高并发多 Agent 压测、Playwright 级浏览器验证和 SWE-bench 复测仍在后续优化路线中。

## 架构概览

```text
┌──────────────────────────────┐
│ React 18 + TypeScript UI      │
│ frontend/                     │
│ build output: frontend/dist   │
└───────────────┬──────────────┘
                │ HTTP / SockJS-STOMP
                ▼
┌──────────────────────────────┐
│ Python FastAPI Backend        │
│ backend-python/app.py         │
│ port 8080                     │
│ REST / WebSocket / QueryLoop  │
│ tools / permissions / agents  │
│ memory / MCP / verification   │
└───────────────┬──────────────┘
                │ HTTP
                ▼
┌──────────────────────────────┐
│ Python Analysis Service       │
│ python-service/src            │
│ port 8000                     │
│ tokenizer / code intelligence │
└──────────────────────────────┘
```

当前运行架构是 FastAPI + React + Python service。核心编排层使用 Python 的 FastAPI、asyncio/thread、JSON state + SQLite sync layer 和文件系统发现机制来承接。

## 快速开始

### 本地启动

```powershell
cp .env.example .env

cd frontend
npm ci
npm run build
cd ..

.\python-service\venv\Scripts\python.exe run.py
```

启动后访问：

- React UI / API: `http://localhost:8080`
- Python fallback UI: `http://localhost:8080/ui`
- Python realtime fallback UI: `http://localhost:8080/ui/realtime`
- Python analysis service: `http://localhost:8000`

如果没有现成虚拟环境，可以自行创建：

```powershell
python -m venv python-service\venv
.\python-service\venv\Scripts\python.exe -m pip install -r backend-python\requirements.txt
.\python-service\venv\Scripts\python.exe -m pip install -r python-service\requirements.txt

cd frontend
npm ci
npm run build
cd ..

.\python-service\venv\Scripts\python.exe run.py
```

### Docker 启动

```bash
cp .env.example .env
docker compose up -d --build
```

Docker 镜像会：

- 使用 Node 22 构建 `frontend/dist`
- 使用 Python 3.12 运行 `backend-python/app.py`
- 同容器启动 `python-service`
- 暴露 `http://localhost:8080`

当前 `docker-compose.yml` 显式传入 XFYun、DashScope、DeepSeek、Moonshot、Zhipu、MiniMax、ZenMux 和通用 `LLM_API_KEY` 配置。ZenMux 需要额外填写自己的 OpenAI-compatible base URL。

## LLM 配置

后端使用 OpenAI-compatible Chat Completions 格式。没有有效 API Key 时，系统会走本地 fallback，方便演示 UI 和后端链路；配置 Key 后会调用真实模型。

### iFlytek / XFYun MaaS

如果使用截图里的讯飞 MaaS 服务，在 `.env` 中填写：

```env
LLM_PROVIDER_XFYUN_API_KEY=your-real-api-key
# 可选；默认已匹配截图
# LLM_PROVIDER_XFYUN_BASE_URL=https://maas-api.cn-huabei-1.xf-yun.com/v2
# LLM_PROVIDER_XFYUN_MODEL=xopqwen36v35b
```

### 已接入的 Python 后端 Provider

`backend-python/app.py` 当前实际识别这些环境变量，按顺序选择第一个有效 Key：

```env
LLM_PROVIDER_XFYUN_API_KEY=your-xfyun-key
LLM_PROVIDER_DASHSCOPE_API_KEY=your-dashscope-key
LLM_PROVIDER_DEEPSEEK_API_KEY=your-deepseek-key
LLM_PROVIDER_MOONSHOT_API_KEY=your-moonshot-key
LLM_PROVIDER_ZHIPU_API_KEY=your-zhipu-key
LLM_PROVIDER_MINIMAX_API_KEY=your-minimax-key
LLM_PROVIDER_ZENMUX_API_KEY=your-zenmux-key
LLM_PROVIDER_ZENMUX_BASE_URL=your-zenmux-openai-compatible-base-url

# 通用兼容模式
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_DEFAULT_MODEL=qwen3.7-max
```

每个 Provider 也支持独立 `*_BASE_URL` 和 `*_MODEL` 环境变量，例如 `LLM_PROVIDER_ZHIPU_MODEL=glm-5.1`、`LLM_PROVIDER_MINIMAX_MODEL=MiniMax-M3`。

LLM 调用链路已接入 model-aware retry、降级策略和真实 usage 回写：`claude`、`qwen`、`deepseek` 有独立重试次数、指数退避间隔和 Retry-After 处理规则，`qwen3.7-max` 等模型会按降级链回退到轻量/备用模型；模型返回的 prompt/completion/cache token 会写回 assistant message、query events 和 REST 返回。

## 核心能力

| 能力 | Python Edition 当前实现 |
| --- | --- |
| 浏览器全流程操控 | React 主界面由 Python backend 服务；无构建产物时可使用 `/ui` fallback |
| 普通聊天 | `/api/query` 和 `/ui/chat` 支持创建会话、保存消息、调用 LLM、写入 assistant 回复 |
| LLM Provider | 支持 XFYun、DashScope、DeepSeek、Moonshot、Zhipu、MiniMax、ZenMux 和通用 OpenAI-compatible，并接入 model-aware retry、错误分类、模型降级链与真实 usage 回写 |
| 实时工作区 | 提供 SockJS/STOMP 风格 `/ws`，推送 query、tool、permission、verify、swarm 等事件，并支持 user queue、client ACK/NACK、REST replay、destination 过滤、订阅/退订生命周期、last-message-id 订阅恢复和轻量背压统计 |
| QueryEngine | `QueryLoopState` 支持 prepare、model_call、streaming、tool_running、waiting_permission、completed、failed、aborted 等阶段；大工具输出会保留原文并生成 `summary` 写入 tool call 与实时 `tool_result` event |
| Token 预算 | 支持 TokenCounter 核心规则：中文比例、JSON/code/text、图片 token、message 结构开销、precise tokenizer fallback |
| 上下文恢复 | 主 `/api/query` 已接入五层 ContextCascade：snip selection、micro compact、auto compact、collapse drain、reactive compact，并补 413 cause、超长 prompt recovery event 和媒体剥离 metadata |
| 终止策略 | 已加入 `TerminationDecision`，能记录 continue / wait / stop / abort，并通过 `termination_decision` event 暴露 |
| 工具系统 | `ToolRegistry` 注册 48+ 内置/兼容工具，覆盖文件、搜索、Git、命令、PowerShell、REPL、Notebook、LSP、Memory、Agent、Task、Cron、VerifyJourney、上下文辅助等；FileEdit/MultiEdit 支持 hash 冲突检测、五策略 fuzzy match、all-or-nothing 批量编辑、snapshot-before-write、unified diff、原子写入，并把工具写入前快照接入 history snapshot/diff/rewind 和 React History 面板 |
| 命令安全 | 支持危险命令分级、敏感路径/密钥读取审批、密钥外传阻断、PowerShell 编码/下载执行阻断、反向 shell/管道执行阻断、输出脱敏截断、退出码/超时/阻断分类 metadata |
| 权限控制 | 高风险工具可进入 permission request；支持等待、拒绝、批准、用户中断、倒计时、超时自动拒绝和事件回传 |
| 多 Agent | 支持 SubAgent、后台 Agent、Team/Swarm 编排、显式 Coordinator workflow、worker 状态、并发限制、权限冒泡与批量审批 |
| Swarm 协作 | 支持 shared task list、phase barrier、barrier 驱动 workflow 自动推进、orphan worker recovery、phase-aware mailbox、mailbox ack/replay/recover |
| 记忆系统 | `MemdirService` 支持 BM25、CJK token、title boost、分类过滤、rerank boost、updatedAt boost |
| MCP | 支持 server 状态、resources、prompts、capabilities、local invocation、stdio 与 http/streamable_http 真实 JSON-RPC 连接、wrapped tools 动态注册到 ToolRegistry/LLM 工具列表、tools/call 调用语义、content 提取、结果缓存降级、1MB 截断保护、schema validation、auth failure cache 和 reconnect backoff |
| Skill / Plugin / Hook | 支持文件系统 skill、plugin manifest 发现、hook 注册与执行 |
| 文件与 Git | 提供文件树、搜索、上传附件、Git log/diff/blame、history snapshot/rewind |
| 代码分析 | 提供复杂度、代码路径、代码图、变更影响等 Python API |
| 验证证据 | 支持 VerifyJourney、HTTP/API verifier、evidence bundle/blob、signal 计算和前端面板数据 |
| Docker 部署 | Dockerfile 和 compose 使用 Python runtime，负责启动 FastAPI backend 与 Python analysis service |

## 页面与 API

常用页面：

- `/`：React 主界面
- `/ui`：Python fallback UI
- `/ui/realtime`：轻量实时工作区
- `/docs`：FastAPI OpenAPI 文档

常用 API：

- `GET /api/health`：健康检查
- `POST /api/query`：普通 query
- `POST /api/query/stream`：SSE 风格 query
- `POST /api/sessions`、`GET /api/sessions`：会话管理
- `GET /api/tools`：工具列表
- `GET /api/memory/entries`、`GET /api/memory/search`：记忆系统
- `GET /api/mcp/servers`、`GET /api/mcp/capabilities`：MCP 状态
- `POST /api/verify/journey`：运行时验证
- `POST /api/swarm`、`GET /api/swarm/{id}`、`POST /api/swarm/{id}/recover`：多 Agent / Swarm 与重启恢复
- `GET /api/files/tree`、`GET /api/git/log`：文件与 Git

当前 `backend-python/app.py` 中有 200+ 个 FastAPI route，用于支撑 React 页面和后端能力。

## 关键设计

### 1. Python-native QueryLoop

一次用户请求会被建模为 `QueryLoopState`：

1. 创建 session / loop
2. 估算 system、history、user、tool、memory 的 token budget
3. 进入 prepare 阶段，必要时触发五层 ContextCascade
4. 执行显式 tool calls，并记录 tool_use / tool_result；长工具输出同步生成摘要，避免前端和后续上下文直接处理超大结果
5. 高风险工具进入 waiting_permission
6. 后台 Agent 完成后把结果注入上下文
7. 调用 OpenAI-compatible LLM
8. 记录 stream_delta、model_call、final_response、termination_decision
9. 持久化 session、message、queryLoop、events

这个设计让 React Realtime 页面能看到中间过程，而不是只看到最终回答。

### 2. TokenCounter 核心逻辑

Python 版实现了 token 估算的关键规则：

- 默认字符/token 比例约为 `3.5`
- JSON 更密集，按约 `2.0`
- 自然语言按约 `4.0`
- 中文比例越高，chars/token 越低
- 图片按 `ceil(width * height / 750)` 估算，异常尺寸 fallback `85`
- message list 额外计算每条 message 的结构开销
- `PRECISE_TOKENIZER` 开启时请求 `python-service /api/tokenizer/count`，失败则回退启发式估算

### 3. 工具、权限与终止策略

所有工具集中注册到 `ToolRegistry`。执行前会经过权限策略判断，高风险工具可返回 `permission request`，QueryLoop 切到 `waiting_permission`，前端收到审批事件后继续批准或拒绝。

当前 Python 版已把权限等待、正常结束、错误和用户中断统一记录为 `TerminationDecision`，包括：

- `continue`：继续工具循环或模型调用
- `wait`：等待用户权限
- `stop`：正常结束、达到上限或错误停止
- `abort`：用户中断

同时实现了 QueryEngine 常见 stop 分支：`tool_use` 继续工具循环，`max_tokens/length` 触发输出上限恢复（升级 maxTokens、注入续写提示、达到上限后停止），`withhold` 扣留可恢复错误并继续上下文恢复，非标准模型 `stopReason` 会作为 `modelStopReason` 元数据落到 termination event。

### 4. 多 Agent 与 Swarm

Python 版用 `asyncio` + 显式 Coordinator workflow 实现 Team / Swarm / SubAgent 的主要体验：

- 全局 Agent 并发限制
- 单会话 Agent 并发限制
- 嵌套深度限制
- 后台 Agent task tracker
- Research / Synthesis / Implementation / Verification 四阶段 Coordinator workflow
- Team/Swarm worker 状态
- shared task list
- phase barrier，且 barrier release 会推进 Coordinator workflow
- orphan worker recovery：进程重启后若持久化 worker 仍是运行态但 live task 已丢失，启动扫描或手动 recover 可重排进 durable queue
- phase-aware mailbox
- mailbox ack / replay / recover
- TeamMailbox / SharedTaskList / LeaderPermissionBridge 兼容事件字段：mailbox、shared task、permission bubble 均会进入 coordinator event envelope，并带 messageId、phaseIndex、creatorId、riskLevel、timeout/deadline 等前端可展示字段
- 权限冒泡、倒计时、超时自动拒绝和批量审批

当前版本主要依赖 `asyncio`、线程池和内存/JSON 状态；四阶段 workflow、barrier 推进、startup recovery 和三类协作事件字段主链路已接入，但超大并发压测、可配置恢复策略、移动端审批和 Agent DAG 端到端展示仍需继续补齐。

### 5. Memdir 记忆检索

`MemdirService` 支持四类记忆：

- episodic：交互经历
- semantic：知识事实
- procedural：流程经验
- team：团队协作知识

检索时结合 BM25、标题 boost、CJK tokenization、分类过滤、rerank boost 和更新时间 boost，让 Agent 能基于历史经验做上下文增强。

### 6. React 主界面与 Python fallback

React 是主界面，负责 Chat、Realtime、Dashboard、Sessions、Tools、Tasks、Settings、Files、Activity、Verify、MCP、Memory 等页面体验。Python fallback 页面保留给后端调试和无前端构建产物时的演示。

## 开发与验证

完整的小规模可用性测试清单见 [docs/python-edition-testing.md](docs/python-edition-testing.md)。该文档记录了最近一次实测结果、推荐复跑命令和当前测试边界。

安装依赖：

```powershell
.\python-service\venv\Scripts\python.exe -m pip install -r backend-python\requirements.txt
.\python-service\venv\Scripts\python.exe -m pip install -r python-service\requirements.txt

cd frontend
npm ci
npm run build
cd ..
```

运行后端测试：

```powershell
.\python-service\venv\Scripts\python.exe -m pytest backend-python\tests -q
```

常用目标回归：

```powershell
.\python-service\venv\Scripts\python.exe -m pytest backend-python\tests\test_query_runtime.py -q
.\python-service\venv\Scripts\python.exe -m pytest backend-python\tests\test_app.py -q -k "rest_query_persists or permission_wait_emits or mailbox_ack_replay"
```

语法检查：

```powershell
.\python-service\venv\Scripts\python.exe -m py_compile backend-python\app.py backend-python\zhikun_py\query_runtime.py
```

前端构建：

```powershell
cd frontend
npm run build
```

## 代码结构

```text
.
├── backend-python/
│   ├── app.py                       # FastAPI 主应用，REST/WS/API 聚合入口
│   ├── requirements.txt
│   ├── tests/                       # Python 后端回归测试
│   └── zhikun_py/
│       ├── query_runtime.py          # QueryLoop、TokenCounter、termination strategy
│       ├── tools.py                  # ToolRegistry、工具实现、Agent 工具
│       ├── permissions.py            # 权限策略
│       ├── memdir_runtime.py         # BM25 + rerank 记忆检索
│       ├── mcp_runtime.py            # MCP 状态与调用
│       ├── verify_runtime.py         # runtime verification
│       ├── correction_runtime.py     # self-correction loop
│       ├── sqlite_store.py           # SQLite 同步层
│       └── websocket_runtime.py      # WebSocket session manager
├── frontend/
│   ├── src/                          # React UI、stores、panels、visualization
│   └── dist/                         # npm run build 产物
├── python-service/
│   └── src/                          # tokenizer、代码分析辅助服务
├── configuration/
│   └── mcp/                          # MCP capability registry
├── docs/
│   └── python-source-differences.md  # 能力边界与后续优化路线
├── Dockerfile
├── docker-compose.yml
└── run.py
```

## 项目边界

Python Edition 的目标是提供一个可运行、可演示、可继续扩展的浏览器端 AI 编程 Agent 系统。

当前需要诚实区分三类：

- 已经可运行：FastAPI 后端、React 服务、QueryLoop、工具结果摘要事件、五层 ContextCascade 主链路、48+ 工具注册、FileEdit/MultiEdit 冲突检测/五策略 fuzzy/snapshot/diff/原子写入/history rewind/React History 面板、权限等待、LLM 调用、记忆检索、MCP 状态/stdio 与 http/streamable_http 真实 JSON-RPC/wrapped tools 动态注册/tools/call/content 提取/缓存降级/截断保护/schema validation/auth failure cache/reconnect backoff、验证证据、多 Agent Coordinator workflow 主链路。
- 已经有骨架但深度仍需补齐：完整 STOMP broker 语义、深层 Bash 沙箱的进程树/恢复策略和跨平台 parser 细节、ContextCascade 的细粒度增量折叠和事件字段完全一致性、插件隔离、Playwright 级浏览器验证、全量工具参数和错误恢复、超大并发多 Agent 压测。
- 尚未独立证明：SWE-bench Lite 官方 harness 结果。

详细差异和后续优化顺序见 [docs/python-source-differences.md](docs/python-source-differences.md)。

## 面试讲法

可以这样概括：

> 这个项目是一个 Python-native AI 编程 Agent 系统，核心能力包括 Agent loop、48+ 工具体系、权限审批、五层 ContextCascade、实时事件、记忆检索、MCP、多 Agent Coordinator workflow 和验证证据。当前版本已经能运行和演示主链路，同时也把深层安全沙箱、完整 STOMP 语义、ContextCascade 细粒度增量折叠/事件字段、高并发多 Agent 压测和 SWE-bench 复测列为后续优化路线。

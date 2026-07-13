# Python Edition 与源项目差异及后续优化路线

本文对照源项目 README 和 Java/Spring 实现，记录当前 Python Edition 的真实能力边界。它既是迁移验收清单，也是后续优化路线图。

状态说明：

- **已对齐核心逻辑**：Python 版已有可运行实现，并有基本回归测试或实际 UI/API 接线。
- **部分对齐**：Python 版已有主路径或能力骨架，但深度、稳定性、性能或测试覆盖不如源项目。
- **待补齐**：源项目 README 或 Java 实现中的重要能力，Python 版尚未达到 1:1。

## 总览

| 模块 | 源项目能力 | Python Edition 当前状态 | 差异等级 |
| --- | --- | --- | --- |
| 运行架构 | Java Spring Boot 核心编排 + React + Python service | FastAPI 核心编排 + React + Python service | 已按 Python 化目标替换 |
| React 前端 | 完整 React 体验、实时面板、可视化 | React 源码与构建产物恢复，由 Python backend 服务 | 部分对齐 |
| WebSocket/STOMP | Spring WebSocket + STOMP broker，细粒度事件 | SockJS/STOMP 兼容层，可推送 query/tool/permission/verify/swarm 事件，并支持 user queue、client ack/nack、REST replay、destination 过滤、订阅/退订生命周期、last-message-id 订阅恢复和轻量背压统计 | 部分对齐 |
| QueryEngine | 深层状态机、上下文恢复、自纠错、工具循环、终止策略 | Python QueryLoopState + tool execution + tool result summary events + ContextCascade + self-correction + termination decision | 部分对齐 |
| Token 计算 | TokenCounter + precise tokenizer + model ratio | 已恢复核心估算规则并接 tokenizer fallback | 已对齐核心逻辑 |
| 上下文压缩 | 五层压缩级联、增量折叠、413 恢复、媒体剥离 | 已接入五层 ContextCascade：snip、micro compact、auto compact、drain、reactive compact，并补 413 cause、超长 prompt recovery event、媒体剥离 metadata | 部分对齐 |
| LLM Provider | 多 Provider、模型注册、重试、降级链 | XFYun、DashScope、DeepSeek、Moonshot、Zhipu、MiniMax、ZenMux、通用 OpenAI-compatible fallback；已补 model-aware retry、错误分类和模型降级链配置 | 部分对齐 |
| 工具系统 | 48 内置工具 + MCP 动态扩展 | Python ToolRegistry 已注册 48+ 内置/兼容工具，覆盖文件、搜索、Git、命令、Agent、Task、Cron、LSP、Memory、Verify、上下文辅助 | 部分对齐 |
| Bash 安全 | 8 层 Bash 沙箱、错误分类、恢复策略、进程树管理 | 命令风险分类、敏感路径、权限策略、密钥环境变量审批、反向 shell/管道执行阻断、输出截断、退出码/超时分类 | 部分对齐 |
| 权限系统 | 14 步权限管道、批量审批、Signal | PermissionPolicy + WebSocket permission flow + Signal 部分恢复 | 部分对齐 |
| 多 Agent | Team / Swarm / SubAgent，Virtual Thread 并发，Coordinator 工作流 | Python asyncio/thread 编排、后台 Agent、显式 Coordinator workflow、Team/Swarm 状态、并发限制 | 部分对齐 |
| 记忆系统 | 三层记忆 + BM25 + rerank | Memdir BM25 + rerank + 分类记忆 | 已对齐核心逻辑 |
| MCP | SSE/stdio/http transport、资源、提示词、工具发现 | MCP state/resources/prompts/capabilities/local invocation、stdio 与 http/streamable_http 真实 JSON-RPC 连接、wrapped tools 动态注册到 ToolRegistry/LLM 工具列表、tools/call 语义、content 提取、结果缓存降级、1MB 截断保护、工具 schema 验证、鉴权失败缓存、重连退避状态 | 部分对齐 |
| Plugin/Skill | Java SPI 插件、热加载、沙箱隔离、Markdown skill | 文件系统 plugin/skill 发现、hook 执行 | 部分对齐 |
| Verification | Browser/HTTP/Auto verifier、证据链、JSONPath、前端实时进度 | VerifyRuntime、EvidenceBundle、HTTP/API verifier、Signal、前端面板 | 部分对齐 |
| 可视化 | Mermaid、Agent DAG、Git timeline、复杂度 treemap、代码路径图等 | React 组件保留，Python API 提供复杂度/路径/图/影响数据 | 部分对齐 |
| 数据持久化 | SQLite + migration + project/global schema | JSON state + SQLite sync layer | 部分对齐 |
| SWE-bench | 官方 46.3% Resolve 报告 | Python Edition 尚未重新跑官方 SWE-bench | 待补齐 |

## 已经比较接近源项目的部分

### 1. REST API 与 UI 骨架

Python 后端已经覆盖大量源项目 API 面：

- sessions：创建、列表、详情、删除、resume、compact、export、snapshot
- query：REST query、SSE query、WebSocket query、query loop events
- tools：工具列表、详情、启用状态、LLM tool definitions
- permissions：规则管理、审批响应
- memory：entries、分类、搜索、更新、删除
- MCP：servers、resources、prompts、capabilities、local invocation、stdio/http/streamable_http transport JSON-RPC、wrapped tools registration、tools/call invocation、content extraction、cache fallback、result truncation、schema validation、auth failure cache、reconnect backoff
- skills/plugins/hooks：发现、安装记录、reload、hook 执行
- files/git：文件树、搜索、上传、Git log/diff/blame
- verification/evidence：运行检查、证据包、blob 下载
- swarm/agent：团队、worker、权限等待、状态推送

主要差异在“深度行为”，不是“有没有 endpoint”。

### 2. Token 计算

Python 版已恢复源项目 `TokenCounter` 的核心机制：

- JSON/code/text/chinese 类型判断
- 中文比例影响 chars/token
- message list 按内容块估算，并加每条消息固定结构开销
- image token 估算
- `PRECISE_TOKENIZER` 开启时调用 `python-service /api/tokenizer/count`
- tokenizer 调用失败时 fallback 到启发式估算
- system + history + user + tool + memory 已进入主 QueryLoop token budget breakdown

后续可优化：

1. 接入更完整的 model capability registry，而不是只用当前 `tokenCharRatio`。
2. 对真实 OpenAI/DashScope/Kimi/XFYun token usage 做回写校准。
3. 把不同模型的多模态上下文窗口和图片 token 规则做成可配置模型能力。

### 3. Memdir 搜索与 rerank

Python 版已经有：

- BM25
- title boost
- CJK tokenization
- category filter
- procedural/team semantic boost
- updatedAt boost
- rerank boost

后续可优化：

1. 用真实 embedding/reranker 替代启发式 rerank。
2. 增加跨会话自动写入记忆策略。
3. 加入 memory aging、冲突合并、来源可信度。

### 4. P1 QueryEngine 可信度

已补强：

- 完整 context budget：system、history、user、tool、memory 已进入主 QueryLoop，并在事件中输出 breakdown。
- 主 QueryLoop self-correction loop：可解析 compile、pytest、JUnit、Jest 失败，进入 `self_correcting` phase，最多 3 次再提示模型修复。
- 模型错误分类与 fallback：区分 auth、rate_limit、overloaded、prompt_too_long、server、timeout、network，并支持 retryable 错误切换 fallback model。
- model-aware retry 与降级链：已补 `claude`、`qwen`、`deepseek` 对应的重试次数、指数退避间隔、Retry-After 策略和 `claude-sonnet-4-6 -> qwen3.7-max -> deepseek-v4-flash`、`qwen3.7-max -> qwen3.7-plus -> deepseek-v4-flash` 等源项目降级链；主 LLM 调用路径已接入该策略。
- 模型真实 usage 回写：OpenAI-compatible 响应中的 `prompt_tokens`、`completion_tokens`、cache token 等 usage 会规范化为 `inputTokens`、`outputTokens`、`cacheReadInputTokens`、`cacheCreationInputTokens`，并写入 session pending usage、assistant message、`message_complete`、`cost_update` 和 REST query 返回。
- tool retry：retryable 工具失败会记录 `TOOL_RETRY` recovery 和 `tool_retry` event。
- termination strategy：权限等待、正常结束、错误、用户中断会形成 `TerminationDecision`，并通过 `termination_decision` event 暴露。
- termination strategy 深层 stop 分支：已补 `max_tokens/length` 恢复生命周期（升级 maxTokens、注入续写提示、恢复上限）、`tool_use` 继续执行、`withhold` 扣留可恢复错误和 `model_stop_reason` 元数据保留。
- 五层 ContextCascade：主 `/api/query` 在上下文压力或显式 `collapseContext` 下执行 `snip_selection -> micro_compact -> auto_compact -> collapse_drain -> reactive_compact`，并通过 `compact_event`、`compact_complete`、`queryLoop.contextCascade` 暴露层级结果。
- ContextCascade 413 / 媒体剥离：主 QueryLoop 已支持 `recoveryCause=http_413`、`httpStatus=413` 和 `errorType=prompt_too_long` 触发专门恢复；`image/media/video/audio/file` blocks 会被替换成轻量占位文本，并在 reactive layer 与 recovery event 中记录 `mediaStrippedCount`、`estimatedMediaTokensFreed` 和 `recoveryCause`。
- 工具结果摘要：大工具输出保留原始 `result.content`，同时在 `QueryLoopState.toolCalls[].summary` 和实时 `tool_result` event 中写入 `summary/originalChars/lineCount/truncated`，减少前端和后续上下文直接处理超长输出的压力。

仍可继续优化：

1. 把 self-correction 的修复 prompt 与源项目 Java 实现逐条对齐。
2. 增加真实项目级编译/测试失败的端到端自修复用例。
3. 增加真实 Provider 故障注入和端到端 QueryLoop 回归，验证 retry / fallback / self-correction 串联行为。
4. 补齐 Java ContextCascade 中更细粒度增量折叠策略和事件字段完全一致性。

### 5. P2 多 Agent 深化

已补强：

- durable queued task 基础：`/api/swarm/{id}/tasks` 可追加并查看持久化 queued/running/completed 共享任务视图。
- worker phase barrier：worker 可上报 Research/Synthesis/Implementation/Verification 阶段，swarm 可检查 barrier 并在全员到达后 release。
- 显式 Coordinator workflow：Swarm 创建时会挂载 Research/Synthesis/Implementation/Verification 四阶段 workflow，phase barrier release 会自动推进 workflow，并通过 `workflow_phase_advanced` coordinator event 留痕。
- workflow 恢复：Swarm 内持久化的 `workflow` dict 可恢复回 `CoordinatorWorkflowEngine` active 状态，避免只依赖内存控制器。
- orphan worker recovery：启动时会自动扫描持久化运行态 worker；`POST /api/swarm/{id}/recover` 也可手动扫描。若 live asyncio task 已丢失，则把任务重建进 durable queue，并通过 `worker_requeued_after_recovery` coordinator event 留痕。
- phase-aware mailbox：mail 支持 phase/task/channel 元数据，mailbox 可按 phase/channel 过滤并局部 drain。
- mailbox ack/replay/recover：mailbox journal 支持确认、回放和恢复未 ack 消息。
- 多 Agent 权限冒泡与批量审批：支持全局/单 swarm pending permission batch allow/deny，pending 查询会返回 `expiresAt`、`remainingMs`、`elapsedMs`，并自动将超时请求 deny、释放 waiter、写入 worker 失败结果。
- TeamMailbox / SharedTaskList / LeaderPermissionBridge 事件字段：Coordinator event envelope 已补 `type`、`ts`、`uuid`、`sessionId`、`workflowId`、`swarmId`、`teamPrefix`、`eventType`、`payload`；mailbox 事件补 `messageId`、`contentLength`、`phaseIndex`、`mailboxDepth`；shared task 事件补 `teamName`、`creatorId`、`status`、`assigneeId`、`completedAt`；permission bubble 事件补 `riskLevel`、`timeoutMs`、`expiresAt`、`remainingMs`、`pendingRequestCount`、`leaderSessionId`。

仍可继续优化：

1. 增加 recovery 策略配置：是否自动启动、是否自动启动恢复任务、恢复批次上限和异常告警。
2. 增加大并发 worker 压力测试和超时预算排除策略。
3. 把 worker 自动上报阶段、任务分配策略和 Coordinator prompt 与源项目 Java 行为逐项对齐。
4. 继续补前端 Agent DAG / 移动端审批 / 批量审批失败重试提示的端到端展示语义。

## 部分对齐但需要深化的部分

### 1. React 完整体验

当前状态：

- React 源码已恢复。
- Python 后端会服务 `frontend/dist/index.html`。
- 主要 store、panel、visualization 组件仍存在。
- WebSocket/SockJS 兼容入口已恢复。

差异：

- 部分 UI 组件依赖的事件 payload 仍是 Python 简化版。
- 部分高级面板只显示基础状态，缺少源项目完整交互。
- 文件树、终端、Monaco、复杂图表等深层交互需要逐项回归验证。

后续优化：

1. 建立 React 页面级 Playwright 回归。
2. 为 Chat、Realtime、Activity、Verify、MCP、Memory、Swarm 各写一条端到端用例。
3. 对照源项目 STOMP event schema 补齐 payload 字段。
4. 恢复移动端审批、证据包联动、Agent DAG 的完整实时体验。

### 2. WebSocket / STOMP 实时链路

当前状态：

- Python 后端提供 SockJS/STOMP 风格入口。
- 可推送 query event、tool event、permission event、verification event、swarm event。
- `/user/queue/messages` 支持 `ack:auto` 和 `ack:client/client-individual` 语义。
- WebSocket manager 已支持消息 ID、投递计数、ACK、NACK、replay journal。
- REST 兼容接口已补 `deliver`、`ack`、`nack`、`replay`，便于断线恢复和测试工具验证。
- STOMP subscription lifecycle 已补：`SUBSCRIBE` 会注册 subscriptionId、destination、ack mode；`UNSUBSCRIBE` 会释放订阅；delivery/replay 会按 `/user/queue/*`、`/topic/*` destination 过滤，MESSAGE header 会保留真实 destination。
- STOMP session resume 已补轻量语义：`SUBSCRIBE` 支持 `last-message-id` / `lastMessageId` / `x-last-message-id`，会按 destination 只回放 last id 之后仍 pending 的消息，避免重连后跨 topic 串消息。
- 轻量 backpressure/order 已补：每个 session 的 queued message 会分配单调 `sequence`；pending queue 超过上限会丢弃最旧消息并累计 `droppedCount`；`/api/ws/sessions/{id}/messages/stats` 可查询 pending、journal、dropped、lastSequence 和 backpressure 状态。

差异：

- 没有 Spring STOMP broker 的完整语义。
- user destination 的完整 Spring 解析规则和断线重连背压处理仍未达到 Spring broker 级别。
- 一些前端 store 需要的高级事件字段仍需补齐。

后续优化：

1. 定义稳定的 Python event schema。
2. 继续对齐源项目 Spring STOMP broker 的 user destination 解析细节。
3. 增加 WebSocket reconnect/session resume 压力测试、并发订阅压力测试和更完整背压策略。
4. 增加并发订阅、消息顺序和背压测试。

### 3. 工具系统与安全沙箱

当前状态：

- Python ToolRegistry 已注册 48+ 内置/兼容工具，覆盖文件读写、目录/Glob/Grep、Git status/log/diff/show/blame、命令执行、REPL、LSP、Agent、Task、Cron、Memory、Verify、TokenCount、ContextStatus、ToolSearch、CommandClassify 等。
- 有危险命令 token、敏感路径、权限策略。
- 已补密钥环境变量读取审批、Windows 根目录递归删除阻断、敏感内容输出脱敏。
- 已补敏感文件/密钥环境变量读取后通过 curl/wget/PowerShell 上传的外传阻断。
- 已补 PowerShell `-EncodedCommand`、Base64 动态执行、`IEX + DownloadString/Invoke-WebRequest` 下载后执行阻断。
- FileEdit 已补 hash 冲突检测、源项目式五策略 fuzzy match（精确匹配之后依次尝试智能引号、尾部空白、换行符、Tab/空格、通用空白归一化）、同目录临时文件 + replace 原子写入；原子提交失败会保留原文件并清理临时文件。
- FileWrite/FileEdit/MultiEdit 工具结果已补 `snapshotBeforeWrite`、`beforeHash`、`afterHash` 和 unified `diff` metadata；主 QueryLoop 工具执行已把写入前快照接入全局 file history API，可通过 `history/snapshots` 查看、`history/diff?toMessageId=current` 对比当前工作区，并用 `history/rewind` 恢复。
- MultiEdit 已补 all-or-nothing 批量编辑：任一 edit 未命中会整体失败且不落盘，全部命中后一次原子写入，并返回每条 edit 的 `matchStrategies`。
- 命令执行现在会返回 sandbox metadata：`readOnly`、`category`、`exitCode`、`errorType`、`timedOut`、`durationMs`、输出字节数和截断标记；被策略阻断的命令也会返回 `blockLevel`、`blockReason` 和 `sandbox.decision=blocked`。
- 已补敏感文件读取、密钥环境变量读取、curl/wget 管道执行、反向 shell、密钥外传、编码 PowerShell 执行等风险识别测试。

差异：

- 源项目 README 宣称的 8 层 Bash 沙箱、进程树 kill、指数退避恢复、跨平台 shell parser，Python 版还没有完整 1:1。
- 48+ 工具已达到注册数量和主类目覆盖，FileEdit 核心可靠性已补强，但参数 schema、错误恢复、权限冒泡、前端展示字段还需要逐工具对照源项目验收。
- Notebook 深层编辑、安全测试覆盖和更多复杂编辑场景仍需补强。
- PowerShell/Bash 的平台细节目前比 Java 版简化。

后续优化：

1. 继续复刻 Bash/PowerShell parser，覆盖更完整的 quoting、subshell、process substitution、heredoc、redirect 和 Windows 参数边界。
2. 增加真实进程树 kill、错误恢复策略、指数退避和恢复建议。
3. React Sidebar 已补 History 面板，可展示工具级 snapshot、加载 diff 并触发 rewind；后续继续补 Playwright 级端到端浏览器回归。
4. 扩展安全回归测试：路径穿越、环境变量泄漏、危险命令、外部 root、进程树终止。

### 4. Verification 与 Evidence

当前状态：

- Python 版支持 verification request、evidence bundle/blob、Signal 计算。
- React Evidence/Verify 组件仍在。

差异：

- 源项目 BrowserVerifier / HttpApiVerifier / VerifierFactory 的 handler 类型更多。
- JSONPath 断言、浏览器截图、视频、HAR、console capture 的完整证据链仍需补强。
- 前端移动端审批联动需要端到端确认。

后续优化：

1. 接入 Playwright 浏览器验证。
2. 完整支持 screenshot / video / har / console / command / test / diff 七类证据。
3. 对 evidence blob 做内容寻址、去重和生命周期管理。
4. 为 VerifyJourneyPanel 写实时进度测试。

### 5. Plugin / Skill / MCP 深度

当前状态：

- skill/plugin/hook 有文件系统发现和注册。
- MCP 有 server 状态、resources、prompts、capabilities、local invocation、stdio 与 http/streamable_http 真实 JSON-RPC 连接、wrapped tools 动态注册到 ToolRegistry/LLM 工具列表、tools/call 调用语义、content 数组提取、非实时工具成功结果缓存降级、1MB 截断保护、工具 schema validation、auth failure cache 和 reconnect backoff 状态。

差异：

- Python 版不是 Java SPI 插件系统。
- 插件隔离、热重载安全、classloader 等 Java 特性没有直接等价物。
- MCP stdio 与 http/streamable_http transport 已有真实 JSON-RPC 连接与后端回归；SSE/websocket 真连接深度仍需继续对齐；tools/call 语义、结果提取/截断/缓存降级、auth/reconnect/backoff 已补状态机与后端回归，但还缺跨 transport 压测。

后续优化：

1. 设计 Python 插件 sandbox 与隔离加载机制。
2. 补齐 plugin manifest 校验、禁用、重载和安全测试。
3. 继续完整实现 MCP SSE/websocket transport 真连接行为；stdio 与 http/streamable_http 已有真实 JSON-RPC 回归。
4. 补 MCP 端到端 transport 回归、鉴权恢复策略和异常链路压测。

## 当前不应对外宣称已经 1:1 的能力

以下源 README 中的能力，Python Edition 目前不能直接照搬成“已完全支持”：

1. **SWE-bench Lite 46.3% 官方成绩**
   - 这是源项目评测结果。
   - Python Edition 还没有独立跑官方 harness。

2. **Java SPI 插件系统**
   - Python 版是 plugin manifest / filesystem discovery，不是 Java SPI。

3. **Java Virtual Thread 级多 Agent 并发**
   - Python 版有并发编排，但运行时机制不同。

4. **8 层 Bash 沙箱 + 308 项安全测试**
   - Python 版有安全策略雏形，但测试覆盖和沙箱深度未达到源项目宣称。

5. **Java 版 ContextCascade 的所有细节完全等价**
   - Python 版已有五层 ContextCascade 主链路，并已补媒体剥离、413 专门恢复、工具结果摘要事件和真实模型 usage 回写；但更细粒度增量折叠策略、Java 事件字段和端到端故障注入还未逐项 1:1。

6. **所有 48+ 工具行为完全等价**
   - Python ToolRegistry 已注册 48+ 内置/兼容工具，但行为深度、参数 schema、错误恢复、权限细节仍需逐项验收。

7. **完整 Browser semantic snapshot**
   - Python 版有 browser replay/evidence 相关接口，但 `/snap` 语义快照链路需要继续确认和补齐。

8. **完整国产多 Provider 前端切换体验**
   - Python 后端选择链路已支持 XFYun、DashScope、DeepSeek、Moonshot、Zhipu、MiniMax、ZenMux 和通用 OpenAI-compatible；前端模型选择、模型能力注册、按 Provider 分组展示还需要继续对齐。

## 建议后续优先级

### P0：面试演示稳定性

- 固定本地启动脚本，避免 Windows `Path/PATH` 环境问题。
- 确保 `python run.py`、Docker、React build 三条路径都可稳定启动。
- 准备一条可重复演示链路：提问 -> LLM 回复 -> 工具调用 -> 权限审批 -> 文件读取/修改 -> evidence/verify。

### P1：QueryEngine 深层一致性

- 在已补 `max_tokens`、`tool_use`、`withhold`、`model_stop_reason` 分支的基础上，增加真实 QueryLoop 端到端回归。
- 把 self-correction prompt 和源项目 Java 版逐条对齐。
- 增加真实 Provider 故障注入和端到端 QueryLoop 回归，验证 retry / fallback / self-correction 串联行为。
- 增加真实 repo 的端到端 query loop 回归。

### P2：多 Agent 深化

- 增加 durable recovery 策略配置和恢复任务自动启动选项。
- 增加 worker mailbox、shared task list、phase barrier、workflow advance 的并发压力测试。
- 继续细化移动端审批入口、Agent DAG 展示和前端批量审批失败重试提示。
- 在已补 `TeamMailbox`、`SharedTaskList`、`LeaderPermissionBridge` 核心事件字段的基础上，继续做端到端 UI 回归和高并发事件顺序测试。

### P3：安全与工具深度

- Bash/PowerShell sandbox 1:1。
- FileEdit 原子写入、冲突检测、fuzzy match。
- NotebookEdit 深层语义编辑。
- 安全测试矩阵。

### P4：评测与可证明性

- 重跑 Python Edition 的 SWE-bench Lite。
- 发布 Python 版评测报告，避免沿用源项目 Java 结果。
- 建立端到端 CI：backend tests + frontend build + WebSocket smoke + Playwright UI smoke。

## 面试时的诚实表述

推荐说法：

> 这个版本是对源项目的 Python 化复刻。我已经把核心运行路径迁到 FastAPI，包括 React UI 服务、QueryLoop 状态机、五层 ContextCascade 主链路、48+ 工具体系、权限、记忆检索、MCP、验证证据和多 Agent Coordinator workflow 主链路。它不是简单 demo，但也不是所有 Java 深层能力都 1:1 完成。差异主要集中在安全沙箱深度、Java ContextCascade 的细粒度增量折叠/事件字段、Spring STOMP broker 语义、Virtual Thread 级多 Agent 并发、Java SPI 插件隔离和 SWE-bench 复测。我把这些差异整理成路线图，方便后续逐项补齐。

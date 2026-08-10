# Architecture

## Request flow

```text
React UI
  -> HTTP / WebSocket
  -> FastAPI session layer
  -> QueryLoop
  -> LLM provider
  -> Tool registry
  -> Permission policy
  -> Tool execution
  -> Verification
  -> Event stream and final response
```

## Components

- `backend-python/`: API、会话、QueryLoop、工具、权限、记忆、MCP 和多 Agent 协作。
- `frontend/`: React/TypeScript 交互界面和 WebSocket 事件展示。
- `python-service/`: Token 估算与代码分析能力。
- `configuration/`: MCP 和运行配置模板。

## Safety boundary

文件写入、命令执行和 Git 操作在执行前经过工作区边界检查、风险分类和权限策略。当前版本面向本地受控工作区，不将应用层检查等同于容器或虚拟机级安全隔离。

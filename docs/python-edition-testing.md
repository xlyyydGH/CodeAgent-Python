# Python Edition 测试与可用性证明

本文记录当前项目的可用性验证口径、已实测结果和推荐复跑命令。目标不是声明全量测试全部通过，而是给出一组能快速证明项目可运行、核心 Agent 链路可用的小规模测试。

## 最近一次实测结论

测试日期：2026-07-02

| 类别 | 命令 | 结果 | 说明 |
| --- | --- | --- | --- |
| 后端语法检查 | `python -m py_compile backend-python\app.py ...` | 通过 | 覆盖 FastAPI 主入口和关键 runtime 模块 |
| 前端 TypeScript 编译 | `npx.cmd tsc --noEmit` | 通过 | 类型检查通过 |
| 前端生产构建 | `npx.cmd vite build` | 通过，`built in 36.90s` | 可生成 `frontend/dist` |
| 后端核心 runtime 小测试 | `pytest test_query_runtime.py test_mcp_runtime.py test_memdir_runtime.py test_websocket_runtime.py` | `31 passed` | 覆盖 QueryLoop、MCP、记忆检索、WebSocket runtime |
| FastAPI/WebSocket 小规模接口测试 | 精选 8 条接口用例 | `8 passed` | 覆盖 health、React root、fallback UI、query usage、MCP stdio/http、WS ack/replay |
| 前端 Vitest | `npm.cmd run test:run` | `18 passed` test files，`84 passed` tests，`16 skipped` | 覆盖 store、WebSocket、Agent DAG、History 面板等 |
| 真实服务启动烟测 | 启动 `uvicorn app:app` 后访问 `/api/health` 与 `/` | `/api/health=200`，`/=200 text/html` | 证明后端能启动并服务 React 根页面 |

注意：全量 `python -m pytest backend-python\tests -q` 曾在 4 分钟窗口内超时，因此本文不声明后端全量测试通过。面试或演示前建议优先跑本文的小规模测试集。

## 环境准备

后端 FastAPI 测试依赖当前仓库中的临时依赖缓存：

```powershell
$env:PYTHONPATH='.tmp\pytest-deps'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
```

前端依赖应已安装在 `frontend/node_modules`。如缺失：

```powershell
cd frontend
npm ci
cd ..
```

## 后端语法检查

```powershell
python -m py_compile backend-python\app.py backend-python\zhikun_py\query_runtime.py backend-python\zhikun_py\tools.py backend-python\zhikun_py\mcp_runtime.py backend-python\zhikun_py\websocket_runtime.py backend-python\zhikun_py\memdir_runtime.py backend-python\zhikun_py\verify_runtime.py backend-python\zhikun_py\sqlite_store.py
```

通过标准：命令退出码为 `0`，无语法错误输出。

## 后端核心 runtime 小测试

```powershell
$env:PYTHONPATH='.tmp\pytest-deps'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest backend-python\tests\test_query_runtime.py backend-python\tests\test_mcp_runtime.py backend-python\tests\test_memdir_runtime.py backend-python\tests\test_websocket_runtime.py -q
```

最近一次结果：

```text
31 passed, 1 warning in 4.52s
```

覆盖范围：

- QueryLoop / Token 预算 / ContextCascade / 工具结果摘要
- MCP stdio 与 http/streamable_http JSON-RPC 行为
- Memdir BM25 + rerank 检索
- WebSocket runtime 队列、ACK/NACK、replay 等基础行为

## FastAPI/WebSocket 小规模接口测试

```powershell
$env:PYTHONPATH='.tmp\pytest-deps'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest backend-python\tests\test_app.py::test_health backend-python\tests\test_app.py::test_react_frontend_root_when_built backend-python\tests\test_app.py::test_python_ui_fallback_page backend-python\tests\test_app.py::test_rest_query_returns_real_model_usage_in_message_and_events backend-python\tests\test_app.py::test_mcp_stdio_server_restart_discovers_and_invokes_real_transport backend-python\tests\test_app.py::test_mcp_streamable_http_server_restart_discovers_and_invokes_real_transport backend-python\tests\test_websocket_api.py::test_ws_session_ack_nack_and_replay_rest_contract backend-python\tests\test_websocket_api.py::test_sockjs_stomp_client_ack_removes_delivered_message -q
```

最近一次结果：

```text
8 passed, 1 warning in 60.67s
```

覆盖范围：

- `/api/health` 健康检查
- React root 服务能力
- Python fallback UI
- REST query usage 回写
- MCP stdio transport restart + invoke
- MCP http/streamable_http transport restart + invoke
- WebSocket ACK/NACK/replay REST contract
- SockJS-STOMP client ACK 行为

## 前端类型检查与构建

```powershell
cd frontend
npx.cmd tsc --noEmit
npx.cmd vite build
cd ..
```

最近一次结果：

```text
vite v5.4.21 building for production...
6247 modules transformed.
built in 36.90s
```

构建过程中可能出现 chunk size 或动态导入提示。这类提示不等于构建失败，只要退出码为 `0` 且出现 `built` 即可视为构建通过。

## 前端 Vitest

```powershell
cd frontend
npm.cmd run test:run
cd ..
```

最近一次结果：

```text
Test Files  18 passed (18)
Tests       84 passed | 16 skipped (100)
```

覆盖范围：

- stores 生命周期、路由边界、消息状态、配置状态、Swarm 状态
- WebSocket/STOMP client 基础行为
- Streaming text hook
- AgentDAGChart
- Browser replay timeline
- FileChangesDashboard 与 Sidebar History 面板

Vitest 输出中可能出现 Node `localStorage` experimental warning，以及未连接 WebSocket 的预期日志。这些不是失败，最终以 Vitest 的 passed/failed 汇总为准。

## 真实服务启动烟测

```powershell
$root=(Get-Location).Path
$backend=Join-Path $root 'backend-python'
$env:PYTHONPATH="$root\.tmp\pytest-deps;$backend"
$p=Start-Process -FilePath python -ArgumentList @('-m','uvicorn','app:app','--host','127.0.0.1','--port','8099') -WorkingDirectory $backend -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 15
Invoke-WebRequest 'http://127.0.0.1:8099/api/health' -UseBasicParsing
Invoke-WebRequest 'http://127.0.0.1:8099/' -UseBasicParsing
Stop-Process -Id $p.Id -Force
```

最近一次结果：

```text
HEALTH_STATUS=200
HEALTH_BODY={"status":"ok","service":"zhikuncode-python-backend","version":"0.1.0-python","pythonServiceUrl":"http://127.0.0.1:8000"}
ROOT_STATUS=200
ROOT_CONTENT_TYPE=text/html; charset=utf-8
```

## 面试前推荐测试顺序

1. 后端语法检查。
2. 后端核心 runtime 小测试。
3. FastAPI/WebSocket 小规模接口测试。
4. 前端 `tsc --noEmit`。
5. 前端 `vite build`。
6. 前端 `npm run test:run`。
7. 真实服务启动烟测。

这组测试能证明：项目能编译、能构建、后端能启动、React 页面能被服务、QueryLoop/MCP/Memory/WebSocket 等核心 Agent runtime 有回归覆盖。

## 当前测试边界

- 后端全量 pytest 曾在 4 分钟窗口内超时，建议后续拆分为更细的 CI job。
- Playwright E2E 本次未跑。
- Docker compose 端到端启动本次未跑。
- 真实外部模型调用本次未跑；LLM 相关测试主要覆盖 provider 选择、usage 回写、错误分类、retry/fallback 逻辑。
- 性能压测、高并发多 Agent 压测、安全沙箱全量矩阵本次未跑。

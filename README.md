# CodeAgent Python

基于 **FastAPI、React 与 WebSocket** 的浏览器端 AI 编程 Agent，支持代码检索、工具调用、权限审批、上下文管理、执行验证和多 Agent 协作。

> 基于 MIT 许可的 [ZhikunCode](https://github.com/zhikunqingtao/zhikuncode) 进行 Python-native 架构改写。

## 功能

- QueryLoop 编排模型调用、工具执行、结果回填与验证。
- 文件、搜索、Git、Shell、Memory、Verify 和 MCP 工具。
- Token 预算、长上下文压缩和工具结果摘要。
- 路径检查、命令风险分级、权限审批和输出脱敏。
- SubAgent、Team/Swarm、Mailbox 与结果聚合。
- WebSocket 实时事件和 Docker 部署。

## 运行

```bash
cp .env.example .env
docker compose up -d --build
```

访问 `http://localhost:8080`。模型配置通过环境变量注入，密钥不得提交到仓库。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest backend-python\tests -q
cd frontend
npm run build
```

详细设计见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。当前版本用于本地 Agent 工程验证，不等同于生产级代码执行沙箱。

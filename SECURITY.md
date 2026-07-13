# Security Policy

## Security Architecture

ZhikunCode incorporates multiple layers of security to protect users and their environments:

- **Bash Tool 8-Layer Security Check** — All shell command executions pass through an 8-layer validation pipeline including command parsing, blocklist filtering, path traversal detection, permission verification, sandbox enforcement, argument sanitization, output validation, and audit logging.
- **Permission Control Pipeline** — A structured pipeline governs tool execution permissions, ensuring that sensitive operations require explicit user approval before proceeding.
- **Path Safety Protection** — File system access is restricted to the designated workspace. Path normalization and traversal detection prevent unauthorized access to files outside the allowed scope.

## Reporting Vulnerabilities

**Please do NOT report security vulnerabilities through public GitHub Issues.**

If you discover a security vulnerability, please report it responsibly:

1. Email **alizhikun@gmail.com** with a detailed description
2. Include steps to reproduce the vulnerability
3. Allow reasonable time for a fix before public disclosure

We will acknowledge receipt within 48 hours and aim to provide an initial assessment within 7 business days.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < 1.0   | :x:                |

## Security Best Practices

When using ZhikunCode, we recommend:

- **Keep your `.env` file private** — never commit API keys or secrets to version control
- **Review tool permissions** — always review and approve sensitive operations before execution
- **Use workspace isolation** — run ZhikunCode within a dedicated project directory
- **Keep dependencies updated** — regularly update backend, frontend, and Python dependencies
- **Restrict network access** — in production, bind services to `localhost` or use a reverse proxy with authentication

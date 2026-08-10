# Security Policy

## Security Architecture

CodeAgent-Python incorporates multiple layers of security to protect users and their environments:

- **Bash Tool 8-Layer Security Check** — All shell command executions pass through an 8-layer validation pipeline including command parsing, blocklist filtering, path traversal detection, permission verification, sandbox enforcement, argument sanitization, output validation, and audit logging.
- **Permission Control Pipeline** — A structured pipeline governs tool execution permissions, ensuring that sensitive operations require explicit user approval before proceeding.
- **Path Safety Protection** — File system access is restricted to the designated workspace. Path normalization and traversal detection prevent unauthorized access to files outside the allowed scope.

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it through the repository's private security reporting channel. If that channel is unavailable, open a GitHub issue without including secrets or full exploit details, and request private follow-up.

1. Include a concise description and steps to reproduce
2. Do not publish API keys, tokens, personal data, or weaponized exploit code
3. Allow reasonable time for a fix before public disclosure

We will acknowledge receipt within 48 hours and aim to provide an initial assessment within 7 business days.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < 1.0   | :x:                |

## Security Best Practices

When using CodeAgent-Python, we recommend:

- **Keep your `.env` file private** — never commit API keys or secrets to version control
- **Review tool permissions** — always review and approve sensitive operations before execution
- **Use workspace isolation** — run CodeAgent-Python within a dedicated project directory
- **Keep dependencies updated** — regularly update backend, frontend, and Python dependencies
- **Restrict network access** — in production, bind services to `localhost` or use a reverse proxy with authentication

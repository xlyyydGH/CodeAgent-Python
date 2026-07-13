/**
 * generateUUID — 兼容非安全上下文的 UUID v4 生成器
 *
 * 问题背景:
 *   crypto.randomUUID() 仅在安全上下文（HTTPS / localhost）中可用。
 *   通过 HTTP + 公网 IP 访问时（如 Docker 部署到 ECS），浏览器将其视为
 *   非安全上下文，crypto.randomUUID() 抛出 TypeError 导致功能中断。
 *
 * 降级策略:
 *   1. 优先使用 crypto.randomUUID()（安全上下文，性能最优）
 *   2. 降级使用 crypto.getRandomValues()（非安全上下文也可用，随机性有保障）
 *
 * @see https://developer.mozilla.org/en-US/docs/Web/API/Crypto/randomUUID
 * @see https://developer.mozilla.org/en-US/docs/Web/API/Crypto/getRandomValues
 */
export function generateUUID(): string {
    // 安全上下文可用时优先使用标准 API
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        try {
            return crypto.randomUUID();
        } catch {
            // 某些浏览器在非安全上下文中 randomUUID 存在但调用时抛异常
        }
    }

    // 非安全上下文降级：crypto.getRandomValues 不要求安全上下文
    // 实现 RFC 4122 v4 UUID
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 10
    const hex = [...bytes].map(b => b.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

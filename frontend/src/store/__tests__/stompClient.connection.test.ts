import { beforeEach, describe, expect, test, vi } from 'vitest';

const mockActivate = vi.fn();
const mockDeactivate = vi.fn();
const mockPublish = vi.fn(() => {
    throw new TypeError('There is no underlying STOMP connection');
});

vi.mock('@stomp/stompjs', () => ({
    Client: vi.fn().mockImplementation(() => ({
        activate: mockActivate,
        deactivate: mockDeactivate,
        publish: mockPublish,
        active: true,
        connected: false,
    })),
}));

vi.mock('sockjs-client', () => ({
    default: vi.fn(),
}));

describe('stompClient connection readiness', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        vi.resetModules();
    });

    test('does not report connected or publish while STOMP is only active', async () => {
        const { createStompClient, isWsConnected, sendToServer } = await import('@/api/stompClient');

        createStompClient('session-test', '');

        expect(isWsConnected()).toBe(false);
        expect(sendToServer('/app/chat', { text: 'hello' })).toBe(false);
        expect(mockPublish).not.toHaveBeenCalled();
    });
});

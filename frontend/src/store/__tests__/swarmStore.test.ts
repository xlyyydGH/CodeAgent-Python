import { beforeEach, describe, expect, test, vi } from 'vitest';
import { useSwarmStore } from '@/store/swarmStore';

vi.mock('@/api/stompClient', () => ({
    send: vi.fn(),
    isConnected: vi.fn(() => false),
}));

beforeEach(() => {
    useSwarmStore.setState({
        swarms: new Map(),
        pendingPermissions: [],
        logs: [],
        activeSwarmId: null,
        panelVisible: false,
    });
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve({ success: true }) })));
});

describe('swarmStore permission bubble handling', () => {
    test('resolved permission_bubble removes the pending request instead of adding a duplicate', () => {
        useSwarmStore.getState().addPermissionBubble({
            type: 'permission_bubble',
            requestId: 'perm-1',
            swarmId: 'swarm-1',
            workerId: 'worker-1',
            toolName: 'write_file',
            riskLevel: 'high',
            reason: 'needs approval',
            timeoutMs: 5000,
            remainingMs: 4000,
            expiresAt: '2030-01-01T00:00:00Z',
        });

        useSwarmStore.getState().addPermissionBubble({
            type: 'permission_bubble',
            requestId: 'perm-1',
            swarmId: 'swarm-1',
            workerId: 'worker-1',
            toolName: 'write_file',
            riskLevel: 'high',
            reason: 'needs approval',
            resolved: true,
            decision: 'deny',
        });

        expect(useSwarmStore.getState().pendingPermissions).toHaveLength(0);
    });

    test('resolveAll sends one backend batch request for all pending permissions', () => {
        useSwarmStore.getState().addPermissionBubble({
            type: 'permission_bubble',
            requestId: 'perm-a',
            workerId: 'worker-1',
            toolName: 'write_file',
            riskLevel: 'high',
            reason: 'a',
        });
        useSwarmStore.getState().addPermissionBubble({
            type: 'permission_bubble',
            requestId: 'perm-b',
            workerId: 'worker-2',
            toolName: 'bash',
            riskLevel: 'high',
            reason: 'b',
        });

        useSwarmStore.getState().resolveAll('DENY');

        expect(fetch).toHaveBeenCalledTimes(1);
        expect(fetch).toHaveBeenCalledWith(
            '/api/swarm/permissions/batch',
            expect.objectContaining({
                method: 'POST',
                body: JSON.stringify({ requestIds: ['perm-a', 'perm-b'], decision: 'DENY' }),
            }),
        );
        expect(useSwarmStore.getState().pendingPermissions).toHaveLength(0);
    });
});

import React from 'react';
import { render } from '@testing-library/react';
import { beforeEach, describe, expect, test, vi } from 'vitest';
import App from '@/App';

const mocks = vi.hoisted(() => ({
  useWebSocket: vi.fn(),
  useAPOSInitialization: vi.fn(),
}));

vi.mock('@/hooks/useWebSocket', () => ({
  useWebSocket: mocks.useWebSocket,
}));

vi.mock('@/hooks/useAPOSInitialization', () => ({
  useAPOSInitialization: mocks.useAPOSInitialization,
}));

vi.mock('@/components/layout', () => ({
  AppLayout: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock('@/components/message', () => ({
  MessageList: () => <div data-testid="message-list" />,
}));

vi.mock('@/components/verify/JourneyVerifyPanel', () => ({
  JourneyVerifyPanel: () => null,
}));

vi.mock('@/components/input', () => ({
  PromptInput: () => <div data-testid="prompt-input" />,
}));

vi.mock('@/components/DialogManager', () => ({
  DialogManager: () => null,
}));

vi.mock('@/components/skills/SkillDetailModal', () => ({
  SkillDetailModal: () => null,
}));

vi.mock('@/components/verify/MobileApprovalSheet', () => ({
  MobileApprovalSheet: () => null,
}));

describe('App WebSocket lifecycle', () => {
  beforeEach(() => {
    mocks.useWebSocket.mockClear();
    mocks.useAPOSInitialization.mockClear();
    global.fetch = vi.fn(() => new Promise<Response>(() => {}));
  });

  test('starts the WebSocket client when the chat UI mounts', () => {
    render(<App />);

    expect(mocks.useWebSocket).toHaveBeenCalledTimes(1);
  });
});

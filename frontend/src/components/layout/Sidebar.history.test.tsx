import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Sidebar } from './Sidebar';
import { useSessionStore } from '@/store/sessionStore';

vi.mock('@/components/dashboard/FileChangesDashboard', () => ({
  FileChangesDashboard: ({ sessionId }: { sessionId: string }) => (
    <div data-testid="history-dashboard">history session {sessionId}</div>
  ),
}));

vi.mock('@/components/visualization/backend/APISequenceDiagram', () => ({
  APISequenceDiagram: () => <div />,
}));
vi.mock('@/components/layout/FileTreePanel', () => ({
  FileTreePanel: () => <div />,
}));
vi.mock('@/components/visualization/shared/AgentDAGChart', () => ({
  AgentDAGChart: () => <div />,
}));
vi.mock('@/components/visualization/shared/GitTimeline', () => ({
  GitTimeline: () => <div />,
}));
vi.mock('@/components/visualization/backend/CodeComplexityTreemap', () => ({
  CodeComplexityTreemap: () => <div />,
}));
vi.mock('@/components/visualization/backend/ChangeImpactGraph', () => ({
  ChangeImpactGraph: () => <div />,
}));
vi.mock('@/components/visualization/backend/APIContractViewer', () => ({
  APIContractViewer: () => <div />,
}));
vi.mock('@/components/visualization/backend/CodeDiagramGenerator', () => ({
  CodeDiagramGenerator: () => <div />,
}));
vi.mock('@/components/visualization/backend/CodePathTracer', () => ({
  CodePathTracer: () => <div />,
}));
vi.mock('@/components/apos/ActivityStream', () => ({
  ActivityStream: () => <div />,
}));
vi.mock('@/components/apos/FeatureFlagPanel', () => ({
  FeatureFlagPanel: () => <div />,
}));
vi.mock('@/components/apos/SessionFileExplorer', () => ({
  SessionFileExplorer: () => <div />,
}));
vi.mock('@/api/stompClient', () => ({
  sendToServer: vi.fn(),
}));

describe('Sidebar history tab', () => {
  beforeEach(() => {
    useSessionStore.setState({ sessionId: 'session-1', status: 'idle' });
    global.fetch = vi.fn(async () => new Response(JSON.stringify({ sessions: [], hasMore: false }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })) as unknown as typeof fetch;
  });

  it('opens the file history dashboard for the current session', () => {
    render(<Sidebar />);

    fireEvent.click(screen.getByTitle('History'));

    expect(screen.getByTestId('history-dashboard')).toHaveTextContent('session-1');
  });
});

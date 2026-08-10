import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { FileChangesDashboard } from './FileChangesDashboard';

describe('FileChangesDashboard', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('shows history snapshots, loads diff, and rewinds selected files', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/api/sessions/session-1/history/snapshots')) {
        return new Response(JSON.stringify({
          snapshots: [
            {
              messageId: 'write-history-1',
              trackedFiles: ['backend-python/.test-workspace/tool-history.py'],
              timestamp: '2026-07-01T10:00:00Z',
              files: [
                {
                  filePath: 'backend-python/.test-workspace/tool-history.py',
                  timestamp: '2026-07-01T10:00:00Z',
                },
              ],
            },
          ],
        }), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      if (url.includes('/api/sessions/session-1/history/diff')) {
        return new Response(JSON.stringify({
          files: [{ path: 'backend-python/.test-workspace/tool-history.py', status: 'modified' }],
          diff: '--- tool-history.py (before)\n+++ tool-history.py (after)\n@@\n-before\n+after',
        }), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      if (url.endsWith('/api/sessions/session-1/history/rewind') && init?.method === 'POST') {
        return new Response(JSON.stringify({
          success: true,
          restoredFiles: ['backend-python/.test-workspace/tool-history.py'],
          skippedFiles: [],
          errors: [],
        }), { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      return new Response('{}', { status: 404 });
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<FileChangesDashboard sessionId="session-1" />);

    const fileButton = await screen.findByRole('button', { name: /tool-history\.py/i });
    fireEvent.click(fileButton);

    expect(await screen.findByText('-before')).toBeInTheDocument();
    expect(screen.getByText('+after')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /rewind/i }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/sessions/session-1/history/rewind',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            messageId: 'write-history-1',
            filePaths: ['backend-python/.test-workspace/tool-history.py'],
          }),
        }),
      );
    });
    expect(await screen.findByText(/restored 1 file/i)).toBeInTheDocument();
  });
});

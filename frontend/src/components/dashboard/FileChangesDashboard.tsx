import { useCallback, useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Clock, FileText, GitCompare, Loader2, RotateCcw } from 'lucide-react';

interface HistoryFile {
  filePath: string;
  timestamp?: string;
}

interface HistorySnapshot {
  messageId: string;
  trackedFiles: string[];
  timestamp?: string;
  files?: HistoryFile[];
}

interface HistorySnapshotsResponse {
  snapshots?: HistorySnapshot[];
  byMessage?: Record<string, HistoryFile[]>;
}

interface DiffFile {
  path: string;
  status: string;
}

interface HistoryDiffResponse {
  files?: DiffFile[];
  diff?: string;
}

interface RewindResponse {
  success: boolean;
  restoredFiles?: string[];
  skippedFiles?: string[];
  errors?: string[];
}

interface SnapshotEntry {
  messageId: string;
  filePath: string;
  timestamp?: string;
}

function normalizeSnapshots(data: HistorySnapshotsResponse): SnapshotEntry[] {
  if (Array.isArray(data.snapshots)) {
    return data.snapshots.flatMap((snapshot) => {
      const files = snapshot.trackedFiles?.length
        ? snapshot.trackedFiles
        : (snapshot.files ?? []).map((file) => file.filePath).filter(Boolean);
      return files.map((filePath) => ({
        messageId: snapshot.messageId,
        filePath,
        timestamp: snapshot.timestamp ?? snapshot.files?.find((file) => file.filePath === filePath)?.timestamp,
      }));
    });
  }

  if (data.byMessage && typeof data.byMessage === 'object') {
    return Object.entries(data.byMessage).flatMap(([messageId, files]) =>
      (Array.isArray(files) ? files : []).map((file) => ({
        messageId,
        filePath: file.filePath,
        timestamp: file.timestamp,
      })),
    );
  }

  return [];
}

function fileName(path: string): string {
  const parts = path.replace(/\\/g, '/').split('/');
  return parts[parts.length - 1] || path;
}

function formatTime(value?: string): string {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function diffLineClass(line: string): string {
  if (line.startsWith('+') && !line.startsWith('+++')) return 'bg-green-500/10 text-green-300';
  if (line.startsWith('-') && !line.startsWith('---')) return 'bg-red-500/10 text-red-300';
  if (line.startsWith('@@')) return 'bg-blue-500/10 text-blue-300';
  return 'text-[var(--text-secondary)]';
}

export function FileChangesDashboard({ sessionId }: { sessionId: string }) {
  const [entries, setEntries] = useState<SnapshotEntry[]>([]);
  const [selected, setSelected] = useState<SnapshotEntry | null>(null);
  const [diff, setDiff] = useState<HistoryDiffResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [diffLoading, setDiffLoading] = useState(false);
  const [rewinding, setRewinding] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadSnapshots = useCallback(async () => {
    if (!sessionId) {
      setEntries([]);
      setSelected(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/sessions/${sessionId}/history/snapshots`);
      if (!response.ok) throw new Error(`Snapshot request failed: ${response.status}`);
      const data = await response.json() as HistorySnapshotsResponse;
      const nextEntries = normalizeSnapshots(data);
      setEntries(nextEntries);
      setSelected((current) => current ?? nextEntries[0] ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load history snapshots');
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    void loadSnapshots();
  }, [loadSnapshots]);

  const groupedEntries = useMemo(() => {
    const byFile = new Map<string, SnapshotEntry[]>();
    for (const entry of entries) {
      const list = byFile.get(entry.filePath) ?? [];
      list.push(entry);
      byFile.set(entry.filePath, list);
    }
    return Array.from(byFile.entries()).map(([path, items]) => ({
      path,
      items,
      latest: items[items.length - 1],
    }));
  }, [entries]);

  const loadDiff = useCallback(async (entry: SnapshotEntry) => {
    setSelected(entry);
    setDiff(null);
    setStatus(null);
    setError(null);
    setDiffLoading(true);
    try {
      const params = new URLSearchParams({ fromMessageId: entry.messageId, toMessageId: 'current' });
      const response = await fetch(`/api/sessions/${sessionId}/history/diff?${params.toString()}`);
      if (!response.ok) throw new Error(`Diff request failed: ${response.status}`);
      setDiff(await response.json() as HistoryDiffResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load diff');
    } finally {
      setDiffLoading(false);
    }
  }, [sessionId]);

  const rewindSelected = useCallback(async () => {
    if (!selected) return;
    setRewinding(true);
    setStatus(null);
    setError(null);
    try {
      const response = await fetch(`/api/sessions/${sessionId}/history/rewind`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messageId: selected.messageId, filePaths: [selected.filePath] }),
      });
      if (!response.ok) throw new Error(`Rewind request failed: ${response.status}`);
      const data = await response.json() as RewindResponse;
      if (!data.success) {
        throw new Error((data.errors ?? []).join('; ') || 'Rewind failed');
      }
      const restoredCount = data.restoredFiles?.length ?? 0;
      setStatus(`Restored ${restoredCount} ${restoredCount === 1 ? 'file' : 'files'}`);
      await loadSnapshots();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to rewind file');
    } finally {
      setRewinding(false);
    }
  }, [loadSnapshots, selected, sessionId]);

  const diffLines = diff?.diff ? diff.diff.split('\n') : [];

  if (!sessionId) {
    return (
      <div className="p-4 text-sm text-[var(--text-muted)]">
        No active session.
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--bg-secondary)]">
      <div className="border-b border-[var(--border)] p-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
          <GitCompare className="h-4 w-4" />
          History
        </div>
        <div className="mt-1 text-xs text-[var(--text-muted)]">
          {entries.length} snapshots
        </div>
      </div>

      {error && (
        <div className="m-3 flex items-start gap-2 rounded border border-red-500/40 bg-red-500/10 p-2 text-xs text-red-200">
          <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
      {status && (
        <div className="mx-3 mt-3 rounded border border-green-500/40 bg-green-500/10 p-2 text-xs text-green-200">
          {status}
        </div>
      )}

      <div className="grid min-h-0 flex-1 grid-rows-[minmax(120px,220px)_1fr]">
        <div className="overflow-y-auto border-b border-[var(--border)]">
          {loading ? (
            <div className="flex items-center gap-2 p-3 text-sm text-[var(--text-muted)]">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading history
            </div>
          ) : groupedEntries.length === 0 ? (
            <div className="p-3 text-sm text-[var(--text-muted)]">
              No file history yet.
            </div>
          ) : (
            groupedEntries.map(({ path, items, latest }) => (
              <button
                key={path}
                type="button"
                onClick={() => latest && loadDiff(latest)}
                className={`w-full border-b border-[var(--border)] px-3 py-2 text-left transition-colors hover:bg-[var(--bg-hover)] ${
                  selected?.filePath === path ? 'bg-blue-500/10' : ''
                }`}
              >
                <div className="flex items-center gap-2 text-sm text-[var(--text-primary)]">
                  <FileText className="h-4 w-4 flex-shrink-0" />
                  <span className="truncate">{fileName(path)}</span>
                  <span className="ml-auto text-xs text-[var(--text-muted)]">{items.length}x</span>
                </div>
                <div className="mt-1 flex items-center gap-1 text-xs text-[var(--text-muted)]">
                  <Clock className="h-3 w-3" />
                  <span className="truncate">{path}</span>
                </div>
                {latest?.timestamp && (
                  <div className="mt-1 text-xs text-[var(--text-muted)]">{formatTime(latest.timestamp)}</div>
                )}
              </button>
            ))
          )}
        </div>

        <div className="min-h-0 overflow-y-auto p-3">
          {!selected ? (
            <div className="text-sm text-[var(--text-muted)]">Select a file snapshot.</div>
          ) : (
            <div className="space-y-3">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium text-[var(--text-primary)]">{selected.filePath}</div>
                  <div className="text-xs text-[var(--text-muted)]">message {selected.messageId}</div>
                </div>
                <button
                  type="button"
                  onClick={rewindSelected}
                  disabled={rewinding}
                  className="inline-flex flex-shrink-0 items-center gap-1 rounded border border-[var(--border)] px-2 py-1 text-xs text-[var(--text-primary)] hover:bg-[var(--bg-hover)] disabled:opacity-50"
                >
                  {rewinding ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RotateCcw className="h-3.5 w-3.5" />}
                  Rewind
                </button>
              </div>

              {diffLoading ? (
                <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading diff
                </div>
              ) : diffLines.length > 0 ? (
                <pre className="overflow-x-auto rounded border border-[var(--border)] bg-[var(--bg-primary)] p-2 text-xs leading-5">
                  {diffLines.map((line, index) => (
                    <div key={`${index}-${line}`} className={diffLineClass(line)}>
                      {line || ' '}
                    </div>
                  ))}
                </pre>
              ) : (
                <div className="text-sm text-[var(--text-muted)]">No diff loaded.</div>
              )}

              {diff?.files?.length ? (
                <div className="space-y-1">
                  {diff.files.map((file) => (
                    <div key={`${file.path}-${file.status}`} className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
                      <span className="rounded bg-[var(--bg-tertiary)] px-1.5 py-0.5">{file.status}</span>
                      <span className="truncate">{file.path}</span>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

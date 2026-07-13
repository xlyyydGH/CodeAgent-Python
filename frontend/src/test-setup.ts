import '@testing-library/jest-dom';
import { enableMapSet } from 'immer';

// Enable Immer MapSet plugin for stores that use Map/Set
enableMapSet();

function createMemoryStorage(): Storage {
    const data = new Map<string, string>();
    return {
        get length() {
            return data.size;
        },
        clear: () => data.clear(),
        getItem: (key: string) => data.get(key) ?? null,
        key: (index: number) => Array.from(data.keys())[index] ?? null,
        removeItem: (key: string) => {
            data.delete(key);
        },
        setItem: (key: string, value: string) => {
            data.set(key, String(value));
        },
    };
}

if (typeof globalThis.localStorage === 'undefined') {
    const storage = createMemoryStorage();
    Object.defineProperty(globalThis, 'localStorage', {
        configurable: true,
        value: storage,
    });
    if (typeof window !== 'undefined') {
        Object.defineProperty(window, 'localStorage', {
            configurable: true,
            value: storage,
        });
    }
}

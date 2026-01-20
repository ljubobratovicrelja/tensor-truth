import { create } from "zustand";

interface SessionStore {
  activeSessionId: string | null;
  setActiveSessionId: (id: string | null) => void;
}

export const useSessionStore = create<SessionStore>((set) => ({
  activeSessionId: null,
  setActiveSessionId: (id) => set({ activeSessionId: id }),
}));

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface UIStore {
  // Persisted state
  sidebarOpen: boolean;
  sidebarWidth: number;
  theme: "light" | "dark" | "system";
  // Transient state (not persisted)
  headerHidden: boolean;

  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
  setSidebarWidth: (width: number) => void;
  setTheme: (theme: "light" | "dark" | "system") => void;
  setHeaderHidden: (hidden: boolean) => void;
}

export const useUIStore = create<UIStore>()(
  persist(
    (set) => ({
      sidebarOpen: true,
      sidebarWidth: 256,
      theme: "system",
      headerHidden: false,

      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
      setSidebarWidth: (width) => set({ sidebarWidth: width }),
      setTheme: (theme) => set({ theme }),
      setHeaderHidden: (hidden) => set({ headerHidden: hidden }),
    }),
    {
      name: "tensortruth-ui",
      // Only persist these fields (exclude headerHidden)
      partialize: (state) => ({
        sidebarOpen: state.sidebarOpen,
        sidebarWidth: state.sidebarWidth,
        theme: state.theme,
      }),
    }
  )
);

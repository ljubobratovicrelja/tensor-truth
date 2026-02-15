import { create } from "zustand";
import { persist } from "zustand/middleware";

interface UIStore {
  // Persisted state
  sidebarOpen: boolean;
  sidebarWidth: number;
  rightSidebarOpen: boolean;
  rightSidebarWidth: number;
  theme: "light" | "dark" | "system";
  // Transient state (not persisted)
  headerHidden: boolean;
  inputHidden: boolean;

  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
  setSidebarWidth: (width: number) => void;
  setRightSidebarOpen: (open: boolean) => void;
  toggleRightSidebar: () => void;
  setRightSidebarWidth: (width: number) => void;
  setTheme: (theme: "light" | "dark" | "system") => void;
  setHeaderHidden: (hidden: boolean) => void;
  setInputHidden: (hidden: boolean) => void;
}

export const useUIStore = create<UIStore>()(
  persist(
    (set) => ({
      sidebarOpen: true,
      sidebarWidth: 256,
      rightSidebarOpen: true,
      rightSidebarWidth: 320,
      theme: "system",
      headerHidden: false,
      inputHidden: false,

      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
      setSidebarWidth: (width) => set({ sidebarWidth: width }),
      setRightSidebarOpen: (open) => set({ rightSidebarOpen: open }),
      toggleRightSidebar: () =>
        set((state) => ({ rightSidebarOpen: !state.rightSidebarOpen })),
      setRightSidebarWidth: (width) => set({ rightSidebarWidth: width }),
      setTheme: (theme) => set({ theme }),
      setHeaderHidden: (hidden) => set({ headerHidden: hidden }),
      setInputHidden: (hidden) => set({ inputHidden: hidden }),
    }),
    {
      name: "tensortruth-ui",
      // Only persist these fields (exclude headerHidden)
      partialize: (state) => ({
        sidebarOpen: state.sidebarOpen,
        sidebarWidth: state.sidebarWidth,
        rightSidebarOpen: state.rightSidebarOpen,
        rightSidebarWidth: state.rightSidebarWidth,
        theme: state.theme,
      }),
    }
  )
);

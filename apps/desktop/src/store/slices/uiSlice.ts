import { StateCreator } from "zustand";
import { AppView } from "../types";

export interface UISlice {
  activeView: AppView;
  theme: "dark" | "light";
  appError: string | null;
  
  setActiveView: (view: AppView) => void;
  setTheme: (theme: "dark" | "light") => void;
  clearAppError: () => void;
}

export const createUISlice: StateCreator<UISlice> = (set) => ({
  activeView: "transcript",
  theme: window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark",
  appError: null,
  
  setActiveView: (view) => set({ activeView: view }),
  setTheme: (theme) => set({ theme }),
  clearAppError: () => set({ appError: null }),
});

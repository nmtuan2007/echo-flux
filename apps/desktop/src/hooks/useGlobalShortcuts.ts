import { useEffect } from "react";
import { emit } from "@tauri-apps/api/event";
import { register, unregisterAll } from "@tauri-apps/api/globalShortcut";
import { invoke } from "@tauri-apps/api/tauri";
import { useEngineStore } from "../store/engineStore";

export function useGlobalShortcuts(isOverlay: boolean, shortcutsEnabled: boolean): void {
  useEffect(() => {
    if (isOverlay) return;
    let cancelled = false;
    const syncShortcuts = async () => {
      try {
        await unregisterAll();
        if (!shortcutsEnabled || cancelled) return;
        await register("CommandOrControl+Shift+O", async () => {
          await invoke("toggle_overlay_window");
        });
        await register("CommandOrControl+Shift+M", () => {
          useEngineStore.getState().togglePipeline();
        });
        await register("CommandOrControl+Shift+S", () => {
          const state = useEngineStore.getState();
          const finalEntries = state.entries.filter((e) => e.isFinal && e.text.trim());
          const target =
            [...finalEntries].reverse().find((e) => e.source !== "mic") ||
            [...finalEntries].reverse()[0];
          if (target) {
            state.requestAssist(target.id);
          }
        });
        await register("CommandOrControl+Shift+U", () => {
          void emit("overlay_unlock");
        });
      } catch (e) {
        console.error("Failed to setup global shortcuts:", e);
      }
    };
    void syncShortcuts();
    return () => {
      cancelled = true;
      unregisterAll().catch(() => {});
    };
  }, [isOverlay, shortcutsEnabled]);
}

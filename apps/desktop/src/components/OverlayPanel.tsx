import { invoke } from "@tauri-apps/api/tauri";
import React, { useEffect, useState } from "react";
import { useEngineStore } from "../store/engineStore";
import "../styles/global.css";
import { TranscriptPanel } from "./TranscriptPanel";

export const OverlayPanel: React.FC = () => {
  const { config } = useEngineStore();
  const [isLocked, setIsLocked] = useState(false);

  useEffect(() => {
    document.documentElement.style.backgroundColor = "transparent";
    document.body.style.backgroundColor = "transparent";
    const root = document.getElementById("root");
    if (root) root.style.backgroundColor = "transparent";
  }, []);

  useEffect(() => {
    invoke("set_stealth_mode", { enabled: config.stealthMode }).catch(console.error);
  }, [config.stealthMode]);

  useEffect(() => {
    const SHORTCUT = "CommandOrControl+Shift+U";
    let registered = false;

    const setupUnlock = async () => {
      try {
        const { register, unregister, isRegistered } = await import("@tauri-apps/api/globalShortcut");
        // Unregister first to avoid "already registered" error on hot-reload / re-mount
        if (await isRegistered(SHORTCUT)) {
          await unregister(SHORTCUT);
        }
        await register(SHORTCUT, async () => {
          await invoke("set_ignore_cursor_events", { ignore: false });
          setIsLocked(false);
        });
        registered = true;
      } catch (e) {
        console.error("Failed to bind unlock shortcut", e);
      }
    };

    setupUnlock();

    return () => {
      if (registered) {
        import("@tauri-apps/api/globalShortcut").then(({ unregister }) => {
          unregister(SHORTCUT).catch(() => {});
        });
      }
    };
  }, []);

  const toggleLock = async () => {
    try {
      const newLocked = !isLocked;
      await invoke("set_ignore_cursor_events", { ignore: newLocked });
      setIsLocked(newLocked);
    } catch (e) {
      console.error("Failed to set ignore cursor events", e);
    }
  };

  const handleBackToMain = async () => {
    await invoke("revert_to_main_window").catch(console.error);
  };

  return (
    <div
      className={`overlay-wrapper ${isLocked ? "locked" : ""}`}
      style={{
        width: "100%",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        backgroundColor: isLocked ? "transparent" : "rgba(0,0,0,0.5)",
        transition: "background-color 0.2s",
        position: "relative",
        paddingTop: isLocked ? "0px" : "48px",
      }}
    >
      {!isLocked && (
        <div
          data-tauri-drag-region
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: "48px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0 12px",
            zIndex: 1000,
            background: "linear-gradient(to bottom, rgba(0,0,0,0.8), rgba(0,0,0,0))",
            cursor: "grab",
          }}
        >
          <div
            data-tauri-drag-region
            style={{
              color: "rgba(255,255,255,0.5)",
              fontSize: "13px",
              display: "flex",
              alignItems: "center",
              gap: "6px",
              pointerEvents: "none",
            }}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M5 9h14M5 15h14" />
            </svg>
            Drag to move
          </div>

          <div style={{ display: "flex", gap: "8px" }}>
            <button
              onClick={handleBackToMain}
              style={{
                background: "rgba(59, 130, 246, 0.8)",
                color: "#fff",
                border: "none",
                padding: "4px 12px",
                borderRadius: "4px",
                cursor: "pointer",
                fontSize: "12px",
                fontWeight: "bold",
              }}
            >
              ← Back to Main
            </button>
            <button
              onClick={toggleLock}
              title="Lock to click through to apps behind it. Unlock anytime with Ctrl+Shift+U"
              style={{
                background: "rgba(255,255,255,0.2)",
                color: "#fff",
                border: "none",
                padding: "4px 8px",
                borderRadius: "4px",
                cursor: "pointer",
                fontSize: "12px",
              }}
            >
              Lock (Click-Through)
            </button>
          </div>
        </div>
      )}

      {isLocked && (
        <div style={{ position: "absolute", top: 8, right: 8, zIndex: 1000 }}>
          <span
            style={{
              fontSize: "10px",
              color: "rgba(255,255,255,0.5)",
              textShadow: "1px 1px 2px #000",
            }}
          >
            Press Ctrl+Shift+U to Unlock
          </span>
        </div>
      )}

      <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
        <TranscriptPanel />
      </div>
    </div>
  );
};

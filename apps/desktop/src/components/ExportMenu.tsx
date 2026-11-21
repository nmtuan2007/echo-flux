import { useEffect, useRef, useState } from "react";
import {
  downloadFile,
  exportAsJSON,
  exportAsMarkdown,
  exportAsSRT,
  exportAsText,
  type ExportEntry,
} from "../utils/export";

export interface ExportMenuProps {
  entries: ExportEntry[];
  title?: string;
  summaryMarkdown?: string | null;
  buttonLabel?: string;
  onExportBackup?: () => void;
}

function sanitizeFilename(title?: string): string {
  return (
    (title || "echoflux-session")
      .toLowerCase()
      .replace(/[^a-z0-9-_]+/gi, "-")
      .replace(/^-|-$/g, "") || "echoflux-session"
  );
}

export function ExportMenu({
  entries,
  title,
  summaryMarkdown,
  buttonLabel = "Export ▾",
  onExportBackup,
}: ExportMenuProps) {
  const [open, setOpen] = useState(false);
  const [copiedAll, setCopiedAll] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  const handleExport = (fmt: "txt" | "srt" | "md" | "json") => {
    setOpen(false);
    const base = sanitizeFilename(title);
    if (fmt === "txt") {
      downloadFile(exportAsText(entries), `${base}.txt`);
    } else if (fmt === "srt") {
      downloadFile(exportAsSRT(entries), `${base}.srt`);
    } else if (fmt === "md") {
      downloadFile(
        exportAsMarkdown(entries, title || "EchoFlux Session", summaryMarkdown || null),
        `${base}.md`,
        "text/markdown;charset=utf-8",
      );
    } else {
      downloadFile(exportAsJSON(entries), `${base}.json`, "application/json;charset=utf-8");
    }
  };

  const handleCopyAll = async () => {
    setOpen(false);
    try {
      await navigator.clipboard.writeText(exportAsText(entries));
      setCopiedAll(true);
      setTimeout(() => setCopiedAll(false), 1500);
    } catch {
      // Clipboard unavailable
    }
  };

  return (
    <div className="export-dropdown" ref={menuRef}>
      <button
        type="button"
        className="btn btn-secondary btn-sm"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        title="Export or copy transcript"
      >
        {copiedAll ? "✓ Copied" : buttonLabel}
      </button>
      {open && (
        <div className="export-dropdown-menu" role="menu">
          <button type="button" role="menuitem" onClick={handleCopyAll}>
            Copy All Text
          </button>
          <button type="button" role="menuitem" onClick={() => handleExport("md")}>
            Export Markdown (.md)
          </button>
          <button type="button" role="menuitem" onClick={() => handleExport("txt")}>
            Export Plain Text (.txt)
          </button>
          <button type="button" role="menuitem" onClick={() => handleExport("srt")}>
            Export Subtitles (.srt)
          </button>
          <button type="button" role="menuitem" onClick={() => handleExport("json")}>
            Export JSON (.json)
          </button>
          {onExportBackup && (
            <button
              type="button"
              role="menuitem"
              onClick={() => {
                setOpen(false);
                onExportBackup();
              }}
            >
              Export Full Backup (.json)
            </button>
          )}
        </div>
      )}
    </div>
  );
}

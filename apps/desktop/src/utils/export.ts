export interface ExportEntry {
  text: string;
  translation: string | null;
  timestamp: number;
  speaker?: string;
  startMs?: number | null;
  endMs?: number | null;
}

export function exportAsText(entries: ExportEntry[]): string {
  return entries
    .map((entry) => {
      const time = formatTimestamp(entry.timestamp);
      const speaker = entry.speaker ? `[${entry.speaker}] ` : "";
      let line = `[${time}] ${speaker}${entry.text}`;
      if (entry.translation) {
        line += `\n         ${entry.translation}`;
      }
      return line;
    })
    .join("\n\n");
}

export function exportAsSRT(entries: ExportEntry[]): string {
  if (entries.length === 0) return "";
  const baseTimestamp = entries[0].timestamp;
  const firstStartMs = entries[0].startMs;
  return entries
    .map((entry, index) => {
      const hasExactSpan =
        typeof entry.startMs === "number" &&
        typeof entry.endMs === "number" &&
        entry.endMs > entry.startMs;
      const startSec =
        hasExactSpan && typeof firstStartMs === "number"
          ? Math.max(0, (entry.startMs! - firstStartMs) / 1000)
          : Math.max(0, entry.timestamp - baseTimestamp);
      const start = formatSRTOffset(startSec);
      const duration = hasExactSpan
        ? Math.max((entry.endMs! - entry.startMs!) / 1000, 0.4)
        : estimateDuration(entry.text);
      const end = formatSRTOffset(startSec + duration);

      const speaker = entry.speaker ? `${entry.speaker}: ` : "";
      let text = `${speaker}${entry.text}`;
      if (entry.translation) {
        text += `\n${entry.translation}`;
      }

      return `${index + 1}\n${start} --> ${end}\n${text}`;
    })
    .join("\n\n");
}

export function exportAsMarkdown(
  entries: ExportEntry[],
  title?: string,
  summaryMarkdown?: string | null,
): string {
  const sections: string[] = [];
  sections.push(`# ${title?.trim() || "EchoFlux Meeting Transcript"}`);
  if (summaryMarkdown?.trim()) {
    sections.push(`## Summary\n\n${summaryMarkdown.trim()}`);
  }
  const transcriptLines = entries.map((entry) => {
    const time = formatTimestamp(entry.timestamp);
    const speaker = entry.speaker ? `**${entry.speaker}** ` : "";
    let block = `- **[${time}]** ${speaker}${entry.text}`;
    if (entry.translation) {
      block += `\n  > ${entry.translation}`;
    }
    return block;
  });
  sections.push(`## Transcript\n\n${transcriptLines.join("\n")}`);
  return sections.join("\n\n");
}

export function exportAsJSON(entries: ExportEntry[]): string {
  return JSON.stringify(
    entries.map((entry) => ({
      timestamp: entry.timestamp,
      time: formatTimestamp(entry.timestamp),
      ...(entry.speaker ? { speaker: entry.speaker } : {}),
      text: entry.text,
      translation: entry.translation,
    })),
    null,
    2,
  );
}

export function downloadFile(content: string, filename: string, mimeType = "text/plain;charset=utf-8") {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function formatTimestamp(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatSRTOffset(offsetSeconds: number): string {
  const totalMs = Math.max(0, Math.round(offsetSeconds * 1000));
  const h = String(Math.floor(totalMs / 3_600_000)).padStart(2, "0");
  const m = String(Math.floor((totalMs % 3_600_000) / 60_000)).padStart(2, "0");
  const s = String(Math.floor((totalMs % 60_000) / 1000)).padStart(2, "0");
  const ms = String(totalMs % 1000).padStart(3, "0");
  return `${h}:${m}:${s},${ms}`;
}

function estimateDuration(text: string): number {
  const words = text.split(/\s+/).length;
  const wordsPerSecond = 2.5;
  return Math.max(words / wordsPerSecond, 1.0);
}

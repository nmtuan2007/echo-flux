export interface ExportEntry {
  text: string;
  translation: string | null;
  timestamp: number;
}

export function exportAsText(entries: ExportEntry[]): string {
  return entries
    .map((entry) => {
      const time = formatTimestamp(entry.timestamp);
      let line = `[${time}] ${entry.text}`;
      if (entry.translation) {
        line += `\n         ${entry.translation}`;
      }
      return line;
    })
    .join("\n\n");
}

export function exportAsSRT(entries: ExportEntry[]): string {
  return entries
    .map((entry, index) => {
      const start = formatSRTTime(entry.timestamp);
      const duration = estimateDuration(entry.text);
      const end = formatSRTTime(entry.timestamp + duration);

      let text = entry.text;
      if (entry.translation) {
        text += `\n${entry.translation}`;
      }

      return `${index + 1}\n${start} --> ${end}\n${text}`;
    })
    .join("\n\n");
}

export function exportAsJSON(entries: ExportEntry[]): string {
  return JSON.stringify(
    entries.map((entry) => ({
      timestamp: entry.timestamp,
      time: formatTimestamp(entry.timestamp),
      text: entry.text,
      translation: entry.translation,
    })),
    null,
    2,
  );
}

export function downloadFile(content: string, filename: string) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
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

function formatSRTTime(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  const h = String(date.getUTCHours()).padStart(2, "0");
  const m = String(date.getUTCMinutes()).padStart(2, "0");
  const s = String(date.getUTCSeconds()).padStart(2, "0");
  const ms = String(date.getUTCMilliseconds()).padStart(3, "0");
  return `${h}:${m}:${s},${ms}`;
}

function estimateDuration(text: string): number {
  const words = text.split(/\s+/).length;
  const wordsPerSecond = 2.5;
  return Math.max(words / wordsPerSecond, 1.0);
}

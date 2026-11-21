export interface ExportEntry {
  text: string;
  translation: string | null;
  timestamp: number;
  speaker?: string;
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
  return entries
    .map((entry, index) => {
      const start = formatSRTTime(entry.timestamp);
      const duration = estimateDuration(entry.text);
      const end = formatSRTTime(entry.timestamp + duration);

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

export interface SuggestionOption {
  strategy: string;
  text: string;
}

export interface SuggestionResult {
  options?: SuggestionOption[];
  error?: string;
}

export interface TranscriptEntry {
  id: string;
  text: string;
  translation: string | null;
  isFinal: boolean;
  timestamp: number;
  source?: "mic" | "system" | "both" | null;
  suggestions?: SuggestionResult; 
}

export interface Conversation {
  id: string;
  name: string;
  date: number;
  entries: TranscriptEntry[];
  summary?: string; 
}

export interface AudioDeviceInfo {
  id: string;
  name: string;
}

export interface HubSearchResult {
  id: string;
  downloads: number;
  tags: string[];
  task: string;
}

export interface ModelItem {
  id: string;
  name: string;
  size_mb: number;
  runtime?: string;
  is_downloaded: boolean;
}

export interface EngineConfig {
  host: string;
  port: number;
  modelSize: string;
  language: string;
  device: string;
  translationEnabled: boolean;
  translationBackend: "online" | "marian";
  translationModel: string;
  sourceLang: string;
  targetLang: string;
  vadEnabled: boolean;
  // Dual audio capture
  audioSource: "microphone" | "system" | "both";
  micDeviceId: string;
  speakerDeviceId: string;
  // LLM / AI Assistant
  llmEnabled: boolean;
  llmProviderUrl: string;
  llmApiKey: string;
  llmModel: string;
  stealthMode: boolean;
  hfToken: string;
}

export type AppView = "transcript" | "settings" | "history" | "model_manager";

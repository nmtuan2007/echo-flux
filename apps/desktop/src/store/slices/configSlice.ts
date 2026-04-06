import { StateCreator } from "zustand";
import { EngineConfig } from "../types";

export interface ConfigSlice {
  config: EngineConfig;
  updateConfig: (partial: Partial<EngineConfig>) => void;
}

export const DEFAULT_CONFIG: EngineConfig = {
  host: "127.0.0.1",
  port: 8765,
  modelSize: "small",
  language: "en",
  device: "auto",
  translationEnabled: true,
  translationBackend: "online",
  translationModel: "",
  sourceLang: "en",
  targetLang: "vi",
  vadEnabled: true,
  audioSource: "microphone",
  micDeviceId: "",
  speakerDeviceId: "",
  llmEnabled: false,
  llmProviderUrl: "https://openrouter.ai/api/v1",
  llmApiKey: "",
  llmModel: "openai/gpt-4o-mini",
  stealthMode: false,
  hfToken: "",
};

export const createConfigSlice: StateCreator<ConfigSlice> = (set) => ({
  config: { ...DEFAULT_CONFIG },
  updateConfig: (partial) =>
    set((state) => ({
      config: { ...state.config, ...partial },
    })),
});

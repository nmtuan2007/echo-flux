import { useEffect, useState } from "react";
import { useEngineStore } from "../store/engineStore";

export interface LocalPackStatusProps {
  capability: "stt" | "translation";
  onStatusChange?: (isReady: boolean) => void;
}

export function LocalPackStatus({ capability, onStatusChange }: LocalPackStatusProps) {
  const {
    packStatus,
    hardwareInfo,
    checkPackReadiness,
    installLocalPack,
    cancelLocalPack,
    retryLocalPack,
    isPackChecking,
    downloading,
    downloadPercent,
    downloadModel,
  } = useEngineStore();

  const [installing, setInstalling] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    void checkPackReadiness(capability);
  }, [capability, checkPackReadiness]);

  const item = packStatus[capability];
  const isReady = Boolean(item?.isReady);

  useEffect(() => {
    onStatusChange?.(isReady);
  }, [isReady, onStatusChange]);

  const handleDownload = async () => {
    setErrorMsg(null);
    setInstalling(true);
    const packId = item?.packId || (capability === "stt" ? "whisper-base" : "marian-en-vi");
    const ok = await installLocalPack(packId);
    setInstalling(false);
    if (!ok) {
      setErrorMsg("Download failed or paused. You can resume at any time.");
    }
  };

  const handleCancel = async () => {
    const packId = item?.packId || (capability === "stt" ? "whisper-base" : "marian-en-vi");
    await cancelLocalPack(packId);
    setInstalling(false);
  };

  const handleRetry = async () => {
    setErrorMsg(null);
    setInstalling(true);
    const packId = item?.packId || (capability === "stt" ? "whisper-base" : "marian-en-vi");
    const ok = await retryLocalPack(packId);
    setInstalling(false);
    if (!ok) {
      setErrorMsg("Retry failed. Check network connection.");
    }
  };

  // Hardware tier recommendation calculation
  const cpuCores = hardwareInfo?.cpuCores ?? 8;
  const ramGb = hardwareInfo?.ramGb ?? 16;
  const hasGpu = hardwareInfo?.hasGpu ?? false;

  let recommendedTier = "Standard Tier (Balanced, Recommended)";
  if (hasGpu || (cpuCores >= 8 && ramGb >= 16)) {
    recommendedTier = "Advanced Tier (High Accuracy, GPU/High-CPU)";
  } else if (cpuCores <= 4 || ramGb <= 8) {
    recommendedTier = "Minimal Tier (Lightweight, Low Memory)";
  }

  const title = capability === "stt" ? "Speech-to-Text (Local Whisper)" : "Translation (Local Marian MT)";
  const sizeMb = item?.sizeBytes ? Math.round(item.sizeBytes / (1024 * 1024)) : capability === "stt" ? 145 : 250;

  return (
    <div className="pack-status-card local-pack-status-card" data-capability={capability}>
      <div className="pack-status-header">
        <div className="pack-status-title-row">
          <span className="pack-status-icon">{capability === "stt" ? "🎙️" : "🌐"}</span>
          <h4 className="pack-status-title">{title}</h4>
        </div>
        <div className="pack-status-badge-container">
          {isPackChecking ? (
            <span className="pack-badge pack-badge-checking">Checking...</span>
          ) : isReady ? (
            <span className="pack-badge pack-badge-ready">✓ Ready</span>
          ) : (
            <span className="pack-badge pack-badge-missing">Not Installed</span>
          )}
        </div>
      </div>

      <div className="pack-status-details">
        <p className="pack-desc">
          {capability === "stt"
            ? "Faster-Whisper on-device acoustic model for private audio transcription without cloud latency."
            : "Marian neural machine translation model running locally for real-time bilingual translation."}
        </p>

        {hardwareInfo && (
          <div className="hardware-info-box">
            <span className="hardware-spec">
              System: {cpuCores} Cores · {ramGb} GB RAM {hasGpu ? "· GPU Available" : ""}
            </span>
            <span className="hardware-tier-recommendation">Recommended: {recommendedTier}</span>
          </div>
        )}

        {(installing || downloading) && (
          <div className="pack-progress-area">
            <div className="pack-progress-meta">
              <span>Downloading model pack ({downloadModel || title})...</span>
              <span>{downloadPercent}%</span>
            </div>
            <div className="pack-progress-bar">
              <div className="pack-progress-fill" style={{ width: `${downloadPercent}%` }} />
            </div>
            <div className="pack-progress-actions">
              <button className="btn btn-secondary btn-sm" onClick={handleCancel}>
                Cancel
              </button>
            </div>
          </div>
        )}

        {errorMsg && (
          <div className="pack-error-alert" role="alert">
            <span>⚠️ {errorMsg}</span>
            <button className="btn btn-secondary btn-sm" onClick={handleRetry}>
              Resume Download
            </button>
          </div>
        )}

        {!isReady && !installing && !downloading && !errorMsg && (
          <div className="pack-action-row">
            <button className="btn btn-primary pack-download-btn" onClick={handleDownload}>
              Download Model Pack (~{sizeMb} MB)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

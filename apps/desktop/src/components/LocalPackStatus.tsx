import { useEffect, useState } from "react";
import { useEngineStore } from "../store/engineStore";

export interface LocalPackStatusProps {
  capability: "stt" | "translation";
  disabled?: boolean;
  onStatusChange?: (isReady: boolean) => void;
}

function formatSize(bytes: number): string {
  const mb = Math.round(bytes / (1024 * 1024));
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb} MB`;
}

export function LocalPackStatus({
  capability,
  disabled = false,
  onStatusChange,
}: LocalPackStatusProps) {
  const {
    connected,
    packStatus,
    packDownloads,
    packInstallErrors,
    hardwareInfo,
    services,
    config,
    checkPackReadiness,
    installLocalPack,
    cancelLocalPack,
    retryLocalPack,
    isPackChecking,
  } = useEngineStore();

  const [installing, setInstalling] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const sourceLang =
    (services?.translation?.options?.source_lang as string) || config.sourceLang || "en";
  const targetLang =
    (services?.translation?.options?.target_lang as string) || config.targetLang || "vi";

  useEffect(() => {
    void checkPackReadiness(capability);
  }, [capability, connected, checkPackReadiness, sourceLang, targetLang]);

  const item = packStatus[capability];
  const isReady = Boolean(item?.isReady);
  const languageSupported = item?.languageSupported !== false;
  const packId =
    item?.packId ||
    (capability === "stt" ? "legacy-faster-whisper-small" : "legacy-marian-opus-mt");
  const activeProgress = packDownloads[packId];
  const anotherDownloadActive =
    !activeProgress && Object.keys(packDownloads).length > 0;
  const isDownloadingThis = installing || Boolean(activeProgress);
  const percent = activeProgress?.percent ?? 0;
  const activeModelName = activeProgress?.model;
  const activeStage = activeProgress?.stage;
  const storeErrorMsg = packInstallErrors[packId];
  const displayError = errorMsg || storeErrorMsg || null;

  useEffect(() => {
    onStatusChange?.(isReady);
  }, [isReady, onStatusChange]);

  const langOptions =
    capability === "translation"
      ? { sourceLang, targetLang }
      : undefined;

  const handleDownload = async () => {
    setErrorMsg(null);
    setInstalling(true);
    const ok = await installLocalPack(packId, langOptions);
    if (ok) {
      await checkPackReadiness(capability);
    } else if (!useEngineStore.getState().packInstallErrors[packId]) {
      setErrorMsg("Download failed or cancelled.");
    }
    setInstalling(false);
  };

  const handleCancel = async () => {
    await cancelLocalPack(packId);
    setInstalling(false);
  };

  const handleRetry = async () => {
    setErrorMsg(null);
    setInstalling(true);
    const ok = await retryLocalPack(packId, langOptions);
    if (ok) {
      await checkPackReadiness(capability);
    } else if (!useEngineStore.getState().packInstallErrors[packId]) {
      setErrorMsg("Retry failed. Check network connection.");
    }
    setInstalling(false);
  };

  const cpuCores = hardwareInfo?.cpuCores ?? 8;
  const ramGb = hardwareInfo?.ramGb ?? 16;
  const hasGpu = hardwareInfo?.hasGpu ?? false;
  const isLegacyPack = packId.startsWith("legacy-");

  const title =
    capability === "stt" ? "Offline Speech Model (Whisper)" : "Offline Translation Model (Marian)";
  const sizeLabel = formatSize(
    item?.sizeBytes || (capability === "stt" ? 500_000_000 : 300_000_000),
  );

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
            ? "Runs Faster-Whisper locally on your CPU/GPU with zero cloud latency."
            : `Offline Marian MT model (${sourceLang === "auto" ? "auto" : sourceLang} → ${targetLang}).`}
        </p>

        {hardwareInfo && (
          <div className="hardware-info-box">
            <span className="hardware-spec">
              System: {cpuCores} Cores · {ramGb} GB RAM {hasGpu ? "· GPU Available" : "· CPU Mode"}
            </span>
            {!isLegacyPack && (
              <span className="hardware-tier-recommendation">Tier: {item?.tier || "standard"}</span>
            )}
          </div>
        )}

        {isDownloadingThis && (
          <div className="pack-progress-area">
            <div className="pack-progress-meta">
              <span>
                {activeStage === "converting"
                  ? "Converting model (CTranslate2)..."
                  : `Downloading (${activeModelName || title})...`}
              </span>
              <span>{percent}%</span>
            </div>
            <div className="pack-progress-bar">
              <div className="pack-progress-fill" style={{ width: `${percent}%` }} />
            </div>
            <div className="pack-progress-actions">
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={handleCancel}
                disabled={disabled}
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {displayError && (
          <div className="pack-error-alert" role="alert">
            <span>⚠️ {displayError}</span>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={handleRetry}
              disabled={disabled || anotherDownloadActive}
            >
              Retry Download
            </button>
          </div>
        )}

        {!languageSupported && capability === "translation" && (
          <div className="pack-error-alert" role="status">
            <span>
              ⚠ Offline Marian does not support{" "}
              <code>{sourceLang === "auto" ? "auto" : sourceLang} → {targetLang}</code>. Choose
              a supported pair (e.g. en → vi/ja/ko/zh/de/fr/es, vi → en) or switch to a Cloud
              Translation provider.
            </span>
          </div>
        )}

        {!isReady && !isDownloadingThis && !displayError && languageSupported && (
          <div className="pack-action-row">
            <button
              type="button"
              className="btn btn-primary pack-download-btn"
              onClick={handleDownload}
              disabled={disabled || anotherDownloadActive}
              title={anotherDownloadActive ? "Wait for the current model download to finish" : undefined}
            >
              {anotherDownloadActive
                ? "Waiting for other download..."
                : `Download Model Pack (~${sizeLabel})`}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

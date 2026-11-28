import { useEffect, useState } from "react";
import { useEngineStore } from "../store/engineStore";
import { resolveEffectiveTranslationSourceLang } from "../store/providerCatalog";

export interface LocalPackStatusProps {
  capability: "stt" | "translation";
  disabled?: boolean;
  allowRemove?: boolean;
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
  allowRemove = false,
  onStatusChange,
}: LocalPackStatusProps) {
  const {
    connected,
    engineStarting,
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
    deleteLocalPack,
    isPackChecking,
  } = useEngineStore();

  const [installing, setInstalling] = useState(false);
  const [removing, setRemoving] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const sourceLang = resolveEffectiveTranslationSourceLang(services, config);
  const targetLang =
    (services?.translation?.options?.target_lang as string) || config.targetLang || "vi";
  const qualityPreset =
    (services?.stt?.options?.quality_preset as string) || "balanced";

  useEffect(() => {
    void checkPackReadiness(capability);
  }, [capability, connected, checkPackReadiness, sourceLang, targetLang, qualityPreset]);

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

  const handleRemove = async () => {
    setErrorMsg(null);
    setRemoving(true);
    const ok = await deleteLocalPack(capability, { packId, ...(langOptions || {}) });
    setConfirmRemove(false);
    if (ok) {
      await checkPackReadiness(capability);
    } else {
      setErrorMsg("Failed to remove local model files.");
    }
    setRemoving(false);
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
          {!connected && !item ? (
            <span className="pack-badge pack-badge-checking">
              {engineStarting ? "Starting Engine..." : "Engine Offline"}
            </span>
          ) : isPackChecking ? (
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

        {!isReady && !isDownloadingThis && !displayError && languageSupported && connected && (
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

        {allowRemove && isReady && !isDownloadingThis && connected && (
          <div className="pack-action-row">
            {confirmRemove ? (
              <>
                <button
                  type="button"
                  className="btn btn-danger btn-sm"
                  onClick={handleRemove}
                  disabled={disabled || removing}
                >
                  {removing ? "Removing..." : "Confirm Remove"}
                </button>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => setConfirmRemove(false)}
                  disabled={removing}
                >
                  Cancel
                </button>
              </>
            ) : (
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setConfirmRemove(true)}
                disabled={disabled || removing}
                title="Delete cached local model files to reclaim disk space"
              >
                Remove Cached Model
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

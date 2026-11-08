//! Tauri-owned credential vault.
//!
//! Secrets live in the OS vault (Windows Credential Manager, macOS Keychain). The webview
//! may save, replace, delete, and ask whether a credential is configured; nothing exposed to
//! it returns secret text. `CredentialStore::read_secret` is Rust-only and exists for the
//! future privileged control bridge that forwards provider config to the engine.
//!
//! Errors crossing to the webview carry a fixed code and message. They never echo the
//! secret, the full credential id, or text from the platform/library error, because
//! platform messages can include target names.

use std::sync::{Arc, Mutex};

use serde::Serialize;
use zeroize::{Zeroize, Zeroizing};

/// Service component of every vault entry. On Windows keyring stores the entry under the
/// generic target `<account>.<service>`; on macOS it is the Keychain service name.
const SERVICE_NAME: &str = "com.echoflux.app";

const MAX_SEGMENT_LEN: usize = 48;
const MAX_SECRET_BYTES: usize = 4096;

/// A lowercase alphanumeric run this long, with no `-`/`_` separator, looks like key
/// material rather than a human-chosen configuration name.
const MAX_UNSEPARATED_RUN: usize = 24;

/// Well-known provider key prefixes. An id segment starting with one of these is almost
/// certainly a pasted key, so it is rejected before it can become a vault target name.
const KEY_LIKE_PREFIXES: &[&str] = &["sk-", "sk_", "aiza", "gsk_", "xai-", "ghp_", "bearer"];

/// Below this length a segment is too generic for the "secret contains the id" check to
/// mean anything (a key could legitimately contain `stt` or `default`).
const MIN_SEGMENT_OVERLAP_CHECK: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Capability {
    Stt,
    Translation,
    Assistant,
}

impl Capability {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "stt" => Some(Self::Stt),
            "translation" => Some(Self::Translation),
            "assistant" => Some(Self::Assistant),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Stt => "stt",
            Self::Translation => "translation",
            Self::Assistant => "assistant",
        }
    }
}

/// Stable identity of one stored secret: which service it belongs to, which configuration
/// of that service, and which secret slot within the configuration (e.g. `api_key`).
#[derive(Clone, PartialEq, Eq)]
pub struct CredentialId {
    capability: Capability,
    config_id: String,
    slot: String,
}

// Manual Debug so a stray `{:?}` in a log can never print the full id.
impl std::fmt::Debug for CredentialId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CredentialId")
            .field("capability", &self.capability)
            .finish_non_exhaustive()
    }
}

impl CredentialId {
    pub fn parse(capability: &str, config_id: &str, slot: &str) -> Result<Self, CredentialError> {
        let capability = Capability::parse(capability)
            .ok_or_else(|| CredentialError::new(ErrorCode::InvalidCapability, None))?;
        let invalid = || CredentialError::new(ErrorCode::InvalidCredentialId, Some(capability));
        if !is_valid_segment(config_id) || !is_valid_segment(slot) {
            return Err(invalid());
        }
        Ok(Self {
            capability,
            config_id: config_id.to_owned(),
            slot: slot.to_owned(),
        })
    }

    fn account(&self) -> String {
        format!("{}:{}:{}", self.capability.as_str(), self.config_id, self.slot)
    }
}

fn is_valid_segment(segment: &str) -> bool {
    if segment.is_empty() || segment.len() > MAX_SEGMENT_LEN {
        return false;
    }
    let charset_ok = segment
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_');
    if !charset_ok || !segment.as_bytes()[0].is_ascii_alphanumeric() {
        return false;
    }
    if KEY_LIKE_PREFIXES.iter().any(|prefix| segment.starts_with(prefix)) {
        return false;
    }
    let longest_run = segment
        .split(['-', '_'])
        .map(str::len)
        .max()
        .unwrap_or(0);
    longest_run < MAX_UNSEPARATED_RUN
}

fn validate_secret(id: &CredentialId, secret: &str) -> Result<(), CredentialError> {
    let invalid = || CredentialError::new(ErrorCode::InvalidSecret, Some(id.capability));
    if secret.trim().is_empty() || secret.len() > MAX_SECRET_BYTES {
        return Err(invalid());
    }
    if secret.chars().any(char::is_control) {
        return Err(invalid());
    }
    // An id derived from the key itself would leak key material into vault target names.
    let overlaps = [&id.config_id, &id.slot]
        .iter()
        .any(|segment| segment.len() >= MIN_SEGMENT_OVERLAP_CHECK && secret.contains(*segment));
    if overlaps {
        return Err(CredentialError::new(ErrorCode::InvalidCredentialId, Some(id.capability)));
    }
    Ok(())
}

/// The only thing the webview learns about a credential.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CredentialStatus {
    pub configured: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidCapability,
    InvalidCredentialId,
    InvalidSecret,
    AlreadyConfigured,
    NotConfigured,
    VaultUnavailable,
    VaultFailure,
    Unsupported,
}

impl ErrorCode {
    fn message(self) -> &'static str {
        match self {
            Self::InvalidCapability => "Unknown credential capability.",
            Self::InvalidCredentialId => "Credential identifier is not valid.",
            Self::InvalidSecret => "Credential value is empty, too long, or malformed.",
            Self::AlreadyConfigured => "A credential is already configured; replace it instead.",
            Self::NotConfigured => "No credential is configured.",
            Self::VaultUnavailable => "The system credential vault is locked or inaccessible.",
            Self::VaultFailure => "The system credential vault reported an error.",
            Self::Unsupported => "Secure credential storage is not supported on this platform.",
        }
    }
}

/// Webview-facing error. Deliberately holds no dynamic text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CredentialError {
    pub code: ErrorCode,
    pub message: &'static str,
    pub capability: Option<Capability>,
}

impl CredentialError {
    fn new(code: ErrorCode, capability: Option<Capability>) -> Self {
        Self {
            code,
            message: code.message(),
            capability,
        }
    }
}

impl std::fmt::Display for CredentialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message)
    }
}

impl std::error::Error for CredentialError {}

/// Backend failure, already stripped of platform detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VaultError {
    Unavailable,
    Failure,
    // Only produced by `UnsupportedVault`, which production code selects off Windows/macOS.
    #[cfg_attr(any(target_os = "windows", target_os = "macos"), allow(dead_code))]
    Unsupported,
}

impl VaultError {
    fn into_credential_error(self, capability: Capability) -> CredentialError {
        let code = match self {
            Self::Unavailable => ErrorCode::VaultUnavailable,
            Self::Failure => ErrorCode::VaultFailure,
            Self::Unsupported => ErrorCode::Unsupported,
        };
        CredentialError::new(code, Some(capability))
    }
}

/// Raw key/value vault. `account` is the full credential id and must stay inside Rust.
pub trait SecretVault: Send + Sync {
    fn write(&self, account: &str, secret: &str) -> Result<(), VaultError>;
    fn read(&self, account: &str) -> Result<Option<Zeroizing<String>>, VaultError>;
    /// Returns whether an entry existed.
    fn remove(&self, account: &str) -> Result<bool, VaultError>;

    fn contains(&self, account: &str) -> Result<bool, VaultError> {
        Ok(self.read(account)?.is_some())
    }
}

/// Windows Credential Manager / macOS Keychain via `keyring`. The backend is chosen by
/// keyring at compile time; the webview API is identical on both.
#[cfg_attr(not(any(target_os = "windows", target_os = "macos")), allow(dead_code))]
struct OsVault;

#[cfg_attr(not(any(target_os = "windows", target_os = "macos")), allow(dead_code))]
impl OsVault {
    fn entry(account: &str) -> Result<keyring::Entry, VaultError> {
        keyring::Entry::new(SERVICE_NAME, account).map_err(redact_keyring_error)
    }
}

impl SecretVault for OsVault {
    fn write(&self, account: &str, secret: &str) -> Result<(), VaultError> {
        Self::entry(account)?
            .set_password(secret)
            .map_err(redact_keyring_error)
    }

    fn read(&self, account: &str) -> Result<Option<Zeroizing<String>>, VaultError> {
        match Self::entry(account)?.get_password() {
            Ok(secret) => Ok(Some(Zeroizing::new(secret))),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(err) => Err(redact_keyring_error(err)),
        }
    }

    fn remove(&self, account: &str) -> Result<bool, VaultError> {
        match Self::entry(account)?.delete_credential() {
            Ok(()) => Ok(true),
            Err(keyring::Error::NoEntry) => Ok(false),
            Err(err) => Err(redact_keyring_error(err)),
        }
    }
}

/// Drops every piece of dynamic detail from a keyring error. `BadEncoding` carries the raw
/// stored bytes, so those are wiped before the error is discarded.
#[cfg_attr(not(any(target_os = "windows", target_os = "macos")), allow(dead_code))]
fn redact_keyring_error(err: keyring::Error) -> VaultError {
    match err {
        keyring::Error::NoStorageAccess(_) => VaultError::Unavailable,
        keyring::Error::BadEncoding(mut bytes) => {
            bytes.zeroize();
            VaultError::Failure
        }
        _ => VaultError::Failure,
    }
}

/// Used where no OS vault backend is compiled in. keyring would otherwise fall back to an
/// in-memory mock and silently lose secrets on restart.
#[cfg_attr(any(target_os = "windows", target_os = "macos"), allow(dead_code))]
struct UnsupportedVault;

impl SecretVault for UnsupportedVault {
    fn write(&self, _account: &str, _secret: &str) -> Result<(), VaultError> {
        Err(VaultError::Unsupported)
    }

    fn read(&self, _account: &str) -> Result<Option<Zeroizing<String>>, VaultError> {
        Err(VaultError::Unsupported)
    }

    fn remove(&self, _account: &str) -> Result<bool, VaultError> {
        Err(VaultError::Unsupported)
    }
}

pub struct CredentialStore {
    vault: Box<dyn SecretVault>,
    // Serialises check-then-write so save/replace outcomes are deterministic when the
    // webview fires overlapping commands.
    write_lock: Mutex<()>,
}

impl CredentialStore {
    pub fn new(vault: Box<dyn SecretVault>) -> Self {
        Self {
            vault,
            write_lock: Mutex::new(()),
        }
    }

    pub fn production() -> Self {
        #[cfg(any(target_os = "windows", target_os = "macos"))]
        let vault: Box<dyn SecretVault> = Box::new(OsVault);
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        let vault: Box<dyn SecretVault> = Box::new(UnsupportedVault);
        Self::new(vault)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ()> {
        // The guarded data is `()`, so a poisoned lock carries no broken invariant.
        self.write_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Stores a new credential. Fails with `AlreadyConfigured` if one exists.
    pub fn save(&self, id: &CredentialId, secret: &str) -> Result<CredentialStatus, CredentialError> {
        validate_secret(id, secret)?;
        let account = id.account();
        let map_err = |e: VaultError| e.into_credential_error(id.capability);
        let _guard = self.lock();
        if self.vault.contains(&account).map_err(map_err)? {
            return Err(CredentialError::new(ErrorCode::AlreadyConfigured, Some(id.capability)));
        }
        self.vault.write(&account, secret).map_err(map_err)?;
        Ok(CredentialStatus { configured: true })
    }

    /// Overwrites an existing credential. Fails with `NotConfigured` if none exists.
    pub fn replace(
        &self,
        id: &CredentialId,
        secret: &str,
    ) -> Result<CredentialStatus, CredentialError> {
        validate_secret(id, secret)?;
        let account = id.account();
        let map_err = |e: VaultError| e.into_credential_error(id.capability);
        let _guard = self.lock();
        if !self.vault.contains(&account).map_err(map_err)? {
            return Err(CredentialError::new(ErrorCode::NotConfigured, Some(id.capability)));
        }
        self.vault.write(&account, secret).map_err(map_err)?;
        Ok(CredentialStatus { configured: true })
    }

    /// Idempotent: deleting a missing credential succeeds and reports not configured.
    pub fn delete(&self, id: &CredentialId) -> Result<CredentialStatus, CredentialError> {
        let _guard = self.lock();
        self.vault
            .remove(&id.account())
            .map_err(|e| e.into_credential_error(id.capability))?;
        Ok(CredentialStatus { configured: false })
    }

    pub fn status(&self, id: &CredentialId) -> Result<CredentialStatus, CredentialError> {
        let configured = self
            .vault
            .contains(&id.account())
            .map_err(|e| e.into_credential_error(id.capability))?;
        Ok(CredentialStatus { configured })
    }

    /// Rust-only raw read for the future privileged control bridge. Never wire this to a
    /// Tauri command.
    #[allow(dead_code)]
    pub(crate) fn read_secret(&self, id: &CredentialId) -> Result<Zeroizing<String>, CredentialError> {
        self.vault
            .read(&id.account())
            .map_err(|e| e.into_credential_error(id.capability))?
            .ok_or_else(|| CredentialError::new(ErrorCode::NotConfigured, Some(id.capability)))
    }
}

pub type SharedCredentialStore = Arc<CredentialStore>;

/// Vault calls can block (Keychain may show a system prompt), so they run off the async
/// runtime's worker threads.
async fn run_blocking<F>(capability: &str, job: F) -> Result<CredentialStatus, CredentialError>
where
    F: FnOnce() -> Result<CredentialStatus, CredentialError> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(job).await.unwrap_or_else(|_| {
        Err(CredentialError::new(ErrorCode::VaultFailure, Capability::parse(capability)))
    })
}

// Webview commands. Every one returns `CredentialStatus` at most; none returns secret text.
// JS passes `configId` for `config_id` (Tauri 1 camelCases command arguments).

#[tauri::command]
pub async fn credential_save(
    store: tauri::State<'_, SharedCredentialStore>,
    capability: String,
    config_id: String,
    slot: String,
    secret: String,
) -> Result<CredentialStatus, CredentialError> {
    let secret = Zeroizing::new(secret);
    let id = CredentialId::parse(&capability, &config_id, &slot)?;
    let store = Arc::clone(&store);
    run_blocking(&capability, move || store.save(&id, &secret)).await
}

#[tauri::command]
pub async fn credential_replace(
    store: tauri::State<'_, SharedCredentialStore>,
    capability: String,
    config_id: String,
    slot: String,
    secret: String,
) -> Result<CredentialStatus, CredentialError> {
    let secret = Zeroizing::new(secret);
    let id = CredentialId::parse(&capability, &config_id, &slot)?;
    let store = Arc::clone(&store);
    run_blocking(&capability, move || store.replace(&id, &secret)).await
}

#[tauri::command]
pub async fn credential_delete(
    store: tauri::State<'_, SharedCredentialStore>,
    capability: String,
    config_id: String,
    slot: String,
) -> Result<CredentialStatus, CredentialError> {
    let id = CredentialId::parse(&capability, &config_id, &slot)?;
    let store = Arc::clone(&store);
    run_blocking(&capability, move || store.delete(&id)).await
}

#[tauri::command]
pub async fn credential_status(
    store: tauri::State<'_, SharedCredentialStore>,
    capability: String,
    config_id: String,
    slot: String,
) -> Result<CredentialStatus, CredentialError> {
    let id = CredentialId::parse(&capability, &config_id, &slot)?;
    let store = Arc::clone(&store);
    run_blocking(&capability, move || store.status(&id)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// In-memory fake. Tests never construct `OsVault`, so the developer's real vault is
    /// never read or modified.
    #[derive(Default)]
    struct MemoryVault {
        entries: Mutex<HashMap<String, Zeroizing<String>>>,
    }

    impl SecretVault for MemoryVault {
        fn write(&self, account: &str, secret: &str) -> Result<(), VaultError> {
            let mut entries = self.entries.lock().unwrap();
            entries.insert(account.to_owned(), Zeroizing::new(secret.to_owned()));
            Ok(())
        }

        fn read(&self, account: &str) -> Result<Option<Zeroizing<String>>, VaultError> {
            Ok(self.entries.lock().unwrap().get(account).cloned())
        }

        fn remove(&self, account: &str) -> Result<bool, VaultError> {
            Ok(self.entries.lock().unwrap().remove(account).is_some())
        }
    }

    struct FailingVault(VaultError);

    impl SecretVault for FailingVault {
        fn write(&self, _account: &str, _secret: &str) -> Result<(), VaultError> {
            Err(self.0)
        }

        fn read(&self, _account: &str) -> Result<Option<Zeroizing<String>>, VaultError> {
            Err(self.0)
        }

        fn remove(&self, _account: &str) -> Result<bool, VaultError> {
            Err(self.0)
        }
    }

    const SECRET: &str = "test-secret-value-not-real-0001";

    fn memory_store() -> CredentialStore {
        CredentialStore::new(Box::new(MemoryVault::default()))
    }

    fn id() -> CredentialId {
        CredentialId::parse("translation", "openai-compatible-main", "api_key").unwrap()
    }

    fn assert_redacted(err: &CredentialError) {
        let json = serde_json::to_string(err).unwrap();
        let rendered = format!("{json} {err} {err:?}");
        for forbidden in [SECRET, "openai-compatible-main", "translation:openai-compatible-main"] {
            assert!(!rendered.contains(forbidden), "error leaked {forbidden:?}: {rendered}");
        }
    }

    #[test]
    fn save_then_status_reports_configured() {
        let store = memory_store();
        assert_eq!(store.status(&id()).unwrap(), CredentialStatus { configured: false });
        assert_eq!(store.save(&id(), SECRET).unwrap(), CredentialStatus { configured: true });
        assert_eq!(store.status(&id()).unwrap(), CredentialStatus { configured: true });
    }

    #[test]
    fn save_twice_is_rejected_and_keeps_original() {
        let store = memory_store();
        store.save(&id(), SECRET).unwrap();
        let err = store.save(&id(), "another-test-value").unwrap_err();
        assert_eq!(err.code, ErrorCode::AlreadyConfigured);
        assert_redacted(&err);
        assert_eq!(store.read_secret(&id()).unwrap().as_str(), SECRET);
    }

    #[test]
    fn replace_requires_existing_and_overwrites() {
        let store = memory_store();
        let err = store.replace(&id(), SECRET).unwrap_err();
        assert_eq!(err.code, ErrorCode::NotConfigured);
        assert_eq!(store.status(&id()).unwrap().configured, false);

        store.save(&id(), SECRET).unwrap();
        store.replace(&id(), "replacement-test-value").unwrap();
        assert_eq!(store.read_secret(&id()).unwrap().as_str(), "replacement-test-value");
    }

    #[test]
    fn delete_is_idempotent() {
        let store = memory_store();
        assert_eq!(store.delete(&id()).unwrap(), CredentialStatus { configured: false });
        store.save(&id(), SECRET).unwrap();
        assert_eq!(store.delete(&id()).unwrap(), CredentialStatus { configured: false });
        assert_eq!(store.status(&id()).unwrap().configured, false);
        assert_eq!(store.read_secret(&id()).unwrap_err().code, ErrorCode::NotConfigured);
    }

    #[test]
    fn ids_are_isolated_by_capability_config_and_slot() {
        let store = memory_store();
        store.save(&id(), SECRET).unwrap();
        for other in [
            CredentialId::parse("assistant", "openai-compatible-main", "api_key").unwrap(),
            CredentialId::parse("translation", "openai-compatible-alt", "api_key").unwrap(),
            CredentialId::parse("translation", "openai-compatible-main", "org_id").unwrap(),
        ] {
            assert_eq!(store.status(&other).unwrap().configured, false);
        }
    }

    #[test]
    fn account_is_stable() {
        assert_eq!(id().account(), "translation:openai-compatible-main:api_key");
    }

    #[test]
    fn rejects_invalid_ids_without_echoing_input() {
        let key_like = [
            ("stt", "sk-proj-abc", "api_key"),
            ("stt", "aizasyexample", "api_key"),
            ("stt", "abcdefghijklmnopqrstuvwxyz0123", "api_key"),
            ("stt", "Mixed-Case", "api_key"),
            ("stt", "", "api_key"),
            ("stt", "has space", "api_key"),
            ("stt", "-leading", "api_key"),
            ("stt", "ok", "slot:with:colon"),
        ];
        for (cap, config, slot) in key_like {
            let err = CredentialId::parse(cap, config, slot).unwrap_err();
            assert_eq!(err.code, ErrorCode::InvalidCredentialId, "{config}/{slot}");
            let json = serde_json::to_string(&err).unwrap();
            if !config.is_empty() {
                assert!(!json.contains(config));
            }
        }
        let err = CredentialId::parse("llm-secret", "main", "api_key").unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidCapability);
        assert!(!serde_json::to_string(&err).unwrap().contains("llm-secret"));
    }

    #[test]
    fn rejects_id_derived_from_secret() {
        let store = memory_store();
        let id = CredentialId::parse("assistant", "value-not-real", "api_key").unwrap();
        let err = store.save(&id, SECRET).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidCredentialId);
        assert!(!serde_json::to_string(&err).unwrap().contains(SECRET));
        assert_eq!(store.status(&id).unwrap().configured, false);
    }

    #[test]
    fn rejects_malformed_secrets() {
        let store = memory_store();
        let too_long = "a".repeat(MAX_SECRET_BYTES + 1);
        for bad in ["", "   ", "line\nbreak", "nul\0byte", too_long.as_str()] {
            let err = store.save(&id(), bad).unwrap_err();
            assert_eq!(err.code, ErrorCode::InvalidSecret);
        }
        assert_eq!(store.status(&id()).unwrap().configured, false);
    }

    #[test]
    fn vault_failures_are_redacted() {
        for (vault_err, code) in [
            (VaultError::Unavailable, ErrorCode::VaultUnavailable),
            (VaultError::Failure, ErrorCode::VaultFailure),
            (VaultError::Unsupported, ErrorCode::Unsupported),
        ] {
            let store = CredentialStore::new(Box::new(FailingVault(vault_err)));
            for err in [
                store.save(&id(), SECRET).unwrap_err(),
                store.replace(&id(), SECRET).unwrap_err(),
                store.delete(&id()).unwrap_err(),
                store.status(&id()).unwrap_err(),
                store.read_secret(&id()).unwrap_err(),
            ] {
                assert_eq!(err.code, code);
                assert_eq!(err.capability, Some(Capability::Translation));
                assert_redacted(&err);
            }
        }
    }

    #[test]
    fn keyring_errors_lose_platform_detail() {
        let detail = format!("CredWrite failed for translation:openai-compatible-main {SECRET}");
        let access = keyring::Error::NoStorageAccess(detail.clone().into());
        assert_eq!(redact_keyring_error(access), VaultError::Unavailable);
        let platform = keyring::Error::PlatformFailure(detail.into());
        assert_eq!(redact_keyring_error(platform), VaultError::Failure);
        let encoding = keyring::Error::BadEncoding(SECRET.as_bytes().to_vec());
        assert_eq!(redact_keyring_error(encoding), VaultError::Failure);
        assert_eq!(redact_keyring_error(keyring::Error::NoEntry), VaultError::Failure);
    }

    #[test]
    fn unsupported_vault_never_pretends_to_store() {
        let store = CredentialStore::new(Box::new(UnsupportedVault));
        assert_eq!(store.save(&id(), SECRET).unwrap_err().code, ErrorCode::Unsupported);
        assert_eq!(store.status(&id()).unwrap_err().code, ErrorCode::Unsupported);
    }

    #[test]
    fn webview_payloads_have_no_secret_field() {
        let status = serde_json::to_value(CredentialStatus { configured: true }).unwrap();
        assert_eq!(status, serde_json::json!({ "configured": true }));
        let err = serde_json::to_value(CredentialError::new(ErrorCode::NotConfigured, None)).unwrap();
        let keys: Vec<&str> = err.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys.len(), 3);
        for key in ["code", "message", "capability"] {
            assert!(keys.contains(&key));
        }
    }

    #[test]
    fn credential_id_debug_hides_config_and_slot() {
        let rendered = format!("{:?}", id());
        assert!(!rendered.contains("openai-compatible-main"));
        assert!(!rendered.contains("api_key"));
    }
}

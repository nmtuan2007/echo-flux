//! Privileged bridge between the webview, the credential vault, and the engine.
//!
//! Both paths are Rust-owned, so neither the control token nor a secret value ever reaches
//! the webview. React sends public config plus credential references; Rust resolves the
//! references against the vault and returns only safe, fixed-shape results.
//!
//! Test Connection (`provider_test_connection`):
//!
//! ```text
//! React --{capability, service: public config + credentialRefs}--> Rust
//! Rust: refs -> CredentialStore::read_secret            (blocking pool)
//! Rust: spawn `python -m engine.providers.validation`   (stdin/stdout piped, stderr null)
//! Rust --one JSON request {id, capability, service, secrets}, then stdin EOF--> child
//! child --one JSON line {id, capability, providerId, status, ..., errorCode}--> Rust
//! Rust: check id, capability, provider, and field shapes --> safe result --> React
//! ```
//!
//! Validation runs in a short-lived child rather than as an engine socket command, so the
//! documented WebSocket contract is untouched and a paid provider session can never outlive
//! the call: on timeout, a malformed reply, or a vanished child, the child is killed and
//! reaped. Secrets travel only over the child's stdin, never argv or environment.
//!
//! Start/Stop forwarding (`engine_forward_start`, `engine_forward_stop`):
//!
//! ```text
//! Rust --{"op":"auth","role":"control","token":<control token>,"protocol":1}--> engine
//! Rust --{"op":"request","id":<id>,"command":{"type":"start"|"stop",...}}-->     engine
//! engine --{"op":"response","id":<id>,"message":{...}}* then {"op":"complete"}--> Rust
//! ```
//!
//! The control token and port are captured from the engine's launch environment by
//! [`CapturingSpawner`] and used only while `EngineManager::status` still reports that run.
//! Nothing here is called by React yet; the settings and session tasks wire it up.
//!
//! Allowlisted command forwarding (`engine_forward_command`):
//!
//! ```text
//! React --{command: {type, ...}, credentials?: {hf_token: ref}}--> Rust
//! Rust: allowlist, size, and recursive secret-key checks; hf_token ref -> vault (blocking pool)
//! Rust --auth control, then {"op":"request","id":<id>,"command":{..., "hf_token"?}}--> engine
//! engine --response* / complete / late worker-thread responses--> Rust
//! ```
//!
//! An authenticated event socket cannot send commands, so device, model, Hub, and Assist
//! commands take this path. Worker-thread replies (Hub search, suggestions, summary chunks)
//! arrive after `complete`, so collection continues until the command's terminal message
//! type or an application `error` message, bounded by a per-command timeout
//! ([`forward_policy`]). Every exit closes the socket.
//!
//! Packaging risk: only debug builds can locate the validator (the repository `.venv`
//! interpreter). Release builds have no validator entrypoint, just as the frozen engine
//! sidecar from EF-005 is still unresolved, so Test Connection fails closed there with
//! `validator_unavailable`.
//!
//! Zeroization is best effort: request buffers, resolved secrets, and the captured token
//! are `Zeroizing`, but serde_json values and the WebSocket library's send buffer hold
//! transient copies that are freed without wiping.

use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::Message;
use zeroize::{Zeroize, Zeroizing};

use crate::engine_process::{
    self, EngineChild, EngineSpawner, EngineState, EngineStatus, LaunchSpec, LaunchTarget,
    SharedEngineManager, ENV_CONTROL_TOKEN, ENV_EVENT_TOKEN, ENV_PORT,
};
use crate::secure_store::{
    self, CredentialError, CredentialId, CredentialStore, SharedCredentialStore,
};

/// Python module run as the Test Connection child.
pub const VALIDATOR_MODULE: &str = "engine.providers.validation";

/// Longer than the validator's own 15 s probe timeout plus interpreter start-up.
const VALIDATOR_TIMEOUT: Duration = Duration::from_secs(30);
const VALIDATOR_EXIT_GRACE: Duration = Duration::from_millis(500);
const VALIDATOR_POLL: Duration = Duration::from_millis(20);
const CONTROL_TIMEOUT: Duration = Duration::from_secs(20);

const MAX_RESULT_BYTES: usize = 8 * 1024;
const MAX_CONTROL_MESSAGE_BYTES: usize = 1024 * 1024;
const MAX_CONTROL_RESPONSES: usize = 64;
/// Pre-sized so serialising a request never reallocates and strands an unwiped copy.
const REQUEST_BUFFER_BYTES: usize = 64 * 1024;
const MAX_CONFIG_KEY_LEN: usize = 96;
const MAX_PROVIDER_ID_LEN: usize = 48;

const REDACTED: &str = "[redacted]";
const CAPABILITIES: [&str; 3] = ["stt", "translation", "assistant"];

/// Mirrors `_SECRET_KEY_NAMES` in engine/core/config.py (lowercased, `_`/`-` removed).
const SECRET_KEY_NAMES: &[&str] = &[
    "apikey",
    "secret",
    "secretkey",
    "clientsecret",
    "token",
    "accesstoken",
    "authtoken",
    "password",
    "authorization",
    "bearer",
    "value",
];

// ---------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidRequest,
    CredentialNotConfigured,
    CredentialUnavailable,
    ValidatorUnavailable,
    ValidatorFailed,
    Timeout,
    InvalidResponse,
    CorrelationMismatch,
    EngineNotRunning,
    ControlUnavailable,
    AuthFailed,
    CommandRejected,
    CommandNotAllowed,
    InvalidCommand,
    SecretInCommand,
    VaultError,
}

impl ErrorCode {
    fn message(self) -> &'static str {
        match self {
            Self::InvalidRequest => "The request is malformed or contains a raw secret.",
            Self::CredentialNotConfigured => "A referenced credential is not configured.",
            Self::CredentialUnavailable => "The system credential vault could not be read.",
            Self::ValidatorUnavailable => "Provider validation is not available in this build.",
            Self::ValidatorFailed => "Provider validation did not complete.",
            Self::Timeout => "The operation timed out.",
            Self::InvalidResponse => "The engine returned an unexpected response.",
            Self::CorrelationMismatch => "The response did not match the request.",
            Self::EngineNotRunning => "The engine is not running.",
            Self::ControlUnavailable => "The engine control channel is unavailable.",
            Self::AuthFailed => "The engine rejected the control credentials.",
            Self::CommandRejected => "The engine rejected the command.",
            Self::CommandNotAllowed => "The command cannot be sent through the control bridge.",
            Self::InvalidCommand => "The command is malformed.",
            Self::SecretInCommand => "The command must not contain secret values.",
            Self::VaultError => "The referenced credential could not be read.",
        }
    }
}

/// Webview-facing error. Deliberately holds no dynamic text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BridgeError {
    pub code: ErrorCode,
    pub message: &'static str,
}

impl BridgeError {
    fn new(code: ErrorCode) -> Self {
        Self {
            code,
            message: code.message(),
        }
    }
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message)
    }
}

impl std::error::Error for BridgeError {}

impl From<CredentialError> for BridgeError {
    fn from(err: CredentialError) -> Self {
        let code = match err.code {
            secure_store::ErrorCode::NotConfigured => ErrorCode::CredentialNotConfigured,
            secure_store::ErrorCode::InvalidCapability
            | secure_store::ErrorCode::InvalidCredentialId => ErrorCode::InvalidRequest,
            _ => ErrorCode::CredentialUnavailable,
        };
        Self::new(code)
    }
}

fn invalid_request() -> BridgeError {
    BridgeError::new(ErrorCode::InvalidRequest)
}

fn invalid_response() -> BridgeError {
    BridgeError::new(ErrorCode::InvalidResponse)
}

// ---------------------------------------------------------------------------------------
// Shape checks
// ---------------------------------------------------------------------------------------

fn is_secret_key_name(key: &str) -> bool {
    let normalized: String = key
        .chars()
        .filter(|c| *c != '_' && *c != '-')
        .flat_map(char::to_lowercase)
        .collect();
    SECRET_KEY_NAMES.contains(&normalized.as_str())
}

fn contains_secret_key(value: &Value) -> bool {
    match value {
        Value::Object(map) => map.iter().any(|(key, item)| {
            if key == "credentialRefs" {
                match item {
                    Value::Object(slots) => slots.values().any(contains_secret_key),
                    _ => contains_secret_key(item),
                }
            } else {
                is_secret_key_name(key) || contains_secret_key(item)
            }
        }),
        Value::Array(items) => items.iter().any(contains_secret_key),
        _ => false,
    }
}

fn is_provider_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_PROVIDER_ID_LEN
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
}

fn is_error_code(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value.as_bytes()[0].is_ascii_lowercase()
        && value
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

fn is_fingerprint(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Dotted legacy config key, e.g. `llm.api_key`.
fn is_config_key(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= MAX_CONFIG_KEY_LEN
        && key
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'.')
}

fn leaf(key: &str) -> &str {
    key.rsplit('.').next().unwrap_or(key)
}

// ---------------------------------------------------------------------------------------
// Validator child process
// ---------------------------------------------------------------------------------------

pub struct ValidatorPipes {
    pub stdin: Box<dyn Write + Send>,
    pub stdout: Box<dyn Read + Send>,
}

pub trait ValidatorProcess: Send {
    /// Hands over both pipes once; `None` if they are missing.
    fn take_pipes(&mut self) -> Option<ValidatorPipes>;
    fn has_exited(&mut self) -> io::Result<bool>;
    /// Terminates and reaps the process. Must be safe to call on an exited process.
    fn kill(&mut self) -> io::Result<()>;
}

pub trait ValidatorSpawner: Send + Sync {
    fn spawn(&self, target: &LaunchTarget) -> io::Result<Box<dyn ValidatorProcess>>;
}

pub struct OsValidatorSpawner;

impl ValidatorSpawner for OsValidatorSpawner {
    fn spawn(&self, target: &LaunchTarget) -> io::Result<Box<dyn ValidatorProcess>> {
        let mut command = Command::new(&target.program);
        command
            .args(&target.args)
            .current_dir(&target.working_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // The validator writes nothing to stderr; a null sink also means a misbehaving
            // child can never block on a full pipe.
            .stderr(Stdio::null())
            .env_remove(ENV_CONTROL_TOKEN)
            .env_remove(ENV_EVENT_TOKEN);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        Ok(Box::new(OsValidatorProcess(command.spawn()?)))
    }
}

struct OsValidatorProcess(Child);

impl ValidatorProcess for OsValidatorProcess {
    fn take_pipes(&mut self) -> Option<ValidatorPipes> {
        let stdin = self.0.stdin.take()?;
        let stdout = self.0.stdout.take()?;
        Some(ValidatorPipes {
            stdin: Box::new(stdin),
            stdout: Box::new(stdout),
        })
    }

    fn has_exited(&mut self) -> io::Result<bool> {
        Ok(self.0.try_wait()?.is_some())
    }

    fn kill(&mut self) -> io::Result<()> {
        if self.0.try_wait()?.is_some() {
            return Ok(());
        }
        self.0.kill()?;
        self.0.wait().map(|_| ())
    }
}

/// The validator launch for a given engine launch. Development runs the repository
/// interpreter; the packaged sidecar has no validator entrypoint yet, so it fails closed.
pub fn validator_target(engine: LaunchTarget) -> Result<LaunchTarget, BridgeError> {
    if engine.args == ["-m", "engine.main"] {
        Ok(LaunchTarget {
            program: engine.program,
            args: vec!["-m".to_owned(), VALIDATOR_MODULE.to_owned()],
            working_dir: engine.working_dir,
        })
    } else {
        Err(BridgeError::new(ErrorCode::ValidatorUnavailable))
    }
}

/// Reads up to the first newline. The line is not secret, but the cap keeps a runaway
/// child from growing memory without bound.
fn read_result_line(reader: &mut dyn Read) -> Result<Zeroizing<Vec<u8>>, BridgeError> {
    let mut line = Zeroizing::new(Vec::with_capacity(1024));
    let mut chunk = [0u8; 1024];
    loop {
        let read = match reader.read(&mut chunk) {
            Ok(read) => read,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(BridgeError::new(ErrorCode::ValidatorFailed)),
        };
        if read == 0 {
            break;
        }
        line.extend_from_slice(&chunk[..read]);
        if let Some(end) = line.iter().position(|b| *b == b'\n') {
            line.truncate(end);
            return Ok(line);
        }
        if line.len() > MAX_RESULT_BYTES {
            return Err(invalid_response());
        }
    }
    if line.is_empty() {
        Err(BridgeError::new(ErrorCode::ValidatorFailed))
    } else {
        Ok(line)
    }
}

/// A child that answered gets a short grace period to exit by itself; every other child,
/// and one that lingers, is killed and reaped.
fn reap(process: &mut dyn ValidatorProcess, answered: bool) {
    if answered {
        let deadline = Instant::now() + VALIDATOR_EXIT_GRACE;
        while Instant::now() < deadline {
            match process.has_exited() {
                Ok(true) => return,
                Ok(false) => std::thread::sleep(VALIDATOR_POLL),
                Err(_) => break,
            }
        }
    }
    let _ = process.kill();
}

/// Writes the request, closes stdin, and waits for one result line under `timeout`.
///
/// Pipe I/O runs on a helper thread so a child that never reads or never answers cannot
/// block past the deadline; killing the child closes its pipes, which unblocks the helper.
fn run_validator(
    spawner: &dyn ValidatorSpawner,
    target: &LaunchTarget,
    request: Zeroizing<Vec<u8>>,
    timeout: Duration,
) -> Result<Zeroizing<Vec<u8>>, BridgeError> {
    let mut process = spawner
        .spawn(target)
        .map_err(|_| BridgeError::new(ErrorCode::ValidatorUnavailable))?;
    let pipes = match process.take_pipes() {
        Some(pipes) => pipes,
        None => {
            reap(process.as_mut(), false);
            return Err(BridgeError::new(ErrorCode::ValidatorFailed));
        }
    };

    let (sender, receiver) = mpsc::channel();
    let io_job = move || {
        let ValidatorPipes {
            mut stdin,
            mut stdout,
        } = pipes;
        let written = stdin.write_all(&request).and_then(|()| stdin.flush());
        drop(request);
        // EOF tells the child the request is complete.
        drop(stdin);
        let outcome = match written {
            Ok(()) => read_result_line(stdout.as_mut()),
            Err(_) => Err(BridgeError::new(ErrorCode::ValidatorFailed)),
        };
        let _ = sender.send(outcome);
    };
    if std::thread::Builder::new()
        .name("provider-validator-io".to_owned())
        .spawn(io_job)
        .is_err()
    {
        reap(process.as_mut(), false);
        return Err(BridgeError::new(ErrorCode::ValidatorFailed));
    }

    let outcome = match receiver.recv_timeout(timeout) {
        Ok(outcome) => outcome,
        Err(mpsc::RecvTimeoutError::Timeout) => Err(BridgeError::new(ErrorCode::Timeout)),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            Err(BridgeError::new(ErrorCode::ValidatorFailed))
        }
    };
    reap(process.as_mut(), outcome.is_ok());
    outcome
}

// ---------------------------------------------------------------------------------------
// Validation request and result
// ---------------------------------------------------------------------------------------

/// What React sends: public service config (`services.<capability>` shape) including
/// `credentialRefs`. Any raw secret-named field is rejected before the vault is touched.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TestConnectionRequest {
    pub capability: String,
    pub service: Map<String, Value>,
}

impl fmt::Debug for TestConnectionRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TestConnectionRequest")
            .field("capability", &self.capability)
            .finish_non_exhaustive()
    }
}

#[derive(Serialize)]
struct WireValidationRequest<'a> {
    id: &'a str,
    capability: &'a str,
    service: &'a Map<String, Value>,
    secrets: BTreeMap<&'a str, &'a str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ValidationStatus {
    Valid,
    Invalid,
    Unvalidated,
}

/// Exactly the fields the validator may report. Unknown fields fail parsing, so a future
/// change that adds free text on the Python side cannot silently reach the webview.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WireValidationResult {
    id: Option<String>,
    capability: Option<String>,
    provider_id: Option<String>,
    status: ValidationStatus,
    latency_ms: Option<f64>,
    tested_at: f64,
    fingerprint: Option<String>,
    credential_version: Option<u64>,
    error_code: Option<String>,
}

/// The only thing the webview learns about a Test Connection.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderValidationResult {
    pub capability: String,
    pub provider_id: String,
    pub status: ValidationStatus,
    pub latency_ms: Option<f64>,
    pub tested_at: f64,
    pub fingerprint: Option<String>,
    pub credential_version: Option<u64>,
    pub error_code: Option<String>,
}

fn credential_refs(
    capability: &str,
    service: &Map<String, Value>,
) -> Result<Vec<(String, CredentialId)>, BridgeError> {
    let refs = match service.get("credentialRefs") {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Object(refs)) => refs,
        Some(_) => return Err(invalid_request()),
    };
    refs.iter()
        .map(|(slot, raw)| {
            let fields = raw.as_object().ok_or_else(invalid_request)?;
            // A reference is identity plus version only; anything else could carry a value.
            if fields
                .keys()
                .any(|key| !matches!(key.as_str(), "configId" | "slot" | "version"))
            {
                return Err(invalid_request());
            }
            let config_id = fields
                .get("configId")
                .and_then(Value::as_str)
                .ok_or_else(invalid_request)?;
            let inner_slot = fields
                .get("slot")
                .and_then(Value::as_str)
                .ok_or_else(invalid_request)?;
            if inner_slot != slot {
                return Err(invalid_request());
            }
            let id = CredentialId::parse(capability, config_id, slot)?;
            Ok((slot.clone(), id))
        })
        .collect()
}

fn parse_validation_result(
    line: &[u8],
    expected_id: &str,
    capability: &str,
    provider_id: &str,
) -> Result<ProviderValidationResult, BridgeError> {
    let wire: WireValidationResult =
        serde_json::from_slice(line).map_err(|_| invalid_response())?;
    match wire.id.as_deref() {
        // The validator only drops the id when it could not read the request at all.
        None => return Err(BridgeError::new(ErrorCode::ValidatorFailed)),
        Some(id) if id != expected_id => {
            return Err(BridgeError::new(ErrorCode::CorrelationMismatch))
        }
        Some(_) => {}
    }
    // Python withholds identifiers it considers unsafe; that means the config was rejected.
    let (Some(wire_capability), Some(wire_provider)) =
        (wire.capability.as_deref(), wire.provider_id.as_deref())
    else {
        return Err(invalid_request());
    };
    if wire_capability != capability || wire_provider != provider_id {
        return Err(invalid_response());
    }
    if wire.error_code.as_deref().is_some_and(|code| !is_error_code(code))
        || wire.fingerprint.as_deref().is_some_and(|fp| !is_fingerprint(fp))
        || !wire.tested_at.is_finite()
        || wire.tested_at <= 0.0
        || wire.latency_ms.is_some_and(|ms| !ms.is_finite() || ms < 0.0)
    {
        return Err(invalid_response());
    }
    let consistent = match wire.status {
        ValidationStatus::Valid => wire.error_code.is_none() && wire.fingerprint.is_some(),
        ValidationStatus::Invalid | ValidationStatus::Unvalidated => wire.error_code.is_some(),
    };
    if !consistent {
        return Err(invalid_response());
    }
    Ok(ProviderValidationResult {
        capability: capability.to_owned(),
        provider_id: provider_id.to_owned(),
        status: wire.status,
        latency_ms: wire.latency_ms,
        tested_at: wire.tested_at,
        fingerprint: wire.fingerprint,
        credential_version: wire.credential_version,
        error_code: wire.error_code,
    })
}

// ---------------------------------------------------------------------------------------
// Control-role client
// ---------------------------------------------------------------------------------------

/// This run's loopback port and control token.
pub struct ControlEndpoint {
    port: u16,
    token: Zeroizing<String>,
}

impl fmt::Debug for ControlEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ControlEndpoint")
            .field("port", &self.port)
            .finish_non_exhaustive()
    }
}

type EndpointSlot = Arc<Mutex<Option<ControlEndpoint>>>;

fn lock_slot(slot: &Mutex<Option<ControlEndpoint>>) -> MutexGuard<'_, Option<ControlEndpoint>> {
    // The slot is replaced wholesale, so a poisoned lock carries no broken invariant.
    slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Wraps the real engine spawner and keeps a zeroized copy of each run's port and control
/// token, taken from the same launch environment the engine receives. `EngineManager`
/// serialises starts, so the slot always describes the most recent spawn attempt.
pub struct CapturingSpawner {
    inner: Box<dyn EngineSpawner>,
    slot: EndpointSlot,
}

impl EngineSpawner for CapturingSpawner {
    fn spawn(&self, spec: &LaunchSpec) -> io::Result<Box<dyn EngineChild>> {
        // A new attempt always invalidates the previous run's token, even if it fails.
        *lock_slot(&self.slot) = None;
        let child = self.inner.spawn(spec)?;
        *lock_slot(&self.slot) = capture_endpoint(spec);
        Ok(child)
    }
}

fn capture_endpoint(spec: &LaunchSpec) -> Option<ControlEndpoint> {
    let value = |key: &str| spec.env.iter().find(|(k, _)| *k == key).map(|(_, v)| v);
    let port = value(ENV_PORT)?.parse::<u16>().ok()?;
    let token = value(ENV_CONTROL_TOKEN)?;
    Some(ControlEndpoint {
        port,
        token: Zeroizing::new(token.as_str().to_owned()),
    })
}

/// One serialised control request plus every value that must never come back out.
pub struct PreparedRequest {
    request_id: String,
    frame: Zeroizing<String>,
    redactions: Vec<Zeroizing<String>>,
}

// Manual Debug: the frame and redactions hold secret values.
impl fmt::Debug for PreparedRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedRequest")
            .field("request_id", &self.request_id)
            .finish_non_exhaustive()
    }
}

/// Legacy replies to a forwarded command, correlated by request id and redacted.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ControlForwardResult {
    pub messages: Vec<Value>,
}

/// Start config from React: dotted public keys plus references for the secret keys.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ForwardStartRequest {
    pub config: Map<String, Value>,
    #[serde(default)]
    pub credentials: BTreeMap<String, CredentialRefInput>,
}

impl fmt::Debug for ForwardStartRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForwardStartRequest")
            .field("config_keys", &self.config.len())
            .finish_non_exhaustive()
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CredentialRefInput {
    pub capability: String,
    pub config_id: String,
    pub slot: String,
}

#[derive(Serialize)]
struct AuthFrame<'a> {
    op: &'static str,
    role: &'static str,
    token: &'a str,
    protocol: u32,
}

#[derive(Serialize)]
struct RequestFrame<'a> {
    op: &'static str,
    id: &'a str,
    command: CommandBody<'a>,
}

#[derive(Serialize)]
struct CommandBody<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    config: Option<&'a Map<String, Value>>,
}

fn request_frame(
    request_id: &str,
    kind: &'static str,
    config: Option<&Map<String, Value>>,
) -> Result<Zeroizing<String>, BridgeError> {
    let frame = RequestFrame {
        op: "request",
        id: request_id,
        command: CommandBody { kind, config },
    };
    let mut buffer = Zeroizing::new(Vec::with_capacity(REQUEST_BUFFER_BYTES));
    serde_json::to_writer(&mut *buffer, &frame).map_err(|_| invalid_request())?;
    let text = String::from_utf8(std::mem::take(&mut *buffer)).map_err(|_| invalid_request())?;
    Ok(Zeroizing::new(text))
}

fn inject_credentials(
    store: &CredentialStore,
    credentials: &BTreeMap<String, CredentialRefInput>,
    config: &mut Map<String, Value>,
    injected: &mut Vec<String>,
    redactions: &mut Vec<Zeroizing<String>>,
) -> Result<(), BridgeError> {
    for (key, reference) in credentials {
        // References may only fill secret-named keys; everything else is public config.
        if !is_config_key(key) || !is_secret_key_name(leaf(key)) {
            return Err(invalid_request());
        }
        let id = CredentialId::parse(&reference.capability, &reference.config_id, &reference.slot)?;
        let secret = store.read_secret(&id)?;
        config.insert(key.clone(), Value::String(secret.as_str().to_owned()));
        injected.push(key.clone());
        redactions.push(secret);
    }
    Ok(())
}

fn scrub(config: &mut Map<String, Value>, injected: &[String]) {
    for key in injected {
        if let Some(Value::String(mut value)) = config.remove(key) {
            value.zeroize();
        }
    }
}

fn redact_value(value: &mut Value, needles: &[Zeroizing<String>]) {
    match value {
        Value::String(text) => {
            for needle in needles.iter().filter(|needle| !needle.is_empty()) {
                if text.contains(needle.as_str()) {
                    *text = text.replace(needle.as_str(), REDACTED);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(|item| redact_value(item, needles)),
        Value::Object(map) => map.values_mut().for_each(|item| redact_value(item, needles)),
        _ => {}
    }
}

// ---------------------------------------------------------------------------------------
// Allowlisted command forwarding
// ---------------------------------------------------------------------------------------

/// Summaries carry the transcript, so this sits well above a start config but below the
/// engine's 1 MiB frame limit.
const MAX_FORWARD_COMMAND_BYTES: usize = 512 * 1024;
/// A summary streams one `llm_chunk` per model delta; the cap only stops a runaway engine.
const MAX_FORWARD_RESPONSES: usize = 8192;
const FORWARD_FRAME_OVERHEAD_BYTES: usize = 256;
const HF_TOKEN_KEY: &str = "hf_token";

/// Substrings of secret-looking keys (lowercased, `_`/`-` removed). Broader than
/// `SECRET_KEY_NAMES` because no legitimate legacy command field contains any of them.
const FORWARD_SECRET_KEY_PARTS: &[&str] = &[
    "apikey",
    "token",
    "secret",
    "password",
    "authorization",
    "credential",
    "bearer",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardPolicy {
    /// Legacy message type that ends collection; `None` returns on `complete`.
    pub terminal: Option<&'static str>,
    pub timeout: Duration,
    /// Only the Hugging Face commands may receive a vault-held `hf_token`.
    pub accepts_hf_token: bool,
}

/// The allowlist. `start` and `stop` keep their dedicated commands and are absent on purpose.
pub fn forward_policy(kind: &str) -> Option<ForwardPolicy> {
    let (terminal, seconds, accepts_hf_token) = match kind {
        "list_devices" => (Some("devices_list"), 10, false),
        "request_models_list" => (Some("models_list"), 10, false),
        // Progress and the result are broadcast as events, so `complete` is enough.
        "download_model" => (None, CONTROL_TIMEOUT.as_secs(), true),
        "delete_model" => (Some("model_action_result"), 30, false),
        "search_hub" => (Some("hub_search_results"), 30, true),
        "request_suggestion" => (Some("suggestion_result"), 120, false),
        "request_assist" => (None, 120, false),
        "request_summary" => (Some("llm_done"), 300, false),
        "test_connection" => (Some("test_connection_result"), 30, false),

        // Phase B: Context Profiles (EF-020)
        "profile_create" => (None, 10, false),
        "profile_list" => (None, 10, false),
        "profile_get" => (None, 10, false),
        "profile_update" => (None, 10, false),
        "profile_delete" => (None, 10, false),
        "profile_attach" => (None, 10, false),

        // Phase B: Conversations (EF-020 / EF-021)
        "conversation_create" => (None, 10, false),
        "conversation_start" => (None, 10, false),
        "conversation_list" => (None, 10, false),
        "conversation_get" => (None, 15, false),
        "conversation_rename" => (None, 10, false),
        "conversation_delete" => (None, 10, false),
        "conversation_export" => (None, 30, false),
        "conversation_import" => (None, 30, false),

        // Phase B: Local Model Packs (EF-008)
        "pack_status" => (None, 10, false),
        "pack_install" => (None, 30, false),
        "pack_cancel" => (None, 10, false),
        "pack_retry" => (None, 30, false),
        "pack_check_readiness" => (None, 15, false),

        // Phase B: Provider Manifests (EF-003 / EF-022)
        "get_provider_manifests" => (None, 10, false),

        // Phase B: Summary Streaming & Cancellation (EF-022)
        "summary_cancel" => (None, 10, false),

        _ => return None,
    };
    Some(ForwardPolicy {
        terminal,
        timeout: Duration::from_secs(seconds),
        accepts_hf_token,
    })
}

/// Collapses internal failures onto the fixed codes `engine_forward_command` documents.
fn forward_error(err: BridgeError) -> BridgeError {
    let code = match err.code {
        ErrorCode::CommandNotAllowed
        | ErrorCode::InvalidCommand
        | ErrorCode::SecretInCommand
        | ErrorCode::EngineNotRunning
        | ErrorCode::Timeout
        | ErrorCode::CommandRejected
        | ErrorCode::VaultError => err.code,
        ErrorCode::InvalidRequest => ErrorCode::InvalidCommand,
        ErrorCode::CredentialNotConfigured | ErrorCode::CredentialUnavailable => {
            ErrorCode::VaultError
        }
        // No captured endpoint, a refused connection, or a dropped socket: the run is gone.
        ErrorCode::ControlUnavailable => ErrorCode::EngineNotRunning,
        ErrorCode::AuthFailed
        | ErrorCode::InvalidResponse
        | ErrorCode::CorrelationMismatch
        | ErrorCode::ValidatorUnavailable
        | ErrorCode::ValidatorFailed => ErrorCode::CommandRejected,
    };
    BridgeError::new(code)
}

fn is_forward_secret_key(key: &str) -> bool {
    let normalized: String = key
        .chars()
        .filter(|c| *c != '_' && *c != '-')
        .flat_map(char::to_lowercase)
        .collect();
    FORWARD_SECRET_KEY_PARTS
        .iter()
        .any(|part| normalized.contains(part))
}

fn contains_forward_secret_key(value: &Value) -> bool {
    match value {
        Value::Object(map) => map
            .iter()
            .any(|(key, item)| is_forward_secret_key(key) || contains_forward_secret_key(item)),
        Value::Array(items) => items.iter().any(contains_forward_secret_key),
        _ => false,
    }
}

/// Measures serialized size without buffering, failing as soon as the cap is passed.
struct BoundedCounter(usize);

impl Write for BoundedCounter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0 += buf.len();
        if self.0 > MAX_FORWARD_COMMAND_BYTES {
            return Err(io::Error::other("command too large"));
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A command that passed every check needing neither the vault nor the engine.
pub struct CheckedCommand {
    policy: ForwardPolicy,
    command: Map<String, Value>,
    size: usize,
    hf_token: Option<CredentialId>,
}

// Manual Debug: the command may carry transcript text.
impl fmt::Debug for CheckedCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CheckedCommand")
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

/// Validates a webview command: object with an allowlisted string `type`, bounded size, no
/// secret-named key at any depth, and at most an `hf_token` reference where one is accepted.
pub fn check_forward_command(
    command: Value,
    credentials: Option<Value>,
) -> Result<CheckedCommand, BridgeError> {
    let invalid = || BridgeError::new(ErrorCode::InvalidCommand);
    let Value::Object(command) = command else {
        return Err(invalid());
    };
    let policy = match command.get("type") {
        Some(Value::String(kind)) => {
            forward_policy(kind).ok_or_else(|| BridgeError::new(ErrorCode::CommandNotAllowed))?
        }
        _ => return Err(invalid()),
    };
    let mut counter = BoundedCounter(0);
    serde_json::to_writer(&mut counter, &command).map_err(|_| invalid())?;
    if command
        .iter()
        .any(|(key, value)| is_forward_secret_key(key) || contains_forward_secret_key(value))
    {
        return Err(BridgeError::new(ErrorCode::SecretInCommand));
    }
    let hf_token = match credentials {
        None | Some(Value::Null) => None,
        Some(value) => {
            let mut refs: BTreeMap<String, CredentialRefInput> =
                serde_json::from_value(value).map_err(|_| invalid())?;
            match refs.remove(HF_TOKEN_KEY) {
                None if refs.is_empty() => None,
                Some(reference) if refs.is_empty() && policy.accepts_hf_token => Some(
                    CredentialId::parse(&reference.capability, &reference.config_id, &reference.slot)
                        .map_err(|_| invalid())?,
                ),
                _ => return Err(invalid()),
            }
        }
    };
    Ok(CheckedCommand {
        policy,
        command,
        size: counter.0,
        hf_token,
    })
}

#[derive(Serialize)]
struct ForwardFrame<'a> {
    op: &'static str,
    id: &'a str,
    command: &'a Map<String, Value>,
}

/// How long to keep reading replies for one request.
#[derive(Debug, Clone, Copy)]
struct Collection {
    /// Reply type that ends the request; `None` ends it at `complete`.
    terminal: Option<&'static str>,
    max_messages: usize,
}

const START_STOP_COLLECTION: Collection = Collection {
    terminal: None,
    max_messages: MAX_CONTROL_RESPONSES,
};

type ControlSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

async fn next_frame(socket: &mut ControlSocket) -> Result<Map<String, Value>, BridgeError> {
    loop {
        match socket.next().await {
            Some(Ok(Message::Text(text))) => {
                return match serde_json::from_str::<Value>(&text) {
                    Ok(Value::Object(frame)) => Ok(frame),
                    _ => Err(invalid_response()),
                };
            }
            Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_))) => continue,
            Some(Ok(Message::Binary(_))) => return Err(invalid_response()),
            Some(Ok(Message::Close(_)) | Err(_)) | None => {
                return Err(BridgeError::new(ErrorCode::ControlUnavailable))
            }
        }
    }
}

async fn send_text(socket: &mut ControlSocket, text: String) -> Result<(), BridgeError> {
    socket
        .send(Message::Text(text))
        .await
        .map_err(|_| BridgeError::new(ErrorCode::ControlUnavailable))
}

fn frame_str<'a>(frame: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    frame.get(key).and_then(Value::as_str)
}

async fn exchange(
    socket: &mut ControlSocket,
    token: &str,
    request_id: &str,
    frame: &mut Zeroizing<String>,
    collection: Collection,
) -> Result<Vec<Value>, BridgeError> {
    let auth = AuthFrame {
        op: "auth",
        role: "control",
        token,
        protocol: 1,
    };
    let mut auth_text = Zeroizing::new(serde_json::to_string(&auth).map_err(|_| invalid_request())?);
    send_text(socket, std::mem::take(&mut *auth_text)).await?;

    let reply = next_frame(socket).await?;
    match frame_str(&reply, "op") {
        Some("auth_ok") if frame_str(&reply, "role") == Some("control") => {}
        Some("error") => return Err(BridgeError::new(ErrorCode::AuthFailed)),
        _ => return Err(invalid_response()),
    }

    send_text(socket, std::mem::take(&mut **frame)).await?;
    let mut messages = Vec::new();
    let mut completed = false;
    // Worker-thread replies go out after `complete`, so a command with a terminal type is
    // finished only once both have arrived, in either order.
    let mut terminal_seen = collection.terminal.is_none();
    loop {
        let mut reply = next_frame(socket).await?;
        let matches = frame_str(&reply, "id") == Some(request_id);
        match frame_str(&reply, "op") {
            Some("response") if matches => {
                if messages.len() >= collection.max_messages {
                    return Err(invalid_response());
                }
                let message = reply
                    .remove("message")
                    .or_else(|| reply.remove("data"))
                    .unwrap_or(Value::Null);
                if let Some(terminal) = collection.terminal {
                    // An application error is the command's answer; nothing follows it.
                    let kind = message.get("type").and_then(Value::as_str);
                    terminal_seen |= kind == Some(terminal) || kind == Some("error");
                }
                messages.push(message);
                if completed && terminal_seen {
                    return Ok(messages);
                }
            }
            Some("complete") if matches => {
                completed = true;
                if terminal_seen {
                    return Ok(messages);
                }
            }
            // An id-less error answers a frame the engine could not attribute; it is ours.
            Some("error") if matches || !reply.contains_key("id") => {
                return Err(BridgeError::new(ErrorCode::CommandRejected))
            }
            // Replies to other requests, or anything else, are not ours to interpret.
            _ => {}
        }
    }
}

async fn control_round_trip(
    url: &str,
    token: &str,
    request_id: &str,
    mut frame: Zeroizing<String>,
    collection: Collection,
) -> Result<Vec<Value>, BridgeError> {
    let config = WebSocketConfig {
        max_message_size: Some(MAX_CONTROL_MESSAGE_BYTES),
        max_frame_size: Some(MAX_CONTROL_MESSAGE_BYTES),
        ..Default::default()
    };
    let (mut socket, _) = tokio_tungstenite::connect_async_with_config(url, Some(config), true)
        .await
        .map_err(|_| BridgeError::new(ErrorCode::ControlUnavailable))?;
    let outcome = exchange(&mut socket, token, request_id, &mut frame, collection).await;
    // Best effort; dropping the stream closes the TCP connection either way.
    let _ = socket.close(None).await;
    outcome
}

/// Sends one prepared request over a fresh control-role connection and redacts the replies.
/// The connection is closed on every path; on timeout the dropped future closes the socket.
async fn forward_prepared(
    endpoint: ControlEndpoint,
    prepared: PreparedRequest,
    collection: Collection,
    timeout: Duration,
) -> Result<ControlForwardResult, BridgeError> {
    let PreparedRequest {
        request_id,
        frame,
        mut redactions,
    } = prepared;
    let url = format!("ws://127.0.0.1:{}", endpoint.port);
    let outcome = tokio::time::timeout(
        timeout,
        control_round_trip(&url, &endpoint.token, &request_id, frame, collection),
    )
    .await
    .unwrap_or_else(|_| Err(BridgeError::new(ErrorCode::Timeout)));
    redactions.push(endpoint.token);
    let mut messages = outcome?;
    for message in &mut messages {
        redact_value(message, &redactions);
    }
    Ok(ControlForwardResult { messages })
}

// ---------------------------------------------------------------------------------------
// Bridge
// ---------------------------------------------------------------------------------------

type ValidatorResolver = Box<dyn Fn() -> Result<LaunchTarget, BridgeError> + Send + Sync>;

pub struct ControlBridge {
    validator_spawner: Box<dyn ValidatorSpawner>,
    resolve_validator: ValidatorResolver,
    validator_timeout: Duration,
    control_timeout: Duration,
    /// Upper bound on the per-command forwarding timeouts. Tests shorten it; production
    /// leaves the `forward_policy` table in charge.
    forward_timeout_cap: Duration,
    endpoint: EndpointSlot,
    next_id: AtomicU64,
}

impl ControlBridge {
    pub fn new(
        validator_spawner: Box<dyn ValidatorSpawner>,
        resolve_validator: ValidatorResolver,
        validator_timeout: Duration,
        control_timeout: Duration,
    ) -> Self {
        Self {
            validator_spawner,
            resolve_validator,
            validator_timeout,
            control_timeout,
            forward_timeout_cap: Duration::MAX,
            endpoint: Arc::new(Mutex::new(None)),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn production() -> Self {
        Self::new(
            Box::new(OsValidatorSpawner),
            Box::new(|| {
                let engine = engine_process::default_launch_target()
                    .map_err(|_| BridgeError::new(ErrorCode::ValidatorUnavailable))?;
                validator_target(engine)
            }),
            VALIDATOR_TIMEOUT,
            CONTROL_TIMEOUT,
        )
    }

    /// The engine spawner to give `EngineManager`, sharing this bridge's endpoint slot.
    pub fn capturing_spawner(&self, inner: Box<dyn EngineSpawner>) -> CapturingSpawner {
        CapturingSpawner {
            inner,
            slot: Arc::clone(&self.endpoint),
        }
    }

    fn request_id(&self, prefix: &str) -> String {
        format!("{prefix}-{}", self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Blocking: reads the vault and waits on the validator child.
    pub fn test_connection(
        &self,
        store: &CredentialStore,
        request: TestConnectionRequest,
    ) -> Result<ProviderValidationResult, BridgeError> {
        let TestConnectionRequest {
            capability,
            service,
        } = request;
        if !CAPABILITIES.contains(&capability.as_str()) {
            return Err(invalid_request());
        }
        let provider_id = match service.get("providerId") {
            Some(Value::String(id)) if is_provider_id(id) => id.clone(),
            _ => return Err(invalid_request()),
        };
        // Slot names under credentialRefs (e.g. `api_key`) look secret-named by design; that
        // subtree is checked field by field in `credential_refs` instead.
        if service.iter().any(|(key, value)| {
            key != "credentialRefs" && (is_secret_key_name(key) || contains_secret_key(value))
        }) {
            return Err(invalid_request());
        }
        let refs = credential_refs(&capability, &service)?;
        let target = (self.resolve_validator)()?;

        let mut secrets: Vec<(String, Zeroizing<String>)> = Vec::with_capacity(refs.len());
        for (slot, id) in &refs {
            secrets.push((slot.clone(), store.read_secret(id)?));
        }

        let request_id = self.request_id("val");
        let mut body = Zeroizing::new(Vec::with_capacity(REQUEST_BUFFER_BYTES));
        let wire = WireValidationRequest {
            id: &request_id,
            capability: &capability,
            service: &service,
            secrets: secrets
                .iter()
                .map(|(slot, value)| (slot.as_str(), value.as_str()))
                .collect(),
        };
        let serialised = serde_json::to_writer(&mut *body, &wire);
        drop(wire);
        drop(secrets);
        serialised.map_err(|_| invalid_request())?;

        let line = run_validator(
            self.validator_spawner.as_ref(),
            &target,
            body,
            self.validator_timeout,
        )?;
        parse_validation_result(&line, &request_id, &capability, &provider_id)
    }

    /// The captured endpoint, only while `status` still reports the run it came from.
    pub fn control_endpoint(&self, status: EngineStatus) -> Result<ControlEndpoint, BridgeError> {
        if status.state != EngineState::Running {
            return Err(BridgeError::new(ErrorCode::EngineNotRunning));
        }
        match lock_slot(&self.endpoint).as_ref() {
            Some(endpoint) if Some(endpoint.port) == status.port => Ok(ControlEndpoint {
                port: endpoint.port,
                token: endpoint.token.clone(),
            }),
            _ => Err(BridgeError::new(ErrorCode::ControlUnavailable)),
        }
    }

    /// Blocking: builds a `start` request, resolving credential references into config.
    pub fn prepare_start(
        &self,
        store: &CredentialStore,
        request: ForwardStartRequest,
    ) -> Result<PreparedRequest, BridgeError> {
        let ForwardStartRequest {
            mut config,
            credentials,
        } = request;
        // React must never send a raw secret, even under a key a reference would fill.
        for (key, value) in &config {
            if !is_config_key(key) || is_secret_key_name(leaf(key)) || contains_secret_key(value) {
                return Err(invalid_request());
            }
        }
        let mut injected = Vec::with_capacity(credentials.len());
        let mut redactions = Vec::with_capacity(credentials.len());
        let outcome = inject_credentials(
            store,
            &credentials,
            &mut config,
            &mut injected,
            &mut redactions,
        )
        .and_then(|()| {
            let request_id = self.request_id("start");
            request_frame(&request_id, "start", Some(&config)).map(|frame| (request_id, frame))
        });
        scrub(&mut config, &injected);
        let (request_id, frame) = outcome?;
        Ok(PreparedRequest {
            request_id,
            frame,
            redactions,
        })
    }

    pub fn prepare_stop(&self) -> Result<PreparedRequest, BridgeError> {
        let request_id = self.request_id("stop");
        let frame = request_frame(&request_id, "stop", None)?;
        Ok(PreparedRequest {
            request_id,
            frame,
            redactions: Vec::new(),
        })
    }

    /// Sends one prepared request over a fresh control-role connection. The connection is
    /// closed on every path; on timeout the dropped future closes the socket.
    pub async fn forward(
        &self,
        endpoint: ControlEndpoint,
        prepared: PreparedRequest,
    ) -> Result<ControlForwardResult, BridgeError> {
        forward_prepared(endpoint, prepared, START_STOP_COLLECTION, self.control_timeout).await
    }

    /// `control_endpoint` with its errors collapsed onto the forwarding codes.
    pub fn forward_endpoint(&self, status: EngineStatus) -> Result<ControlEndpoint, BridgeError> {
        self.control_endpoint(status).map_err(forward_error)
    }

    /// Blocking: reads a referenced `hf_token` from the vault and serialises the request
    /// frame. The injected value is wiped from the command map once the frame exists.
    pub fn prepare_forward(
        &self,
        store: &CredentialStore,
        checked: CheckedCommand,
    ) -> Result<PreparedRequest, BridgeError> {
        let CheckedCommand {
            mut command,
            size,
            hf_token,
            ..
        } = checked;
        let mut redactions = Vec::new();
        if let Some(id) = &hf_token {
            let secret = store
                .read_secret(id)
                .map_err(|_| BridgeError::new(ErrorCode::VaultError))?;
            command.insert(HF_TOKEN_KEY.to_owned(), Value::String(secret.as_str().to_owned()));
            redactions.push(secret);
        }
        let request_id = self.request_id("cmd");
        // Room for JSON escaping of the secret, so serialising never reallocates.
        let secret_bytes: usize = redactions.iter().map(|secret| secret.len() * 2).sum();
        let capacity = size + secret_bytes + request_id.len() + FORWARD_FRAME_OVERHEAD_BYTES;
        let mut buffer = Zeroizing::new(Vec::with_capacity(capacity));
        let written = serde_json::to_writer(
            &mut *buffer,
            &ForwardFrame {
                op: "request",
                id: &request_id,
                command: &command,
            },
        );
        if hf_token.is_some() {
            if let Some(Value::String(mut value)) = command.remove(HF_TOKEN_KEY) {
                value.zeroize();
            }
        }
        written.map_err(|_| BridgeError::new(ErrorCode::InvalidCommand))?;
        let text = String::from_utf8(std::mem::take(&mut *buffer))
            .map_err(|_| BridgeError::new(ErrorCode::InvalidCommand))?;
        Ok(PreparedRequest {
            request_id,
            frame: Zeroizing::new(text),
            redactions,
        })
    }

    /// Forwards an allowlisted command and collects its replies until the policy's terminal
    /// message type, under the policy's own timeout.
    pub async fn forward_command(
        &self,
        endpoint: ControlEndpoint,
        prepared: PreparedRequest,
        policy: ForwardPolicy,
    ) -> Result<ControlForwardResult, BridgeError> {
        let collection = Collection {
            terminal: policy.terminal,
            max_messages: MAX_FORWARD_RESPONSES,
        };
        let timeout = policy.timeout.min(self.forward_timeout_cap);
        forward_prepared(endpoint, prepared, collection, timeout)
            .await
            .map_err(forward_error)
    }
}

pub type SharedControlBridge = Arc<ControlBridge>;

async fn run_blocking<T, F>(fallback: ErrorCode, job: F) -> Result<T, BridgeError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, BridgeError> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(job)
        .await
        .unwrap_or_else(|_| Err(BridgeError::new(fallback)))
}

// Webview commands. Results are fixed-shape and redacted; none returns a secret or token.
// If the caller goes away mid-validation, the blocking job still runs to its deadline and
// kills the child, so no provider session is left open.

#[tauri::command]
pub async fn provider_test_connection(
    bridge: tauri::State<'_, SharedControlBridge>,
    store: tauri::State<'_, SharedCredentialStore>,
    request: TestConnectionRequest,
) -> Result<ProviderValidationResult, BridgeError> {
    let bridge = Arc::clone(&bridge);
    let store = Arc::clone(&store);
    run_blocking(ErrorCode::ValidatorFailed, move || {
        bridge.test_connection(&store, request)
    })
    .await
}

#[tauri::command]
pub async fn engine_forward_start(
    bridge: tauri::State<'_, SharedControlBridge>,
    manager: tauri::State<'_, SharedEngineManager>,
    store: tauri::State<'_, SharedCredentialStore>,
    request: ForwardStartRequest,
) -> Result<ControlForwardResult, BridgeError> {
    let bridge = Arc::clone(&bridge);
    let job_bridge = Arc::clone(&bridge);
    let manager = Arc::clone(&manager);
    let store = Arc::clone(&store);
    let (endpoint, prepared) = run_blocking(ErrorCode::ControlUnavailable, move || {
        let endpoint = job_bridge.control_endpoint(manager.status())?;
        Ok((endpoint, job_bridge.prepare_start(&store, request)?))
    })
    .await?;
    bridge.forward(endpoint, prepared).await
}

#[tauri::command]
pub async fn engine_forward_stop(
    bridge: tauri::State<'_, SharedControlBridge>,
    manager: tauri::State<'_, SharedEngineManager>,
) -> Result<ControlForwardResult, BridgeError> {
    let bridge = Arc::clone(&bridge);
    let job_bridge = Arc::clone(&bridge);
    let manager = Arc::clone(&manager);
    let (endpoint, prepared) = run_blocking(ErrorCode::ControlUnavailable, move || {
        let endpoint = job_bridge.control_endpoint(manager.status())?;
        Ok((endpoint, job_bridge.prepare_stop()?))
    })
    .await?;
    bridge.forward(endpoint, prepared).await
}

/// Device, model, Hub, and Assist commands for the authenticated desktop path. The webview
/// may reference a vault-held `hf_token` for `download_model` and `search_hub`; it may never
/// send a secret itself.
#[tauri::command]
pub async fn engine_forward_command(
    bridge: tauri::State<'_, SharedControlBridge>,
    manager: tauri::State<'_, SharedEngineManager>,
    store: tauri::State<'_, SharedCredentialStore>,
    command: Value,
    credentials: Option<Value>,
) -> Result<ControlForwardResult, BridgeError> {
    let bridge = Arc::clone(&bridge);
    let job_bridge = Arc::clone(&bridge);
    let manager = Arc::clone(&manager);
    let store = Arc::clone(&store);
    let (endpoint, policy, prepared) = run_blocking(ErrorCode::VaultError, move || {
        let checked = check_forward_command(command, credentials)?;
        let endpoint = job_bridge.forward_endpoint(manager.status())?;
        let policy = checked.policy;
        Ok((endpoint, policy, job_bridge.prepare_forward(&store, checked)?))
    })
    .await?;
    bridge.forward_command(endpoint, prepared, policy).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_process::{development_target, packaged_target, EngineManager};
    use crate::secure_store::{SecretVault, VaultError};
    use serde_json::json;
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::Condvar;

    const SECRET: &str = "test-secret-value-not-real-0001";
    const TOKEN: &str = "c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0";

    // -- Vault -----------------------------------------------------------------------------

    /// In-memory fake; the developer's real vault is never touched.
    #[derive(Default)]
    struct MemoryVault(Mutex<HashMap<String, Zeroizing<String>>>);

    impl SecretVault for MemoryVault {
        fn write(&self, account: &str, secret: &str) -> Result<(), VaultError> {
            let mut entries = self.0.lock().unwrap();
            entries.insert(account.to_owned(), Zeroizing::new(secret.to_owned()));
            Ok(())
        }

        fn read(&self, account: &str) -> Result<Option<Zeroizing<String>>, VaultError> {
            Ok(self.0.lock().unwrap().get(account).cloned())
        }

        fn remove(&self, account: &str) -> Result<bool, VaultError> {
            Ok(self.0.lock().unwrap().remove(account).is_some())
        }
    }

    fn empty_store() -> CredentialStore {
        CredentialStore::new(Box::new(MemoryVault::default()))
    }

    fn store_with(capability: &str, config_id: &str) -> CredentialStore {
        let store = empty_store();
        let id = CredentialId::parse(capability, config_id, "api_key").unwrap();
        store.save(&id, SECRET).unwrap();
        store
    }

    // -- Fake validator child --------------------------------------------------------------

    #[derive(Clone, Copy, PartialEq)]
    enum Reply {
        Echo,
        WrongId,
        ExtraField,
        Garbage,
        Silent,
        Hang,
    }

    #[derive(Default)]
    struct PipeState {
        request: Vec<u8>,
        stdin_closed: bool,
        replied: bool,
        killed: bool,
    }

    #[derive(Default)]
    struct FakeRun {
        state: Mutex<PipeState>,
        wake: Condvar,
        kills: AtomicUsize,
    }

    impl FakeRun {
        fn request(&self) -> Vec<u8> {
            self.state.lock().unwrap().request.clone()
        }
    }

    struct FakeStdin(Arc<FakeRun>);

    impl Write for FakeStdin {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.state.lock().unwrap().request.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Drop for FakeStdin {
        fn drop(&mut self) {
            self.0.state.lock().unwrap().stdin_closed = true;
            self.0.wake.notify_all();
        }
    }

    struct FakeStdout {
        run: Arc<FakeRun>,
        reply: Reply,
        pending: Vec<u8>,
        answered: bool,
    }

    impl Read for FakeStdout {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if !self.answered {
                let mut state = self.run.state.lock().unwrap();
                while !state.killed && (!state.stdin_closed || self.reply == Reply::Hang) {
                    state = self.run.wake.wait(state).unwrap();
                }
                if state.killed {
                    return Ok(0);
                }
                self.pending = reply_bytes(self.reply, &state.request);
                state.replied = true;
                self.answered = true;
            }
            let count = buf.len().min(self.pending.len());
            buf[..count].copy_from_slice(&self.pending[..count]);
            self.pending.drain(..count);
            Ok(count)
        }
    }

    struct FakeProcess {
        run: Arc<FakeRun>,
        reply: Reply,
        taken: bool,
    }

    impl ValidatorProcess for FakeProcess {
        fn take_pipes(&mut self) -> Option<ValidatorPipes> {
            if self.taken {
                return None;
            }
            self.taken = true;
            Some(ValidatorPipes {
                stdin: Box::new(FakeStdin(Arc::clone(&self.run))),
                stdout: Box::new(FakeStdout {
                    run: Arc::clone(&self.run),
                    reply: self.reply,
                    pending: Vec::new(),
                    answered: false,
                }),
            })
        }

        fn has_exited(&mut self) -> io::Result<bool> {
            let state = self.run.state.lock().unwrap();
            Ok(state.killed || state.replied)
        }

        fn kill(&mut self) -> io::Result<()> {
            self.run.kills.fetch_add(1, Ordering::SeqCst);
            self.run.state.lock().unwrap().killed = true;
            self.run.wake.notify_all();
            Ok(())
        }
    }

    struct FakeValidatorSpawner {
        reply: Reply,
        runs: Mutex<Vec<(LaunchTarget, Arc<FakeRun>)>>,
    }

    impl ValidatorSpawner for Arc<FakeValidatorSpawner> {
        fn spawn(&self, target: &LaunchTarget) -> io::Result<Box<dyn ValidatorProcess>> {
            let run = Arc::new(FakeRun::default());
            self.runs.lock().unwrap().push((target.clone(), Arc::clone(&run)));
            Ok(Box::new(FakeProcess {
                run,
                reply: self.reply,
                taken: false,
            }))
        }
    }

    impl FakeValidatorSpawner {
        fn run(&self, index: usize) -> (LaunchTarget, Arc<FakeRun>) {
            let runs = self.runs.lock().unwrap();
            (runs[index].0.clone(), Arc::clone(&runs[index].1))
        }

        fn spawn_count(&self) -> usize {
            self.runs.lock().unwrap().len()
        }
    }

    /// Plays the Python side: answers from the request it actually received.
    fn reply_bytes(reply: Reply, request: &[u8]) -> Vec<u8> {
        let request: Value = serde_json::from_slice(request).unwrap();
        let mut result = json!({
            "id": request["id"],
            "capability": request["capability"],
            "providerId": request["service"]["providerId"],
            "status": "valid",
            "latencyMs": 12.5,
            "testedAt": 1_700_000_000.25,
            "fingerprint": "ab".repeat(32),
            "credentialVersion": 2,
            "errorCode": null,
        });
        match reply {
            Reply::Echo => {}
            Reply::WrongId => result["id"] = json!("val-999999"),
            Reply::ExtraField => result["message"] = json!(format!("provider said {SECRET}")),
            Reply::Garbage => return format!("Traceback {SECRET}\n").into_bytes(),
            Reply::Silent => return Vec::new(),
            Reply::Hang => unreachable!("a hanging child never answers"),
        }
        let mut bytes = serde_json::to_vec(&result).unwrap();
        bytes.push(b'\n');
        bytes
    }

    fn bridge_with(reply: Reply, validator_timeout: Duration) -> (ControlBridge, Arc<FakeValidatorSpawner>) {
        let spawner = Arc::new(FakeValidatorSpawner {
            reply,
            runs: Mutex::new(Vec::new()),
        });
        let bridge = ControlBridge::new(
            Box::new(Arc::clone(&spawner)),
            Box::new(|| validator_target(development_target(Path::new("repo-root")))),
            validator_timeout,
            Duration::from_millis(300),
        );
        (bridge, spawner)
    }

    fn stt_request(service_extra: Value) -> TestConnectionRequest {
        let mut service = json!({
            "providerId": "openai-realtime",
            "endpoint": "",
            "modelOverride": "",
            "options": {"language": "auto"},
            "credentialRefs": {
                "api_key": {"configId": "openai-main", "slot": "api_key", "version": 2}
            },
        });
        if let (Value::Object(base), Value::Object(extra)) = (&mut service, service_extra) {
            base.extend(extra);
        }
        serde_json::from_value(json!({"capability": "stt", "service": service})).unwrap()
    }

    fn assert_safe_error(err: &BridgeError, code: ErrorCode) {
        assert_eq!(err.code, code);
        let json = serde_json::to_value(err).unwrap();
        let keys: Vec<&str> = json.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys.len(), 2);
        let rendered = format!("{json} {err} {err:?}");
        for forbidden in [SECRET, TOKEN, "openai-main"] {
            assert!(!rendered.contains(forbidden), "error leaked {forbidden:?}: {rendered}");
        }
    }

    // -- Test Connection -------------------------------------------------------------------

    #[test]
    fn test_connection_sends_secret_only_on_stdin_and_returns_safe_fields() {
        let (bridge, spawner) = bridge_with(Reply::Echo, Duration::from_secs(5));
        let store = store_with("stt", "openai-main");
        let result = bridge.test_connection(&store, stt_request(json!({}))).unwrap();

        assert_eq!(result.capability, "stt");
        assert_eq!(result.provider_id, "openai-realtime");
        assert_eq!(result.status, ValidationStatus::Valid);
        assert_eq!(result.fingerprint.as_deref(), Some("ab".repeat(32).as_str()));
        assert_eq!(result.credential_version, Some(2));
        assert_eq!(result.error_code, None);

        let json = serde_json::to_value(&result).unwrap();
        let mut keys: Vec<&str> = json.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "capability",
                "credentialVersion",
                "errorCode",
                "fingerprint",
                "latencyMs",
                "providerId",
                "status",
                "testedAt"
            ]
        );
        assert!(!format!("{json} {result:?}").contains(SECRET));

        let (target, run) = spawner.run(0);
        assert_eq!(target.args, ["-m", VALIDATOR_MODULE]);
        assert!(!target.args.join(" ").contains(SECRET));
        let request: Value = serde_json::from_slice(&run.request()).unwrap();
        assert_eq!(request["secrets"], json!({"api_key": SECRET}));
        assert_eq!(request["capability"], "stt");
        assert!(request["id"].as_str().unwrap().starts_with("val-"));
        assert_eq!(request["service"]["credentialRefs"]["api_key"]["version"], 2);
        let state = run.state.lock().unwrap();
        assert!(state.stdin_closed, "stdin must be closed so the child sees EOF");
    }

    #[test]
    fn hanging_validator_times_out_and_is_killed() {
        let (bridge, spawner) = bridge_with(Reply::Hang, Duration::from_millis(100));
        let store = store_with("stt", "openai-main");
        let started = Instant::now();
        let err = bridge.test_connection(&store, stt_request(json!({}))).unwrap_err();
        assert_safe_error(&err, ErrorCode::Timeout);
        assert!(started.elapsed() < Duration::from_secs(3));
        let (_, run) = spawner.run(0);
        assert_eq!(run.kills.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn mismatched_request_id_is_rejected() {
        let (bridge, spawner) = bridge_with(Reply::WrongId, Duration::from_secs(5));
        let store = store_with("stt", "openai-main");
        let err = bridge.test_connection(&store, stt_request(json!({}))).unwrap_err();
        assert_safe_error(&err, ErrorCode::CorrelationMismatch);
        assert!(spawner.run(0).1.state.lock().unwrap().replied);
    }

    #[test]
    fn unexpected_output_is_rejected_without_echo_and_child_is_reaped() {
        for (reply, code) in [
            (Reply::ExtraField, ErrorCode::InvalidResponse),
            (Reply::Garbage, ErrorCode::InvalidResponse),
            (Reply::Silent, ErrorCode::ValidatorFailed),
        ] {
            let (bridge, spawner) = bridge_with(reply, Duration::from_secs(5));
            let store = store_with("stt", "openai-main");
            let err = bridge.test_connection(&store, stt_request(json!({}))).unwrap_err();
            assert_safe_error(&err, code);
            let (_, run) = spawner.run(0);
            let state = run.state.lock().unwrap();
            assert!(state.replied || state.killed);
        }
    }

    #[test]
    fn missing_credential_never_spawns() {
        let (bridge, spawner) = bridge_with(Reply::Echo, Duration::from_secs(5));
        let err = bridge.test_connection(&empty_store(), stt_request(json!({}))).unwrap_err();
        assert_safe_error(&err, ErrorCode::CredentialNotConfigured);
        assert_eq!(spawner.spawn_count(), 0);
    }

    #[test]
    fn raw_secrets_and_malformed_requests_are_refused_before_the_vault() {
        let (bridge, spawner) = bridge_with(Reply::Echo, Duration::from_secs(5));
        let store = store_with("stt", "openai-main");
        let bad_services = [
            json!({"options": {"apiKey": SECRET}}),
            json!({"secrets": {"api_key": SECRET}}),
            json!({"providerId": "Not Valid"}),
            json!({"credentialRefs": {"api_key": {"configId": "openai-main", "slot": "other"}}}),
            json!({"credentialRefs": {"api_key": {"configId": "openai-main", "slot": "api_key", "value": SECRET}}}),
            json!({"credentialRefs": {"api_key": SECRET}}),
        ];
        for extra in bad_services {
            let err = bridge.test_connection(&store, stt_request(extra)).unwrap_err();
            assert_safe_error(&err, ErrorCode::InvalidRequest);
        }
        let mut wrong_capability = stt_request(json!({}));
        wrong_capability.capability = "llm".to_owned();
        let err = bridge.test_connection(&store, wrong_capability).unwrap_err();
        assert_safe_error(&err, ErrorCode::InvalidRequest);

        let unknown: Result<TestConnectionRequest, _> = serde_json::from_value(
            json!({"capability": "stt", "service": {}, "apiKey": SECRET}),
        );
        assert!(unknown.is_err());
        assert_eq!(spawner.spawn_count(), 0);
    }

    #[test]
    fn validator_target_uses_the_dev_interpreter_and_fails_closed_when_packaged() {
        let dev = validator_target(development_target(Path::new("checkout"))).unwrap();
        assert_eq!(dev.args, ["-m", VALIDATOR_MODULE]);
        assert!(dev.program.starts_with(Path::new("checkout").join(".venv")));
        assert_eq!(dev.working_dir, Path::new("checkout"));

        let err = validator_target(packaged_target(Path::new("install"))).unwrap_err();
        assert_safe_error(&err, ErrorCode::ValidatorUnavailable);
    }

    #[test]
    fn result_parsing_enforces_shape_and_consistency() {
        let base = json!({
            "id": "val-1", "capability": "translation", "providerId": "openai",
            "status": "invalid", "latencyMs": null, "testedAt": 1.5,
            "fingerprint": "0".repeat(64), "credentialVersion": null,
            "errorCode": "auth_failed",
        });
        let parse = |value: &Value| {
            parse_validation_result(&serde_json::to_vec(value).unwrap(), "val-1", "translation", "openai")
        };
        let ok = parse(&base).unwrap();
        assert_eq!((ok.status, ok.error_code.as_deref()), (ValidationStatus::Invalid, Some("auth_failed")));

        let unvalidated = parse(&{
            let mut v = base.clone();
            v["status"] = json!("unvalidated");
            v["errorCode"] = json!("validation_not_supported");
            v
        })
        .unwrap();
        assert_eq!(unvalidated.status, ValidationStatus::Unvalidated);

        let cases: Vec<(Value, ErrorCode)> = vec![
            (json!({"status": "valid"}), ErrorCode::InvalidResponse),
            (json!({"errorCode": null}), ErrorCode::InvalidResponse),
            (json!({"errorCode": format!("Bad {SECRET}")}), ErrorCode::InvalidResponse),
            (json!({"fingerprint": "XYZ"}), ErrorCode::InvalidResponse),
            (json!({"testedAt": -1.0}), ErrorCode::InvalidResponse),
            (json!({"latencyMs": -3.0}), ErrorCode::InvalidResponse),
            (json!({"status": "maybe"}), ErrorCode::InvalidResponse),
            (json!({"providerId": "other"}), ErrorCode::InvalidResponse),
            (json!({"providerId": null}), ErrorCode::InvalidRequest),
            (json!({"id": null}), ErrorCode::ValidatorFailed),
            (json!({"id": "val-2"}), ErrorCode::CorrelationMismatch),
        ];
        for (patch, code) in cases {
            let mut value = base.clone();
            for (key, item) in patch.as_object().unwrap() {
                value[key] = item.clone();
            }
            let err = parse(&value).unwrap_err();
            assert_eq!(err.code, code, "patch {patch}");
        }
        let valid_without_fingerprint = parse(&{
            let mut v = base.clone();
            v["status"] = json!("valid");
            v["errorCode"] = json!(null);
            v["fingerprint"] = json!(null);
            v
        });
        assert_eq!(valid_without_fingerprint.unwrap_err().code, ErrorCode::InvalidResponse);
    }

    // -- Endpoint capture ------------------------------------------------------------------

    struct QuietChild(Arc<AtomicBool>);

    impl EngineChild for QuietChild {
        fn has_exited(&mut self) -> io::Result<bool> {
            Ok(self.0.load(Ordering::SeqCst))
        }

        fn request_stop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }

        fn kill(&mut self) -> io::Result<()> {
            self.0.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    #[derive(Default)]
    struct FlakySpawner {
        fail: AtomicBool,
    }

    impl EngineSpawner for Arc<FlakySpawner> {
        fn spawn(&self, _spec: &LaunchSpec) -> io::Result<Box<dyn EngineChild>> {
            if self.fail.load(Ordering::SeqCst) {
                return Err(io::Error::new(io::ErrorKind::Other, "spawn failed"));
            }
            Ok(Box::new(QuietChild(Arc::default())))
        }
    }

    #[test]
    fn capturing_spawner_exposes_the_token_only_for_the_live_run() {
        let (bridge, _) = bridge_with(Reply::Echo, Duration::from_secs(5));
        let inner = Arc::new(FlakySpawner::default());
        let manager = EngineManager::new(
            Box::new(bridge.capturing_spawner(Box::new(Arc::clone(&inner)))),
            Box::new(|| Ok(development_target(Path::new("repo-root")))),
            Duration::from_millis(20),
        );

        let err = bridge.control_endpoint(manager.status()).unwrap_err();
        assert_safe_error(&err, ErrorCode::EngineNotRunning);

        let info = manager.start().unwrap();
        let endpoint = bridge.control_endpoint(manager.status()).unwrap();
        assert_eq!(endpoint.port, info.port);
        assert_eq!(endpoint.token.len(), 64);
        assert!(endpoint.token.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_ne!(endpoint.token.as_str(), info.event_token);
        assert!(!format!("{endpoint:?}").contains(endpoint.token.as_str()));

        let stale = EngineStatus {
            state: EngineState::Running,
            port: Some(info.port.wrapping_add(1)),
        };
        assert_safe_error(&bridge.control_endpoint(stale).unwrap_err(), ErrorCode::ControlUnavailable);

        manager.stop().unwrap();
        assert_safe_error(
            &bridge.control_endpoint(manager.status()).unwrap_err(),
            ErrorCode::EngineNotRunning,
        );

        // A failed spawn clears the previous run's token.
        inner.fail.store(true, Ordering::SeqCst);
        assert!(manager.start().is_err());
        let running_elsewhere = EngineStatus {
            state: EngineState::Running,
            port: Some(info.port),
        };
        assert_safe_error(
            &bridge.control_endpoint(running_elsewhere).unwrap_err(),
            ErrorCode::ControlUnavailable,
        );
    }

    // -- Start/Stop request building -------------------------------------------------------

    fn start_request(value: Value) -> ForwardStartRequest {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn start_frame_injects_referenced_secrets_and_rejects_raw_ones() {
        let (bridge, _) = bridge_with(Reply::Echo, Duration::from_secs(5));
        let store = store_with("assistant", "openai-main");
        let prepared = bridge
            .prepare_start(
                &store,
                start_request(json!({
                    "config": {"asr.model_size": "small", "llm.enabled": true},
                    "credentials": {
                        "llm.api_key": {"capability": "assistant", "configId": "openai-main", "slot": "api_key"}
                    }
                })),
            )
            .unwrap();
        let frame: Value = serde_json::from_str(&prepared.frame).unwrap();
        assert_eq!(frame["op"], "request");
        assert_eq!(frame["id"], prepared.request_id.as_str());
        assert!(prepared.request_id.starts_with("start-"));
        assert_eq!(frame["command"]["type"], "start");
        assert_eq!(frame["command"]["config"]["asr.model_size"], "small");
        assert_eq!(frame["command"]["config"]["llm.api_key"], SECRET);
        assert_eq!(prepared.redactions.len(), 1);
        assert_eq!(prepared.redactions[0].as_str(), SECRET);

        let prepared_with_services = bridge
            .prepare_start(
                &store,
                start_request(json!({
                    "config": {
                        "services": {
                            "stt": {
                                "credentialRefs": {
                                    "api_key": {"configId": "dg-main", "slot": "api_key", "version": 1}
                                }
                            }
                        }
                    },
                    "credentials": {}
                })),
            )
            .unwrap();
        let frame_services: Value = serde_json::from_str(&prepared_with_services.frame).unwrap();
        assert_eq!(
            frame_services["command"]["config"]["services"]["stt"]["credentialRefs"]["api_key"]["slot"],
            "api_key"
        );

        let stop = bridge.prepare_stop().unwrap();
        let frame: Value = serde_json::from_str(&stop.frame).unwrap();
        assert_eq!(frame, json!({"op": "request", "id": stop.request_id, "command": {"type": "stop"}}));

        let rejected = [
            json!({"config": {"llm.api_key": SECRET}}),
            json!({"config": {"llm": {"token": SECRET}}}),
            json!({"config": {"Bad Key": 1}}),
            json!({"config": {}, "credentials": {"llm.model": {"capability": "assistant", "configId": "openai-main", "slot": "api_key"}}}),
            json!({"config": {}, "credentials": {"llm.api_key": {"capability": "vault", "configId": "openai-main", "slot": "api_key"}}}),
        ];
        for value in rejected {
            let err = bridge.prepare_start(&store, start_request(value)).unwrap_err();
            assert_safe_error(&err, ErrorCode::InvalidRequest);
        }
        let missing = start_request(json!({
            "config": {},
            "credentials": {"llm.api_key": {"capability": "assistant", "configId": "other-main", "slot": "api_key"}}
        }));
        assert_safe_error(
            &bridge.prepare_start(&store, missing).unwrap_err(),
            ErrorCode::CredentialNotConfigured,
        );
    }

    // -- Control-role client against a fake engine -----------------------------------------

    #[derive(Clone, Copy)]
    enum Script {
        Happy,
        RejectAuth,
        Silent,
        CommandError,
    }

    #[derive(Default)]
    struct ServerLog {
        frames: Vec<Value>,
        client_closed: bool,
    }

    type ServerSocket = tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>;

    async fn recv_json(socket: &mut ServerSocket) -> Value {
        loop {
            match socket.next().await {
                Some(Ok(Message::Text(text))) => return serde_json::from_str(&text).unwrap(),
                Some(Ok(_)) => continue,
                other => panic!("fake engine expected a frame, got {other:?}"),
            }
        }
    }

    async fn send_json(socket: &mut ServerSocket, value: Value) {
        socket.send(Message::Text(value.to_string())).await.unwrap();
    }

    /// Loopback fake of the engine's control role. Records frames and whether the client
    /// closed its connection.
    async fn fake_engine(script: Script) -> (u16, tokio::task::JoinHandle<ServerLog>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
            let mut log = ServerLog::default();
            log.frames.push(recv_json(&mut socket).await);
            if let Script::RejectAuth = script {
                send_json(&mut socket, json!({"op": "error", "code": "auth_failed", "message": "Authentication failed"})).await;
                let _ = socket.close(None).await;
            } else {
                send_json(&mut socket, json!({"op": "auth_ok", "role": "control", "protocol": 1})).await;
                let request = recv_json(&mut socket).await;
                let id = request["id"].as_str().unwrap().to_owned();
                log.frames.push(request);
                match script {
                    Script::Happy => {
                        send_json(&mut socket, json!({"op": "response", "id": "someone-else", "message": {"type": "error"}})).await;
                        send_json(&mut socket, json!({"op": "response", "id": id, "message": {
                            "type": "status", "status": "started", "note": format!("{TOKEN} {SECRET}")
                        }})).await;
                        send_json(&mut socket, json!({"op": "complete", "id": "someone-else"})).await;
                        send_json(&mut socket, json!({"op": "complete", "id": id})).await;
                    }
                    Script::CommandError => {
                        send_json(&mut socket, json!({"op": "error", "code": "command_failed", "message": "Command failed", "id": id})).await;
                    }
                    Script::Silent | Script::RejectAuth => {}
                }
            }
            loop {
                match socket.next().await {
                    None | Some(Err(_)) | Some(Ok(Message::Close(_))) => break,
                    Some(Ok(_)) => continue,
                }
            }
            log.client_closed = true;
            log
        });
        (port, handle)
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    fn endpoint(port: u16) -> ControlEndpoint {
        ControlEndpoint {
            port,
            token: Zeroizing::new(TOKEN.to_owned()),
        }
    }

    #[test]
    fn forwarded_command_authenticates_correlates_and_redacts() {
        runtime().block_on(async {
            let (port, server) = fake_engine(Script::Happy).await;
            let (bridge, _) = bridge_with(Reply::Echo, Duration::from_secs(5));
            let mut prepared = bridge.prepare_stop().unwrap();
            prepared.redactions.push(Zeroizing::new(SECRET.to_owned()));
            let request_id = prepared.request_id.clone();

            let result = bridge.forward(endpoint(port), prepared).await.unwrap();
            let log = server.await.unwrap();

            assert_eq!(log.frames[0], json!({"op": "auth", "role": "control", "token": TOKEN, "protocol": 1}));
            assert_eq!(log.frames[1], json!({"op": "request", "id": request_id, "command": {"type": "stop"}}));
            assert_eq!(result.messages.len(), 1);
            assert_eq!(result.messages[0]["status"], "started");
            let rendered = serde_json::to_string(&result).unwrap();
            assert!(!rendered.contains(TOKEN) && !rendered.contains(SECRET), "{rendered}");
            assert!(rendered.contains(REDACTED));
            assert!(log.client_closed);
        });
    }

    #[test]
    fn auth_rejection_and_command_errors_are_fixed_codes() {
        runtime().block_on(async {
            for (script, code) in [
                (Script::RejectAuth, ErrorCode::AuthFailed),
                (Script::CommandError, ErrorCode::CommandRejected),
            ] {
                let (port, server) = fake_engine(script).await;
                let (bridge, _) = bridge_with(Reply::Echo, Duration::from_secs(5));
                let err = bridge.forward(endpoint(port), bridge.prepare_stop().unwrap()).await.unwrap_err();
                assert_safe_error(&err, code);
                assert!(server.await.unwrap().client_closed);
            }
        });
    }

    #[test]
    fn silent_engine_times_out_and_the_socket_is_closed() {
        runtime().block_on(async {
            let (port, server) = fake_engine(Script::Silent).await;
            let (bridge, _) = bridge_with(Reply::Echo, Duration::from_secs(5));
            let started = Instant::now();
            let err = bridge.forward(endpoint(port), bridge.prepare_stop().unwrap()).await.unwrap_err();
            assert_safe_error(&err, ErrorCode::Timeout);
            assert!(started.elapsed() < Duration::from_secs(3));
            let log = tokio::time::timeout(Duration::from_secs(3), server).await.unwrap().unwrap();
            assert!(log.client_closed);
        });
    }

    #[test]
    fn unreachable_engine_is_control_unavailable() {
        runtime().block_on(async {
            // Accept and immediately drop: the WebSocket handshake fails at once. (A refused
            // connect is retried by Windows for longer than the test's control timeout.)
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            tokio::spawn(async move {
                if let Ok((stream, _)) = listener.accept().await {
                    drop(stream);
                }
            });
            let (bridge, _) = bridge_with(Reply::Echo, Duration::from_secs(5));
            let err = bridge.forward(endpoint(port), bridge.prepare_stop().unwrap()).await.unwrap_err();
            assert_safe_error(&err, ErrorCode::ControlUnavailable);
        });
    }

    // -- Allowlisted command forwarding ----------------------------------------------------

    enum Step {
        Send(Value),
        Pause(Duration),
    }

    /// Loopback fake of the control role that plays `script(request_id)` after the request,
    /// then waits for the client to close. Sends after the client has gone are ignored.
    async fn scripted_engine<F>(script: F) -> (u16, tokio::task::JoinHandle<ServerLog>)
    where
        F: FnOnce(&str) -> Vec<Step> + Send + 'static,
    {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
            let mut log = ServerLog::default();
            log.frames.push(recv_json(&mut socket).await);
            send_json(&mut socket, json!({"op": "auth_ok", "role": "control", "protocol": 1})).await;
            let request = recv_json(&mut socket).await;
            let id = request["id"].as_str().unwrap().to_owned();
            log.frames.push(request);
            for step in script(&id) {
                match step {
                    Step::Send(frame) => {
                        let _ = socket.send(Message::Text(frame.to_string())).await;
                    }
                    Step::Pause(delay) => tokio::time::sleep(delay).await,
                }
            }
            loop {
                match socket.next().await {
                    None | Some(Err(_)) | Some(Ok(Message::Close(_))) => break,
                    Some(Ok(_)) => continue,
                }
            }
            log.client_closed = true;
            log
        });
        (port, handle)
    }

    fn forward_bridge(cap: Duration) -> ControlBridge {
        let (mut bridge, _) = bridge_with(Reply::Echo, Duration::from_secs(5));
        bridge.forward_timeout_cap = cap;
        bridge
    }

    fn hf_ref() -> Value {
        json!({"hf_token": {"capability": "stt", "configId": "huggingface", "slot": "hf_token"}})
    }

    async fn forward_json(
        bridge: &ControlBridge,
        store: &CredentialStore,
        port: u16,
        command: Value,
        credentials: Option<Value>,
    ) -> Result<ControlForwardResult, BridgeError> {
        let checked = check_forward_command(command, credentials)?;
        let policy = checked.policy;
        let prepared = bridge.prepare_forward(store, checked)?;
        bridge.forward_command(endpoint(port), prepared, policy).await
    }

    fn message_types(result: &ControlForwardResult) -> Vec<&str> {
        result.messages.iter().map(|m| m["type"].as_str().unwrap_or("")).collect()
    }

    #[test]
    fn forward_allowlist_maps_each_command_to_its_terminal_and_timeout() {
        let table = [
            ("list_devices", Some("devices_list"), 10, false),
            ("request_models_list", Some("models_list"), 10, false),
            ("download_model", None, 20, true),
            ("delete_model", Some("model_action_result"), 30, false),
            ("search_hub", Some("hub_search_results"), 30, true),
            ("request_suggestion", Some("suggestion_result"), 120, false),
            ("request_assist", None, 120, false),
            ("request_summary", Some("llm_done"), 300, false),
            ("test_connection", Some("test_connection_result"), 30, false),
            ("profile_create", None, 10, false),
            ("profile_list", None, 10, false),
            ("profile_get", None, 10, false),
            ("profile_update", None, 10, false),
            ("profile_delete", None, 10, false),
            ("profile_attach", None, 10, false),
            ("conversation_create", None, 10, false),
            ("conversation_start", None, 10, false),
            ("conversation_list", None, 10, false),
            ("conversation_get", None, 15, false),
            ("conversation_rename", None, 10, false),
            ("conversation_delete", None, 10, false),
            ("conversation_export", None, 30, false),
            ("conversation_import", None, 30, false),
            ("pack_status", None, 10, false),
            ("pack_install", None, 30, false),
            ("pack_cancel", None, 10, false),
            ("pack_retry", None, 30, false),
            ("pack_check_readiness", None, 15, false),
            ("get_provider_manifests", None, 10, false),
            ("summary_cancel", None, 10, false),
        ];
        for (kind, terminal, seconds, accepts_hf_token) in table {
            let checked = check_forward_command(json!({"type": kind}), None).unwrap();
            let expected = ForwardPolicy {
                terminal,
                timeout: Duration::from_secs(seconds),
                accepts_hf_token,
            };
            assert_eq!(checked.policy, expected, "{kind}");
            assert!(checked.hf_token.is_none());
        }

        for kind in ["start", "stop", "Start", "unknown", "", "partial"] {
            let err = check_forward_command(json!({"type": kind, "config": {}}), None).unwrap_err();
            assert_safe_error(&err, ErrorCode::CommandNotAllowed);
        }

        let text = |len: usize| json!({"type": "request_summary", "entries": [{"source": "mic", "text": "x".repeat(len)}]});
        let malformed = [
            json!(null),
            json!("list_devices"),
            json!(["list_devices"]),
            json!({}),
            json!({"type": 7}),
            json!({"kind": "list_devices"}),
            text(MAX_FORWARD_COMMAND_BYTES),
        ];
        for command in malformed {
            assert_safe_error(&check_forward_command(command, None).unwrap_err(), ErrorCode::InvalidCommand);
        }
        assert!(check_forward_command(text(MAX_FORWARD_COMMAND_BYTES - 256), None).is_ok());
    }

    #[test]
    fn forward_rejects_secret_named_keys_at_any_depth_without_echo() {
        let commands = [
            json!({"type": "download_model", "model_id": "tiny", "model_type": "asr", "hf_token": SECRET}),
            json!({"type": "search_hub", "query": "whisper", "HF-Token": SECRET}),
            json!({"type": "request_suggestion", "entry_id": "e-1", "llm_config": {"api_key": SECRET}}),
            json!({"type": "request_summary", "entries": [{"source": "mic", "text": "hi", "meta": {"apiKey": SECRET}}]}),
            json!({"type": "list_devices", "headers": [{"Authorization": format!("Bearer {SECRET}")}]}),
            json!({"type": "delete_model", "nested": {"deep": [{"clientSecret": SECRET}]}}),
            json!({"type": "request_models_list", "PASSWORD": SECRET}),
            json!({"type": "request_models_list", "credentials": {"x": SECRET}}),
            json!({"type": "request_models_list", "BEARER": SECRET}),
        ];
        for command in commands {
            let err = check_forward_command(command, None).unwrap_err();
            assert_safe_error(&err, ErrorCode::SecretInCommand);
        }

        let reference = || json!({"capability": "stt", "configId": "huggingface", "slot": "hf_token"});
        let download = || json!({"type": "download_model", "model_id": "tiny", "model_type": "asr"});
        let bad_credentials = [
            (json!({"type": "list_devices"}), json!({"hf_token": reference()})),
            (json!({"type": "request_summary", "entries": []}), json!({"hf_token": reference()})),
            (download(), json!({"api_key": reference()})),
            (download(), json!({"hf_token": reference(), "extra": reference()})),
            (download(), json!({"hf_token": SECRET})),
            (download(), json!([SECRET])),
            (
                download(),
                json!({"hf_token": {"capability": "stt", "configId": "huggingface", "slot": "hf_token", "value": SECRET}}),
            ),
            (
                download(),
                json!({"hf_token": {"capability": "vault", "configId": "huggingface", "slot": "hf_token"}}),
            ),
        ];
        for (command, credentials) in bad_credentials {
            let err = check_forward_command(command, Some(credentials)).unwrap_err();
            assert_safe_error(&err, ErrorCode::InvalidCommand);
        }

        for credentials in [json!({}), json!(null)] {
            let checked = check_forward_command(json!({"type": "list_devices"}), Some(credentials)).unwrap();
            assert!(checked.hf_token.is_none());
        }
    }

    #[test]
    fn forward_injects_hf_token_from_the_vault_and_redacts_it() {
        runtime().block_on(async {
            let store = empty_store();
            let id = CredentialId::parse("stt", "huggingface", "hf_token").unwrap();
            store.save(&id, SECRET).unwrap();
            let (port, server) = scripted_engine(|id| {
                vec![
                    Step::Send(json!({"op": "response", "id": id, "message": {
                        "type": "status", "note": format!("{SECRET} {TOKEN}")
                    }})),
                    Step::Send(json!({"op": "complete", "id": id})),
                ]
            })
            .await;
            let bridge = forward_bridge(Duration::from_secs(5));
            let checked = check_forward_command(
                json!({"type": "download_model", "model_id": "tiny", "model_type": "asr"}),
                Some(hf_ref()),
            )
            .unwrap();
            let policy = checked.policy;
            let prepared = bridge.prepare_forward(&store, checked).unwrap();
            assert!(prepared.request_id.starts_with("cmd-"));
            assert!(!format!("{prepared:?}").contains(SECRET));
            let request_id = prepared.request_id.clone();

            let result = bridge.forward_command(endpoint(port), prepared, policy).await.unwrap();
            let log = server.await.unwrap();

            assert_eq!(log.frames[0], json!({"op": "auth", "role": "control", "token": TOKEN, "protocol": 1}));
            assert_eq!(
                log.frames[1],
                json!({"op": "request", "id": request_id, "command": {
                    "type": "download_model", "model_id": "tiny", "model_type": "asr", "hf_token": SECRET
                }})
            );
            // download_model returns at `complete`; its progress arrives as events.
            assert_eq!(message_types(&result), ["status"]);
            let rendered = format!("{} {result:?}", serde_json::to_string(&result).unwrap());
            assert!(!rendered.contains(SECRET) && !rendered.contains(TOKEN), "{rendered}");
            assert!(rendered.contains(REDACTED));
            assert!(log.client_closed);

            let checked =
                check_forward_command(json!({"type": "search_hub", "query": "whisper"}), Some(hf_ref())).unwrap();
            assert!(checked.hf_token.is_some());
            let err = bridge.prepare_forward(&empty_store(), checked).unwrap_err();
            assert_safe_error(&err, ErrorCode::VaultError);
        });
    }

    #[test]
    fn forward_keeps_collecting_after_complete_until_the_terminal_reply() {
        runtime().block_on(async {
            let bridge = forward_bridge(Duration::from_secs(5));
            let store = empty_store();

            // A worker-thread reply after `complete`, with unrelated traffic in between.
            let (port, server) = scripted_engine(|id| {
                vec![
                    Step::Send(json!({"op": "complete", "id": id})),
                    Step::Pause(Duration::from_millis(50)),
                    Step::Send(json!({"op": "event", "event": {"type": "partial", "text": "hi"}})),
                    Step::Send(json!({"op": "response", "id": "cmd-other", "message": {"type": "suggestion_result"}})),
                    Step::Send(json!({"op": "response", "id": id, "message": {
                        "type": "suggestion_result", "entry_id": "e-1", "options": ["a"]
                    }})),
                ]
            })
            .await;
            let command = json!({"type": "request_suggestion", "entry_id": "e-1", "target_text": "q?", "context": []});
            let result = forward_json(&bridge, &store, port, command, None).await.unwrap();
            assert_eq!(result.messages, [json!({"type": "suggestion_result", "entry_id": "e-1", "options": ["a"]})]);
            assert!(server.await.unwrap().client_closed);

            // A terminal reply before `complete` is kept and the request still ends cleanly.
            let (port, server) = scripted_engine(|id| {
                vec![
                    Step::Send(json!({"op": "response", "id": id, "message": {
                        "type": "devices_list", "microphones": [], "speakers": []
                    }})),
                    Step::Pause(Duration::from_millis(30)),
                    Step::Send(json!({"op": "complete", "id": id})),
                ]
            })
            .await;
            let result = forward_json(&bridge, &store, port, json!({"type": "list_devices"}), None).await.unwrap();
            assert_eq!(message_types(&result), ["devices_list"]);
            assert!(server.await.unwrap().client_closed);

            // An application error message is returned as a normal reply and ends the wait.
            let (port, server) = scripted_engine(|id| {
                vec![
                    Step::Send(json!({"op": "complete", "id": id})),
                    Step::Send(json!({"op": "response", "id": id, "message": {
                        "type": "error", "message": "LLM not configured."
                    }})),
                ]
            })
            .await;
            let command = json!({"type": "request_summary", "entries": []});
            let result = forward_json(&bridge, &store, port, command, None).await.unwrap();
            assert_eq!(result.messages, [json!({"type": "error", "message": "LLM not configured."})]);
            assert!(server.await.unwrap().client_closed);
        });
    }

    #[test]
    fn forward_collects_batched_summary_chunks_through_llm_done() {
        runtime().block_on(async {
            let chunk = |id: &str, n: usize| {
                Step::Send(json!({"op": "response", "id": id, "message": {"type": "llm_chunk", "text": format!("c{n} ")}}))
            };
            let (port, server) = scripted_engine(move |id| {
                let mut steps = vec![chunk(id, 0), Step::Send(json!({"op": "complete", "id": id}))];
                // More chunks than the start/stop response cap, in bursts.
                for n in 1..=200 {
                    steps.push(chunk(id, n));
                    if n % 50 == 0 {
                        steps.push(Step::Pause(Duration::from_millis(10)));
                    }
                }
                steps.push(Step::Send(json!({"op": "response", "id": id, "message": {"type": "llm_done"}})));
                steps.push(chunk(id, 999));
                steps
            })
            .await;
            let bridge = forward_bridge(Duration::from_secs(5));
            let command = json!({"type": "request_summary", "entries": [{"source": "mic", "text": "we agreed"}]});
            let result = forward_json(&bridge, &empty_store(), port, command, None).await.unwrap();

            assert_eq!(result.messages.len(), 202);
            let texts: Vec<&str> = result.messages[..201].iter().map(|m| m["text"].as_str().unwrap()).collect();
            let expected: Vec<String> = (0..=200).map(|n| format!("c{n} ")).collect();
            assert_eq!(texts, expected);
            assert_eq!(result.messages[201], json!({"type": "llm_done"}));
            assert!(server.await.unwrap().client_closed);
        });
    }

    #[test]
    fn forward_times_out_per_command_and_closes_the_socket() {
        runtime().block_on(async {
            // `complete` without the terminal `models_list`.
            let (port, server) = scripted_engine(|id| vec![Step::Send(json!({"op": "complete", "id": id}))]).await;
            let bridge = forward_bridge(Duration::from_millis(200));
            let started = Instant::now();
            let command = json!({"type": "request_models_list"});
            let err = forward_json(&bridge, &empty_store(), port, command, None).await.unwrap_err();
            assert_safe_error(&err, ErrorCode::Timeout);
            assert!(started.elapsed() >= Duration::from_millis(200));
            assert!(started.elapsed() < Duration::from_secs(3));
            let log = tokio::time::timeout(Duration::from_secs(3), server).await.unwrap().unwrap();
            assert!(log.client_closed);
        });
    }

    #[test]
    fn forward_engine_error_frames_are_command_rejected() {
        runtime().block_on(async {
            let bridge = forward_bridge(Duration::from_secs(5));
            let scripts: [fn(&str) -> Vec<Step>; 2] = [
                |id| {
                    vec![
                        Step::Send(json!({"op": "response", "id": id, "message": {"type": "status"}})),
                        Step::Send(json!({"op": "error", "code": "command_failed", "message": "Command failed", "id": id})),
                        Step::Send(json!({"op": "complete", "id": id})),
                    ]
                },
                |_| vec![Step::Send(json!({"op": "error", "code": "invalid_frame", "message": "Invalid frame"}))],
            ];
            for script in scripts {
                let (port, server) = scripted_engine(script).await;
                let command = json!({"type": "delete_model", "model_id": "tiny", "model_type": "asr"});
                let err = forward_json(&bridge, &empty_store(), port, command, None).await.unwrap_err();
                assert_safe_error(&err, ErrorCode::CommandRejected);
                assert!(server.await.unwrap().client_closed);
            }

            let (port, server) = fake_engine(Script::RejectAuth).await;
            let err = forward_json(&bridge, &empty_store(), port, json!({"type": "list_devices"}), None)
                .await
                .unwrap_err();
            assert_safe_error(&err, ErrorCode::CommandRejected);
            assert!(server.await.unwrap().client_closed);
        });
    }

    #[test]
    fn forward_requires_a_running_engine() {
        let bridge = forward_bridge(Duration::from_secs(5));
        let statuses = [
            EngineStatus { state: EngineState::Stopped, port: None },
            EngineStatus { state: EngineState::Exited, port: None },
            // Reported running, but no endpoint was captured for this run.
            EngineStatus { state: EngineState::Running, port: Some(9) },
        ];
        for status in statuses {
            assert_safe_error(&bridge.forward_endpoint(status).unwrap_err(), ErrorCode::EngineNotRunning);
        }

        runtime().block_on(async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            tokio::spawn(async move {
                if let Ok((stream, _)) = listener.accept().await {
                    drop(stream);
                }
            });
            let err = forward_json(&bridge, &empty_store(), port, json!({"type": "list_devices"}), None)
                .await
                .unwrap_err();
            assert_safe_error(&err, ErrorCode::EngineNotRunning);
        });
    }
}

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

/// Same grace period `EngineManager::production` uses; main.rs builds the manager itself so
/// it can install the capturing spawner.
pub const ENGINE_STOP_GRACE: Duration = Duration::from_secs(2);

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
        Value::Object(map) => map
            .iter()
            .any(|(key, item)| is_secret_key_name(key) || contains_secret_key(item)),
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
    if wire.error_code.as_deref().map_or(false, |code| !is_error_code(code))
        || wire.fingerprint.as_deref().map_or(false, |fp| !is_fingerprint(fp))
        || !wire.tested_at.is_finite()
        || wire.tested_at <= 0.0
        || wire.latency_ms.map_or(false, |ms| !ms.is_finite() || ms < 0.0)
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
    loop {
        let mut reply = next_frame(socket).await?;
        let matches = frame_str(&reply, "id") == Some(request_id);
        match frame_str(&reply, "op") {
            Some("response") if matches => {
                if messages.len() >= MAX_CONTROL_RESPONSES {
                    return Err(invalid_response());
                }
                messages.push(reply.remove("message").unwrap_or(Value::Null));
            }
            Some("complete") if matches => return Ok(messages),
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
) -> Result<Vec<Value>, BridgeError> {
    let mut config = WebSocketConfig::default();
    config.max_message_size = Some(MAX_CONTROL_MESSAGE_BYTES);
    config.max_frame_size = Some(MAX_CONTROL_MESSAGE_BYTES);
    let (mut socket, _) = tokio_tungstenite::connect_async_with_config(url, Some(config), true)
        .await
        .map_err(|_| BridgeError::new(ErrorCode::ControlUnavailable))?;
    let outcome = exchange(&mut socket, token, request_id, &mut frame).await;
    // Best effort; dropping the stream closes the TCP connection either way.
    let _ = socket.close(None).await;
    outcome
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
        let PreparedRequest {
            request_id,
            frame,
            mut redactions,
        } = prepared;
        let url = format!("ws://127.0.0.1:{}", endpoint.port);
        let outcome = tokio::time::timeout(
            self.control_timeout,
            control_round_trip(&url, &endpoint.token, &request_id, frame),
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
}

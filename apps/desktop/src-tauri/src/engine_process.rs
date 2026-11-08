//! Tauri-owned engine process manager. Dormant: nothing starts the engine until the bridge
//! integration task calls `engine_start`; the legacy "run the engine separately" path is
//! untouched.
//!
//! Each run gets a fresh loopback port and two independent random tokens. Bootstrap values
//! reach Python through the child's environment only, never the command line, which other
//! local users can read from the process table. The control token stays inside Rust; the
//! webview only ever sees the port and the event token.
//!
//! Lifecycle (all transitions happen under one mutex, so there is never a second child):
//!
//! ```text
//! Stopped | Exited --start ok-->   Running   (new port + tokens)
//! Stopped | Exited --start err-->  Stopped   (redacted error)
//! Running          --start-->      Running   (same run, no spawn)
//! Running          --child dies--> Exited    (observed on the next start/status)
//! Running          --stop-->       Stopped   (close stdin, grace period, then kill)
//! Stopped | Exited --stop-->       Stopped   (no-op)
//! ```
//!
//! Python does not read these tokens or watch stdin yet; both arrive with the authenticated
//! event plane. Until then a stop always ends in a kill after the grace period.

use std::fmt;
use std::io;
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde::Serialize;
use zeroize::Zeroizing;

/// 256 bits per token, hex encoded.
const TOKEN_BYTES: usize = 32;
pub const STOP_GRACE: Duration = Duration::from_secs(2);
const STOP_POLL: Duration = Duration::from_millis(25);

/// Base name of the bundled engine binary. Tauri 1 `externalBin` entries are installed next
/// to the app executable with the target-triple suffix stripped.
pub const ENGINE_SIDECAR_NAME: &str = "echoflux-engine";

/// Development override for the repository root; defaults to the checkout this crate was
/// built from.
const DEV_ROOT_ENV: &str = "ECHOFLUX_ENGINE_ROOT";

pub const ENV_HOST: &str = "ECHOFLUX_HOST";
pub const ENV_PORT: &str = "ECHOFLUX_PORT";
pub const ENV_CONTROL_TOKEN: &str = "ECHOFLUX_CONTROL_TOKEN";
pub const ENV_EVENT_TOKEN: &str = "ECHOFLUX_EVENT_TOKEN";

// ---------------------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------------------

/// One per-run bearer token. Wiped on drop and never printed by `Debug`.
pub struct Token(Zeroizing<String>);

impl Token {
    fn generate() -> Result<Self, EngineError> {
        let mut bytes = Zeroizing::new([0u8; TOKEN_BYTES]);
        getrandom::getrandom(bytes.as_mut())
            .map_err(|_| EngineError::new(ErrorCode::TokenGenerationFailed))?;
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut encoded = Zeroizing::new(String::with_capacity(TOKEN_BYTES * 2));
        for byte in bytes.iter() {
            encoded.push(HEX[usize::from(byte >> 4)] as char);
            encoded.push(HEX[usize::from(byte & 0x0f)] as char);
        }
        Ok(Self(encoded))
    }

    fn expose(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Debug for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Token(<redacted>)")
    }
}

// ---------------------------------------------------------------------------------------
// Webview-facing types
// ---------------------------------------------------------------------------------------

/// Everything the webview needs to open the event plane. There is deliberately no control
/// token field.
#[derive(Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineConnectionInfo {
    pub port: u16,
    pub event_token: String,
}

impl fmt::Debug for EngineConnectionInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EngineConnectionInfo")
            .field("port", &self.port)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineState {
    Stopped,
    Running,
    /// The child ended without a stop request. Its port and tokens are already discarded.
    Exited,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct EngineStatus {
    pub state: EngineState,
    pub port: Option<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    EngineNotFound,
    SpawnFailed,
    PortUnavailable,
    TokenGenerationFailed,
    LaunchUnresolved,
    StopFailed,
}

impl ErrorCode {
    fn message(self) -> &'static str {
        match self {
            Self::EngineNotFound => "The engine executable could not be found.",
            Self::SpawnFailed => "The engine process could not be started.",
            Self::PortUnavailable => "No free loopback port could be allocated for the engine.",
            Self::TokenGenerationFailed => "Secure engine tokens could not be generated.",
            Self::LaunchUnresolved => "The engine location could not be determined.",
            Self::StopFailed => "The engine process could not be stopped.",
        }
    }
}

/// Webview-facing error. Holds no dynamic text: OS spawn errors can carry the executable
/// path, and nothing here may ever echo a token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EngineError {
    pub code: ErrorCode,
    pub message: &'static str,
}

impl EngineError {
    fn new(code: ErrorCode) -> Self {
        Self {
            code,
            message: code.message(),
        }
    }

    fn from_spawn(err: io::Error) -> Self {
        match err.kind() {
            io::ErrorKind::NotFound | io::ErrorKind::PermissionDenied => {
                Self::new(ErrorCode::EngineNotFound)
            }
            _ => Self::new(ErrorCode::SpawnFailed),
        }
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message)
    }
}

impl std::error::Error for EngineError {}

// ---------------------------------------------------------------------------------------
// Launch resolution
// ---------------------------------------------------------------------------------------

/// Where the engine lives and how to invoke it, before any per-run values are added.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchTarget {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub working_dir: PathBuf,
}

/// Development: the repo's virtualenv interpreter running `-m engine.main`.
pub fn development_target(repo_root: &Path) -> LaunchTarget {
    let python = if cfg!(windows) {
        repo_root.join(".venv").join("Scripts").join("python.exe")
    } else {
        repo_root.join(".venv").join("bin").join("python")
    };
    LaunchTarget {
        program: python,
        args: vec!["-m".to_owned(), "engine.main".to_owned()],
        working_dir: repo_root.to_path_buf(),
    }
}

/// Packaged: the frozen engine sidecar installed beside the app executable.
pub fn packaged_target(exe_dir: &Path) -> LaunchTarget {
    let name = format!("{}{}", ENGINE_SIDECAR_NAME, std::env::consts::EXE_SUFFIX);
    LaunchTarget {
        program: exe_dir.join(name),
        args: Vec::new(),
        working_dir: exe_dir.to_path_buf(),
    }
}

/// Debug builds run the engine from source; release builds expect the bundled sidecar.
/// Resolved per start so a missing interpreter or sidecar never affects app startup.
pub fn default_launch_target() -> Result<LaunchTarget, EngineError> {
    if cfg!(debug_assertions) {
        let root = match std::env::var_os(DEV_ROOT_ENV) {
            Some(root) => PathBuf::from(root),
            // src-tauri -> desktop -> apps -> repository root
            None => Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join(".."),
        };
        Ok(development_target(&root))
    } else {
        let exe = std::env::current_exe()
            .map_err(|_| EngineError::new(ErrorCode::LaunchUnresolved))?;
        let dir = exe
            .parent()
            .ok_or_else(|| EngineError::new(ErrorCode::LaunchUnresolved))?;
        Ok(packaged_target(dir))
    }
}

/// Full spawn request for one run. `Debug` hides the environment because it holds tokens.
pub struct LaunchSpec {
    pub target: LaunchTarget,
    pub env: Vec<(&'static str, Zeroizing<String>)>,
}

impl fmt::Debug for LaunchSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let keys: Vec<&str> = self.env.iter().map(|(key, _)| *key).collect();
        f.debug_struct("LaunchSpec")
            .field("target", &self.target)
            .field("env_keys", &keys)
            .finish()
    }
}

/// Binds `127.0.0.1:0` to have the OS pick a free port, then releases it for the engine.
/// Another process could take the port in between; the engine then fails to bind and the
/// run shows up as `Exited`, and the next start allocates a new port.
pub fn allocate_loopback_port() -> Result<u16, EngineError> {
    let unavailable = |_| EngineError::new(ErrorCode::PortUnavailable);
    let listener =
        TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)).map_err(unavailable)?;
    let port = listener.local_addr().map_err(unavailable)?.port();
    if port == 0 {
        return Err(EngineError::new(ErrorCode::PortUnavailable));
    }
    Ok(port)
}

// ---------------------------------------------------------------------------------------
// Process abstraction
// ---------------------------------------------------------------------------------------

pub trait EngineChild: Send {
    /// Returns true once the process has exited.
    fn has_exited(&mut self) -> io::Result<bool>;
    /// Asks the process to exit on its own. The engine will treat stdin EOF as shutdown.
    fn request_stop(&mut self);
    /// Forcefully terminates and reaps the process.
    fn kill(&mut self) -> io::Result<()>;
}

pub trait EngineSpawner: Send + Sync {
    fn spawn(&self, spec: &LaunchSpec) -> io::Result<Box<dyn EngineChild>>;
}

pub struct OsSpawner;

impl EngineSpawner for OsSpawner {
    fn spawn(&self, spec: &LaunchSpec) -> io::Result<Box<dyn EngineChild>> {
        let mut command = Command::new(&spec.target.program);
        command
            .args(&spec.target.args)
            .current_dir(&spec.target.working_dir)
            .stdin(Stdio::piped());
        for (key, value) in &spec.env {
            command.env(key, value.as_str());
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        Ok(Box::new(OsChild(command.spawn()?)))
    }
}

struct OsChild(Child);

impl EngineChild for OsChild {
    fn has_exited(&mut self) -> io::Result<bool> {
        Ok(self.0.try_wait()?.is_some())
    }

    fn request_stop(&mut self) {
        drop(self.0.stdin.take());
    }

    fn kill(&mut self) -> io::Result<()> {
        self.0.kill()?;
        self.0.wait().map(|_| ())
    }
}

// ---------------------------------------------------------------------------------------
// Manager
// ---------------------------------------------------------------------------------------

struct RunningEngine {
    child: Box<dyn EngineChild>,
    port: u16,
    event_token: Token,
    // Held for the future privileged control bridge; never serialised.
    #[allow(dead_code)]
    control_token: Token,
}

impl RunningEngine {
    fn connection_info(&self) -> EngineConnectionInfo {
        EngineConnectionInfo {
            port: self.port,
            event_token: self.event_token.expose().to_owned(),
        }
    }
}

enum Slot {
    Stopped,
    Running(RunningEngine),
    Exited,
}

type TargetResolver = Box<dyn Fn() -> Result<LaunchTarget, EngineError> + Send + Sync>;

pub struct EngineManager {
    spawner: Box<dyn EngineSpawner>,
    resolve_target: TargetResolver,
    stop_grace: Duration,
    slot: Mutex<Slot>,
}

impl EngineManager {
    pub fn new(
        spawner: Box<dyn EngineSpawner>,
        resolve_target: TargetResolver,
        stop_grace: Duration,
    ) -> Self {
        Self {
            spawner,
            resolve_target,
            stop_grace,
            slot: Mutex::new(Slot::Stopped),
        }
    }

    #[allow(dead_code)]
    pub fn production() -> Self {
        Self::new(Box::new(OsSpawner), Box::new(default_launch_target), STOP_GRACE)
    }

    fn lock(&self) -> MutexGuard<'_, Slot> {
        // Every mutation leaves the slot in a valid state before anything can panic.
        self.slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Turns a running slot whose child has died into `Exited`.
    fn refresh(slot: &mut Slot) {
        if let Slot::Running(run) = slot {
            // An error here means the handle is unusable; treat the run as gone.
            if run.child.has_exited().unwrap_or(true) {
                *slot = Slot::Exited;
            }
        }
    }

    /// Idempotent: a live run is returned as-is instead of spawning a second engine.
    pub fn start(&self) -> Result<EngineConnectionInfo, EngineError> {
        let mut slot = self.lock();
        Self::refresh(&mut slot);
        if let Slot::Running(run) = &*slot {
            return Ok(run.connection_info());
        }

        let target = (self.resolve_target)()?;
        let port = allocate_loopback_port()?;
        let control_token = Token::generate()?;
        let event_token = Token::generate()?;
        let spec = LaunchSpec {
            target,
            env: vec![
                (ENV_HOST, Zeroizing::new(Ipv4Addr::LOCALHOST.to_string())),
                (ENV_PORT, Zeroizing::new(port.to_string())),
                (ENV_CONTROL_TOKEN, Zeroizing::new(control_token.expose().to_owned())),
                (ENV_EVENT_TOKEN, Zeroizing::new(event_token.expose().to_owned())),
            ],
        };

        let child = match self.spawner.spawn(&spec) {
            Ok(child) => child,
            Err(err) => {
                *slot = Slot::Stopped;
                return Err(EngineError::from_spawn(err));
            }
        };
        let run = RunningEngine {
            child,
            port,
            event_token,
            control_token,
        };
        let info = run.connection_info();
        *slot = Slot::Running(run);
        Ok(info)
    }

    /// Idempotent: stopping when nothing runs succeeds.
    pub fn stop(&self) -> Result<EngineStatus, EngineError> {
        let mut slot = self.lock();
        let mut run = match std::mem::replace(&mut *slot, Slot::Stopped) {
            Slot::Running(run) => run,
            Slot::Stopped | Slot::Exited => return Ok(Self::status_of(&slot)),
        };

        run.child.request_stop();
        let deadline = Instant::now() + self.stop_grace;
        loop {
            if run.child.has_exited().unwrap_or(false) {
                return Ok(Self::status_of(&slot));
            }
            if Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(STOP_POLL);
        }

        if run.child.kill().is_err() && !run.child.has_exited().unwrap_or(false) {
            // Keep ownership so a retry can still reach the process and no duplicate starts.
            *slot = Slot::Running(run);
            return Err(EngineError::new(ErrorCode::StopFailed));
        }
        Ok(Self::status_of(&slot))
    }

    pub fn status(&self) -> EngineStatus {
        let mut slot = self.lock();
        Self::refresh(&mut slot);
        Self::status_of(&slot)
    }

    fn status_of(slot: &Slot) -> EngineStatus {
        match slot {
            Slot::Stopped => EngineStatus { state: EngineState::Stopped, port: None },
            Slot::Exited => EngineStatus { state: EngineState::Exited, port: None },
            Slot::Running(run) => EngineStatus {
                state: EngineState::Running,
                port: Some(run.port),
            },
        }
    }

    /// Called on app exit. Errors are ignored: there is nobody left to report them to.
    pub fn shutdown(&self) {
        let _ = self.stop();
    }
}

impl Drop for EngineManager {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub type SharedEngineManager = Arc<EngineManager>;

/// Stop can wait out the grace period, so lifecycle work stays off the async workers.
async fn run_blocking<T, F>(job: F) -> Result<T, EngineError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, EngineError> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(job)
        .await
        .unwrap_or_else(|_| Err(EngineError::new(ErrorCode::SpawnFailed)))
}

// Webview commands. None returns the control token.

#[tauri::command]
pub async fn engine_start(
    manager: tauri::State<'_, SharedEngineManager>,
) -> Result<EngineConnectionInfo, EngineError> {
    let manager = Arc::clone(&manager);
    run_blocking(move || manager.start()).await
}

#[tauri::command]
pub async fn engine_stop(
    manager: tauri::State<'_, SharedEngineManager>,
) -> Result<EngineStatus, EngineError> {
    let manager = Arc::clone(&manager);
    run_blocking(move || manager.stop()).await
}

#[tauri::command]
pub async fn engine_status(
    manager: tauri::State<'_, SharedEngineManager>,
) -> Result<EngineStatus, EngineError> {
    let manager = Arc::clone(&manager);
    run_blocking(move || Ok(manager.status())).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Shared view of one fake process so tests can inspect it after the manager owns it.
    #[derive(Default)]
    struct FakeProcess {
        exited: AtomicBool,
        exit_on_stop_request: bool,
        refuse_kill: bool,
        stop_requests: AtomicUsize,
        kills: AtomicUsize,
    }

    struct FakeChild(Arc<FakeProcess>);

    impl EngineChild for FakeChild {
        fn has_exited(&mut self) -> io::Result<bool> {
            Ok(self.0.exited.load(Ordering::SeqCst))
        }

        fn request_stop(&mut self) {
            self.0.stop_requests.fetch_add(1, Ordering::SeqCst);
            if self.0.exit_on_stop_request {
                self.0.exited.store(true, Ordering::SeqCst);
            }
        }

        fn kill(&mut self) -> io::Result<()> {
            self.0.kills.fetch_add(1, Ordering::SeqCst);
            if self.0.refuse_kill {
                return Err(io::Error::new(io::ErrorKind::Other, "access denied"));
            }
            self.0.exited.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    struct RecordedSpawn {
        args: Vec<String>,
        env: Vec<(&'static str, String)>,
        process: Arc<FakeProcess>,
    }

    /// Never starts a real process. `fail_first` spawns fail with a message that contains a
    /// path and the would-be token, to prove neither reaches the error.
    #[derive(Default)]
    struct FakeSpawner {
        fail_first: usize,
        exit_on_stop_request: bool,
        refuse_kill: bool,
        attempts: AtomicUsize,
        spawns: Mutex<Vec<RecordedSpawn>>,
    }

    impl EngineSpawner for Arc<FakeSpawner> {
        fn spawn(&self, spec: &LaunchSpec) -> io::Result<Box<dyn EngineChild>> {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
            let env: Vec<(&'static str, String)> =
                spec.env.iter().map(|(k, v)| (*k, v.as_str().to_owned())).collect();
            if attempt < self.fail_first {
                let token = env.iter().find(|(k, _)| *k == ENV_CONTROL_TOKEN).unwrap().1.clone();
                let detail = format!("C:\\Users\\someone\\engine.exe not found ({token})");
                return Err(io::Error::new(io::ErrorKind::NotFound, detail));
            }
            let process = Arc::new(FakeProcess {
                exit_on_stop_request: self.exit_on_stop_request,
                refuse_kill: self.refuse_kill,
                ..FakeProcess::default()
            });
            self.spawns.lock().unwrap().push(RecordedSpawn {
                args: spec.target.args.clone(),
                env,
                process: Arc::clone(&process),
            });
            Ok(Box::new(FakeChild(process)))
        }
    }

    impl FakeSpawner {
        fn env(&self, index: usize, key: &str) -> String {
            let spawns = self.spawns.lock().unwrap();
            spawns[index].env.iter().find(|(k, _)| *k == key).unwrap().1.clone()
        }

        fn process(&self, index: usize) -> Arc<FakeProcess> {
            Arc::clone(&self.spawns.lock().unwrap()[index].process)
        }

        fn spawn_count(&self) -> usize {
            self.spawns.lock().unwrap().len()
        }
    }

    fn fake_target() -> LaunchTarget {
        development_target(Path::new("repo-root"))
    }

    fn manager_with(spawner: FakeSpawner) -> (EngineManager, Arc<FakeSpawner>) {
        let spawner = Arc::new(spawner);
        let manager = EngineManager::new(
            Box::new(Arc::clone(&spawner)),
            Box::new(|| Ok(fake_target())),
            Duration::from_millis(50),
        );
        (manager, spawner)
    }

    fn is_token(value: &str) -> bool {
        value.len() == TOKEN_BYTES * 2 && value.bytes().all(|b| b.is_ascii_hexdigit())
    }

    #[test]
    fn tokens_differ_across_starts_and_from_each_other() {
        let (manager, spawner) = manager_with(FakeSpawner {
            exit_on_stop_request: true,
            ..FakeSpawner::default()
        });
        let first = manager.start().unwrap();
        manager.stop().unwrap();
        let second = manager.start().unwrap();

        let control = [spawner.env(0, ENV_CONTROL_TOKEN), spawner.env(1, ENV_CONTROL_TOKEN)];
        let event = [spawner.env(0, ENV_EVENT_TOKEN), spawner.env(1, ENV_EVENT_TOKEN)];
        for token in control.iter().chain(event.iter()) {
            assert!(is_token(token), "token is not 64 hex chars");
        }
        assert_ne!(control[0], event[0]);
        assert_ne!(control[1], event[1]);
        assert_ne!(control[0], control[1]);
        assert_ne!(event[0], event[1]);
        assert_eq!(first.event_token, event[0]);
        assert_eq!(second.event_token, event[1]);
    }

    #[test]
    fn generated_tokens_are_unique() {
        let tokens: std::collections::HashSet<String> =
            (0..64).map(|_| Token::generate().unwrap().expose().to_owned()).collect();
        assert_eq!(tokens.len(), 64);
    }

    #[test]
    fn webview_structures_omit_control_token() {
        let (manager, spawner) = manager_with(FakeSpawner::default());
        let info = manager.start().unwrap();
        let control = spawner.env(0, ENV_CONTROL_TOKEN);

        let json = serde_json::to_value(&info).unwrap();
        let keys: Vec<&str> = json.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"port") && keys.contains(&"eventToken"));

        let status = serde_json::to_string(&manager.status()).unwrap();
        let rendered = format!("{json} {info:?} {status} {:?}", manager.status());
        assert!(!rendered.contains(&control));
        // Debug output is for logs, so it hides the event token too.
        assert!(!format!("{info:?}").contains(&info.event_token));
    }

    #[test]
    fn bootstrap_travels_in_environment_not_arguments() {
        let (manager, spawner) = manager_with(FakeSpawner::default());
        let info = manager.start().unwrap();
        assert_eq!(spawner.env(0, ENV_HOST), "127.0.0.1");
        assert_eq!(spawner.env(0, ENV_PORT), info.port.to_string());

        let args = spawner.spawns.lock().unwrap()[0].args.join(" ");
        assert_eq!(args, "-m engine.main");
        assert!(!args.contains(&spawner.env(0, ENV_CONTROL_TOKEN)));
        assert!(!args.contains(&info.event_token));
    }

    #[test]
    fn duplicate_start_and_stop_are_idempotent() {
        let (manager, spawner) = manager_with(FakeSpawner::default());
        assert_eq!(manager.stop().unwrap().state, EngineState::Stopped);

        let first = manager.start().unwrap();
        let again = manager.start().unwrap();
        assert_eq!(first, again);
        assert_eq!(spawner.spawn_count(), 1);
        assert_eq!(
            manager.status(),
            EngineStatus { state: EngineState::Running, port: Some(first.port) }
        );

        assert_eq!(manager.stop().unwrap().state, EngineState::Stopped);
        assert_eq!(manager.stop().unwrap().state, EngineState::Stopped);
        let process = spawner.process(0);
        assert_eq!(process.stop_requests.load(Ordering::SeqCst), 1);
        assert_eq!(process.kills.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn graceful_exit_skips_kill() {
        let (manager, spawner) = manager_with(FakeSpawner {
            exit_on_stop_request: true,
            ..FakeSpawner::default()
        });
        manager.start().unwrap();
        manager.stop().unwrap();
        assert_eq!(spawner.process(0).kills.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn failed_kill_keeps_ownership() {
        let (manager, spawner) = manager_with(FakeSpawner {
            refuse_kill: true,
            ..FakeSpawner::default()
        });
        manager.start().unwrap();
        assert_eq!(manager.stop().unwrap_err().code, ErrorCode::StopFailed);
        assert_eq!(manager.status().state, EngineState::Running);
        manager.start().unwrap();
        assert_eq!(spawner.spawn_count(), 1);
    }

    #[test]
    fn spawn_failure_is_safe_and_redacted() {
        let (manager, spawner) = manager_with(FakeSpawner {
            fail_first: 1,
            ..FakeSpawner::default()
        });
        let err = manager.start().unwrap_err();
        assert_eq!(err.code, ErrorCode::EngineNotFound);
        let rendered = format!("{} {err} {err:?}", serde_json::to_string(&err).unwrap());
        for forbidden in ["someone", "engine.exe", "repo-root", "not found ("] {
            assert!(!rendered.contains(forbidden), "error leaked {forbidden:?}: {rendered}");
        }
        // No token-length hex run survives into the error.
        let longest_hex_run = rendered
            .split(|c: char| !c.is_ascii_hexdigit())
            .map(str::len)
            .max()
            .unwrap_or(0);
        assert!(longest_hex_run < TOKEN_BYTES, "error may contain a token: {rendered}");
        assert_eq!(manager.status(), EngineStatus { state: EngineState::Stopped, port: None });

        // A later start works normally and gets its own tokens.
        manager.start().unwrap();
        assert_eq!(spawner.spawn_count(), 1);
    }

    #[test]
    fn unexpected_exit_is_reported_and_restartable() {
        let (manager, spawner) = manager_with(FakeSpawner::default());
        let first = manager.start().unwrap();
        spawner.process(0).exited.store(true, Ordering::SeqCst);
        assert_eq!(manager.status(), EngineStatus { state: EngineState::Exited, port: None });

        let second = manager.start().unwrap();
        assert_eq!(spawner.spawn_count(), 2);
        assert_ne!(first.event_token, second.event_token);
        assert_eq!(manager.status().state, EngineState::Running);
    }

    #[test]
    fn dropping_manager_terminates_child() {
        let (manager, spawner) = manager_with(FakeSpawner::default());
        manager.start().unwrap();
        drop(manager);
        assert!(spawner.process(0).exited.load(Ordering::SeqCst));
    }

    #[test]
    fn resolver_failure_does_not_spawn() {
        let spawner = Arc::new(FakeSpawner::default());
        let manager = EngineManager::new(
            Box::new(Arc::clone(&spawner)),
            Box::new(|| Err(EngineError::new(ErrorCode::LaunchUnresolved))),
            Duration::from_millis(10),
        );
        assert_eq!(manager.start().unwrap_err().code, ErrorCode::LaunchUnresolved);
        assert_eq!(spawner.attempts.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn allocated_port_is_free_loopback_port() {
        let port = allocate_loopback_port().unwrap();
        assert_ne!(port, 0);
        TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port)).unwrap();
    }

    #[test]
    fn development_and_packaged_targets() {
        let dev = development_target(Path::new("checkout"));
        assert!(dev.program.starts_with(Path::new("checkout").join(".venv")));
        assert_eq!(dev.args, ["-m", "engine.main"]);
        assert_eq!(dev.working_dir, Path::new("checkout"));

        let packaged = packaged_target(Path::new("install"));
        let expected = format!("{}{}", ENGINE_SIDECAR_NAME, std::env::consts::EXE_SUFFIX);
        assert_eq!(packaged.program, Path::new("install").join(expected));
        assert!(packaged.args.is_empty());
    }

    #[test]
    fn launch_spec_debug_hides_tokens() {
        let spec = LaunchSpec {
            target: fake_target(),
            env: vec![(ENV_CONTROL_TOKEN, Zeroizing::new("f".repeat(64)))],
        };
        let rendered = format!("{spec:?} {:?}", Token::generate().unwrap());
        assert!(rendered.contains(ENV_CONTROL_TOKEN));
        assert!(!rendered.contains(&"f".repeat(64)));
    }
}

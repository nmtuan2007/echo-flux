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
    fn validator_target_uses_the_dev_interpreter_and_sidecar_flag_when_packaged() {
        let dev = validator_target(development_target(Path::new("checkout"))).unwrap();
        assert_eq!(dev.args, ["-m", VALIDATOR_MODULE]);
        assert!(dev.program.starts_with(Path::new("checkout").join(".venv")));
        assert_eq!(dev.working_dir, Path::new("checkout"));

        let packaged = validator_target(packaged_target(Path::new("install"))).unwrap();
        assert_eq!(packaged.args, ["--validate-provider"]);
        assert_eq!(packaged.working_dir, Path::new("install"));
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
            ("list_devices", Some("devices_list"), 10, false, false),
            ("request_models_list", Some("models_list"), 10, false, false),
            ("download_model", None, 20, true, false),
            ("delete_model", Some("model_action_result"), 30, false, false),
            ("search_hub", Some("hub_search_results"), 30, true, false),
            ("request_suggestion", Some("suggestion_result"), 120, false, false),
            ("request_assist", None, 120, false, false),
            ("configure_assistant", None, 15, false, true),
            ("request_summary", Some("llm_done"), 300, false, false),
            ("test_connection", Some("test_connection_result"), 30, false, false),
            ("profile_create", None, 10, false, false),
            ("profile_list", None, 10, false, false),
            ("profile_get", None, 10, false, false),
            ("profile_update", None, 10, false, false),
            ("profile_delete", None, 10, false, false),
            ("profile_attach", None, 10, false, false),
            ("conversation_create", None, 10, false, false),
            ("conversation_start", None, 10, false, false),
            ("conversation_list", None, 10, false, false),
            ("conversation_get", None, 15, false, false),
            ("conversation_rename", None, 10, false, false),
            ("conversation_delete", None, 10, false, false),
            ("conversation_export", None, 30, false, false),
            ("conversation_import", None, 30, false, false),
            ("message_update", None, 10, false, false),
            ("pack_status", None, 10, false, false),
            ("pack_install", None, 600, false, false),
            ("pack_cancel", None, 10, false, false),
            ("pack_retry", None, 600, false, false),
            ("pack_check_readiness", None, 15, false, false),
            ("get_provider_manifests", None, 10, false, false),
            ("audio_probe", None, 10, false, false),
            ("summary_cancel", None, 10, false, false),
        ];
        for (kind, terminal, seconds, accepts_hf_token, accepts_assistant_credentials) in table {
            let checked = check_forward_command(json!({"type": kind}), None).unwrap();
            let expected = ForwardPolicy {
                terminal,
                timeout: Duration::from_secs(seconds),
                accepts_hf_token,
                accepts_assistant_credentials,
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

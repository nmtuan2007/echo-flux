#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use tauri::Manager;

#[tauri::command]
fn get_engine_url(port: Option<u16>) -> String {
    let p = port.unwrap_or(8765);
    format!("ws://127.0.0.1:{}", p)
}

#[tauri::command]
fn create_overlay_window(app: tauri::AppHandle) -> Result<(), String> {
    let existing = app.get_window("overlay");
    if existing.is_some() {
        return Ok(());
    }

    tauri::WindowBuilder::new(&app, "overlay", tauri::WindowUrl::App("index.html".into()))
        .title("EchoFlux Overlay")
        .inner_size(600.0, 200.0)
        .always_on_top(true)
        .decorations(false)
        .transparent(true)
        .skip_taskbar(true)
        .resizable(true)
        .build()
        .map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
fn close_overlay_window(app: tauri::AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_window("overlay") {
        window.close().map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            get_engine_url,
            create_overlay_window,
            close_overlay_window,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

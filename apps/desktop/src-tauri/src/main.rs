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
async fn create_overlay_window(app: tauri::AppHandle) -> Result<(), String> {
    println!("[RUST DEBUG] Executing create_overlay_window...");
    let existing = app.get_window("overlay");
    if existing.is_some() {
        println!("[RUST DEBUG] Window 'overlay' already exists.");
        return Ok(());
    }

    println!("[RUST DEBUG] WindowBuilder is creating 'overlay' window...");

    // Remove strict transparent(true) if it's causing GPU deadlocks on some Windows machines, but try async first.
    tauri::WindowBuilder::new(&app, "overlay", tauri::WindowUrl::App("index.html".into()))
        .title("EchoFlux Overlay")
        .inner_size(600.0, 200.0)
        .always_on_top(true)
        .decorations(false)
        .transparent(true)
        .skip_taskbar(true)
        .resizable(true)
        .build()
        .map_err(|e| {
            println!("[RUST DEBUG] WindowBuilder error: {:?}", e);
            e.to_string()
        })?;

    println!("[RUST DEBUG] Overlay window created successfully.");
    Ok(())
}

#[tauri::command]
fn close_overlay_window(app: tauri::AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_window("overlay") {
        window.close().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
async fn revert_to_main_window(app: tauri::AppHandle) -> Result<(), String> {
    if let Some(main) = app.get_window("main") {
        main.show().map_err(|e| e.to_string())?;
    }
    if let Some(overlay) = app.get_window("overlay") {
        overlay.close().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
async fn toggle_overlay_window(app: tauri::AppHandle) -> Result<(), String> {
    println!("[RUST DEBUG] Executing toggle_overlay_window...");
    if let Some(window) = app.get_window("overlay") {
        println!("[RUST DEBUG] Found existing overly window, closing it...");
        window.close().map_err(|e| {
            println!("[RUST DEBUG] Error closing window: {:?}", e);
            e.to_string()
        })?;
        if let Some(main) = app.get_window("main") {
            main.show().unwrap_or(());
        }
        Ok(())
    } else {
        println!("[RUST DEBUG] Overlay window not found, calling create_overlay_window...");
        if let Some(main) = app.get_window("main") {
            main.hide().unwrap_or(());
        }
        create_overlay_window(app).await
    }
}

#[tauri::command]
fn set_stealth_mode(window: tauri::Window, enabled: bool) {
    #[cfg(target_os = "windows")]
    {
        if let Ok(hwnd) = window.hwnd() {
            let affinity = if enabled { 0x00000011 } else { 0x00000000 };
            unsafe {
                #[link(name = "user32")]
                extern "system" {
                    fn SetWindowDisplayAffinity(hwnd: isize, dwAffinity: u32) -> i32;
                }
                SetWindowDisplayAffinity(hwnd.0 as isize, affinity);
            }
        }
    }
}

#[tauri::command]
fn set_ignore_cursor_events(window: tauri::Window, ignore: bool) -> Result<(), String> {
    window.set_ignore_cursor_events(ignore).map_err(|e| e.to_string())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            get_engine_url,
            create_overlay_window,
            close_overlay_window,
            toggle_overlay_window,
            revert_to_main_window,
            set_stealth_mode,
            set_ignore_cursor_events,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

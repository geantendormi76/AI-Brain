// In: frontend/instant_assistant/src-tauri/src/main.rs
// (最终的、修正了工作目录问题的完整代码)

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{path::BaseDirectory, AppHandle, Manager, RunEvent};
use tauri_plugin_shell::{process::CommandChild, ShellExt};
use std::sync::Mutex;
use anyhow::Result;
use std::path::PathBuf;

struct AppState(Mutex<Vec<CommandChild>>);

fn spawn_and_log_sidecar(
    app_handle: &AppHandle,
    sidecar_name: &str,
    args: Vec<&str>,
    current_dir: Option<PathBuf>,
) -> Result<CommandChild> {
    println!("[Tauri-Launcher] Attempting to spawn: {} with args: {:?}", sidecar_name, args);
    
    let mut command = app_handle.shell().sidecar(sidecar_name)?;
    command = command.args(&args);
    
    if let Some(dir) = current_dir {
        println!("[Tauri-Launcher] Setting CWD for {} to: {:?}", sidecar_name, dir);
        command = command.current_dir(dir);
    }
    
    let (mut rx, child) = command.spawn()?;
    
    println!("[Tauri-Launcher] Spawned process for: {} with PID: {}", sidecar_name, child.pid());

    let sidecar_name_clone = sidecar_name.to_string();

    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                tauri_plugin_shell::process::CommandEvent::Stdout(line) => {
                    println!("[{}-stdout] {}", sidecar_name_clone, String::from_utf8_lossy(&line));
                }
                tauri_plugin_shell::process::CommandEvent::Stderr(line) => {
                    eprintln!("[{}-stderr] {}", sidecar_name_clone, String::from_utf8_lossy(&line));
                }
                tauri_plugin_shell::process::CommandEvent::Terminated(payload) => {
                     eprintln!("[{}-terminated] Exited with code: {:?}, signal: {:?}", sidecar_name_clone, payload.code, payload.signal);
                }
                _ => {}
            }
        }
    });

    Ok(child)
}

fn main() {
    let app_state = AppState(Mutex::new(Vec::new()));

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(app_state)
        .setup(|app| {
            println!("[Tauri] Setting up application and starting sidecars...");
            
            let get_model_path = |model_name: &str| -> Result<String> {
                let path = app.path().resolve(format!("models/{}", model_name), BaseDirectory::Resource)?;
                Ok(dunce::canonicalize(&path)?.to_string_lossy().into_owned())
            };
            
            let qdrant_data_dir = app.path()
                .app_data_dir()?
                .join("zhzAI_data/qdrant_storage");
            
            if !qdrant_data_dir.exists() {
                std::fs::create_dir_all(&qdrant_data_dir)?;
            }
            
            // 关键修正1：获取 services 目录本身的路径
            let services_dir = app.path().resolve("services", BaseDirectory::Resource)?;
            
            let app_handle = app.handle();

            let server_child = spawn_and_log_sidecar(&app_handle, "server", vec![], Some(services_dir.clone()))?;
            app_handle.state::<AppState>().0.lock().unwrap().push(server_child);

            let qdrant_child = spawn_and_log_sidecar(&app_handle, "qdrant", vec![], Some(qdrant_data_dir))?;
            app_handle.state::<AppState>().0.lock().unwrap().push(qdrant_child);
            
            // 关键修正2：为所有 llama-* 实例设置正确的工作目录
            
            let embedding_model_path = get_model_path("bge-small-zh-v1.5-q8_0.gguf")?;
            let embedding_args = vec!["-m", &embedding_model_path, "--embedding", "--port", "8181", "-c", "2048"];
            let embedding_child = spawn_and_log_sidecar(&app_handle, "llama-embedding", embedding_args, Some(services_dir.clone()))?;
            app_handle.state::<AppState>().0.lock().unwrap().push(embedding_child);
            
            let reranker_model_path = get_model_path("bge-reranker-v2-m3-Q4_K_M.gguf")?;
            let reranker_args = vec!["-m", &reranker_model_path, "--port", "8080", "--ctx-size", "8192", "--rerank"];
            let reranker_child = spawn_and_log_sidecar(&app_handle, "llama-rerank", reranker_args, Some(services_dir.clone()))?;
            app_handle.state::<AppState>().0.lock().unwrap().push(reranker_child);

            let chat_model_path = get_model_path("qwen-3-0.6b-instruct-Q8_0.gguf")?;
            let chat_args = vec!["-m", &chat_model_path, "--port", "8282", "-c", "4096", "-ngl", "35"];
            let chat_child = spawn_and_log_sidecar(&app_handle, "llama-chat", chat_args, Some(services_dir.clone()))?;
            app_handle.state::<AppState>().0.lock().unwrap().push(chat_child);

            println!("[Tauri] All sidecar launch commands have been issued.");
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let RunEvent::ExitRequested { .. } = event {
                println!("[Tauri] Exit requested. Killing all sidecar processes...");
                let children_to_kill = std::mem::take(&mut *app_handle.state::<AppState>().0.lock().unwrap());
                for child in children_to_kill {
                    let pid = child.pid();
                    if let Err(e) = child.kill() {
                        eprintln!("[Tauri] Failed to kill child process with PID {:?}: {}", pid, e);
                    }
                }
                println!("[Tauri] All sidecar processes terminated.");
            }
        });
}
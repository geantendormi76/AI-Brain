// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{Manager, RunEvent};
use tauri_plugin_shell::ShellExt;
use std::sync::Mutex;
use tauri_plugin_shell::process::CommandChild;

struct AppState(Mutex<Vec<CommandChild>>);

fn main() {
    let app_state = AppState(Mutex::new(Vec::new()));

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(app_state)
        .setup(|app| {
            println!("[Tauri] Setting up application and starting sidecars...");
            let shell = app.shell();
            let app_handle = app.handle();
            let path_resolver = app_handle.path();

            let resource_dir = path_resolver.resource_dir()
                .expect("Failed to get resource directory.");

            let get_model_path = |model_name: &str| {
                resource_dir.join("models").join(model_name)
                    .to_string_lossy()
                    .to_string()
            };

            // 【修复 E0716】: 不再长时间持有锁。
            // 每次需要添加 child 时，才获取一次锁，并在语句结束时自动释放。
            let app_state = app.state::<AppState>();

            let (_, server_child) = shell.sidecar("server")?.spawn()?;
            app_state.0.lock().unwrap().push(server_child);
            println!("[Tauri] Main backend API server sidecar started.");

            let (_, qdrant_child) = shell.sidecar("qdrant")?.spawn()?;
            app_state.0.lock().unwrap().push(qdrant_child);
            println!("[Tauri] Qdrant sidecar started.");

            let embedding_model = get_model_path("bge-small-zh-v1.5-q8_0.gguf");
            let (_, embedding_child) = shell.sidecar("llama_embedding")?
                .args(["-m", &embedding_model, "--embedding", "--port", "8181", "-c", "2048"])
                .spawn()?;
            app_state.0.lock().unwrap().push(embedding_child);
            println!("[Tauri] Embedding server sidecar started.");
            
            let reranker_model = get_model_path("bge-reranker-v2-m3-Q4_K_M.gguf");
            let (_, reranker_child) = shell.sidecar("llama_rerank")?
                .args(["-m", &reranker_model, "--port", "8080", "--ctx-size", "8192", "--rerank"])
                .spawn()?;
            app_state.0.lock().unwrap().push(reranker_child);
            println!("[Tauri] Reranker server sidecar started.");

            let chat_model = get_model_path("qwen-3-0.6b-instruct-Q8_0.gguf");
            let (_, chat_child) = shell.sidecar("llama_chat")?
                .args(["-m", &chat_model, "--port", "8282", "-c", "4096", "-ngl", "35", "--chat-template", "qwen"])
                .spawn()?;
            app_state.0.lock().unwrap().push(chat_child);
            println!("[Tauri] Chat LLM server sidecar started.");

            println!("[Tauri] All sidecars have been launched.");
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let RunEvent::ExitRequested { .. } = event {
                println!("[Tauri] Exit requested. Killing all sidecar processes...");
                
                // 【修复 E0507】: 使用 `std::mem::take` 来获取 `Vec` 的所有权。
                // 这样我们就可以安全地消耗 (consume) 其中的 `CommandChild`。
                let children_to_kill = std::mem::take(&mut *app_handle.state::<AppState>().0.lock().unwrap());

                for child in children_to_kill {
                    if let Err(e) = child.kill() {
                        eprintln!("[Tauri] Failed to kill child process: {}", e);
                    }
                }
                
                println!("[Tauri] All sidecar processes terminated.");
            }
        });

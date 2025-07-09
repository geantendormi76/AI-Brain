// In: frontend/instant_assistant/src-tauri/src/main.rs
// (修正后的完整代码)

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{Manager, RunEvent};
use tauri_plugin_shell::ShellExt;
use std::sync::Mutex;
use tauri_plugin_shell::process::CommandChild;

// 用于在Tauri的状态管理中存储所有子进程的句柄
struct AppState(Mutex<Vec<CommandChild>>);

fn main() {
    // 初始化应用状态，创建一个线程安全的Vec来存放子进程
    let app_state = AppState(Mutex::new(Vec::new()));

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init()) // 初始化shell插件，用于执行sidecar
        .manage(app_state) // 将我们的App状态交由Tauri管理
        .setup(|app| {
            println!("[Tauri] Setting up application and starting sidecars...");
            let shell = app.shell();
            let app_handle = app.handle();
            let path_resolver = app_handle.path();

            // 获取打包后资源目录的路径
            let resource_dir = path_resolver.resource_dir()
                .expect("Failed to get resource directory.");

            // 一个辅助闭包，用于方便地获取模型文件的完整路径
            let get_model_path = |model_name: &str| {
                resource_dir.join("models").join(model_name)
                    .to_string_lossy()
                    .to_string()
            };

            // 从Tauri状态中获取对AppState的可变引用
            let app_state = app.state::<AppState>();

            // 启动所有后台Sidecar服务
            
            // 1. 启动主后端API服务器
            let (_, server_child) = shell.sidecar("server")?.spawn()?;
            app_state.0.lock().unwrap().push(server_child);
            println!("[Tauri] Main backend API server sidecar started.");

            // 2. 启动向量数据库
            let (_, qdrant_child) = shell.sidecar("qdrant")?.spawn()?;
            app_state.0.lock().unwrap().push(qdrant_child);
            println!("[Tauri] Qdrant sidecar started.");

            // 3. 启动文本嵌入模型服务
            let embedding_model = get_model_path("bge-small-zh-v1.5-q8_0.gguf");
            let (_, embedding_child) = shell.sidecar("llama_embedding")?
                .args(["-m", &embedding_model, "--embedding", "--port", "8181", "-c", "2048"])
                .spawn()?;
            app_state.0.lock().unwrap().push(embedding_child);
            println!("[Tauri] Embedding server sidecar started.");
            
            // 4. 启动重排模型服务
            let reranker_model = get_model_path("bge-reranker-v2-m3-Q4_K_M.gguf");
            let (_, reranker_child) = shell.sidecar("llama_rerank")?
                .args(["-m", &reranker_model, "--port", "8080", "--ctx-size", "8192", "--rerank"])
                .spawn()?;
            app_state.0.lock().unwrap().push(reranker_child);
            println!("[Tauri] Reranker server sidecar started.");

            // 5. 启动聊天大语言模型服务
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
            // 监听应用退出事件，确保优雅地关闭所有后台子进程
            if let RunEvent::ExitRequested { .. } = event {
                println!("[Tauri] Exit requested. Killing all sidecar processes...");
                
                // 从状态中取出所有子进程的句柄
                let children_to_kill = std::mem::take(&mut *app_handle.state::<AppState>().0.lock().unwrap());

                // 遍历并杀死每一个子进程
                for child in children_to_kill {
                    if let Err(e) = child.kill() {
                        eprintln!("[Tauri] Failed to kill child process: {}", e);
                    }
                }
                
                println!("[Tauri] All sidecar processes terminated.");
            }
        });
} // <--- main函数的最后一个 `}`
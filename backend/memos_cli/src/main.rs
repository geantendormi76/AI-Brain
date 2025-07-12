use orchestrator::Orchestrator;
use agent_memos::MemosAgent;
use memos_core::{Agent, Command, Response};
use rustyline::DefaultEditor;
use sysinfo::System;
// 引入标准库中的 env 模块来处理环境变量
use std::env;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    println!("--- Memos Agent CLI Initializing ---");

    // =================================================================
    // == 硬件感知与智能决策模块 (V2 - 带手动覆盖) ==
    // =================================================================
    
    // 1. 优先检查环境变量，作为手动覆盖开关
    let force_performance_mode = env::var("FORCE_PERFORMANCE_MODE").is_ok();

    let reranker_llm_url: Option<&str> = if force_performance_mode {
        println!("[HardwareDetector] FORCE_PERFORMANCE_MODE is set. Forcing Performance-First mode.");
        None
    } else {
        // 2. 如果没有手动覆盖，则执行自动检测
        let mut sys = System::new();
        sys.refresh_memory();
        let total_memory_gb = sys.total_memory() as f64 / 1024.0 / 1024.0;
        println!("[HardwareDetector] Total system memory: {:.2} GB (Note: May be inaccurate in WSL)", total_memory_gb);

        const MEMORY_THRESHOLD_GB: f64 = 12.0;
        if total_memory_gb >= MEMORY_THRESHOLD_GB {
            println!("[HardwareDetector] Auto-detected: Memory is sufficient. Enabling Quality-First mode.");
            Some("http://localhost:8080")
        } else {
            println!("[HardwareDetector] Auto-detected: Memory is limited. Enabling Performance-First mode.");
            None
        }
    };
    // =================================================================

    let qdrant_url = "http://localhost:6334";
    let llm_url = "http://localhost:8282";
    let embedding_url = "http://localhost:8181";
    let memos_agent = MemosAgent::new(qdrant_url, embedding_url).await?; 
    let agents: Vec<Box<dyn Agent>> = vec![Box::new(memos_agent)];
    println!("Agents loaded: {} agent(s)", agents.len());

    let orchestrator = Orchestrator::new(agents, llm_url, reranker_llm_url);
    println!("Orchestrator created.");
    println!("\n欢迎使用 Memos 智能助理 (CLI版)");
    println!("请输入您的指令 (例如: '帮我记一下明天要开会'), 输入 'exit' 或按 Ctrl+C 退出。");
    println!("如果遇到不理想的回答，可以立即输入 /feedback 来帮助我们改进！"); // 新增一行引导

    let mut rl = DefaultEditor::new()?;

    loop {
        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                if input.eq_ignore_ascii_case("exit") {
                    break;
                }

                // --- 新增：处理反馈指令 ---
                if input.eq_ignore_ascii_case("/feedback") {
                    orchestrator.handle_feedback(); // 调用反馈处理逻辑
                    println!("\n[助理]:");
                    println!("感谢您的反馈！");
                    println!("> 已将上一轮不理想的交互记录，保存到了您程序目录下的 `feedback.jsonl` 文件中。");
                    println!("> 如果您愿意帮助我改进，可以将这个文件通过");
                    println!("> 邮箱：geantendormi89@gmail.com"); // 已更新为您的邮箱并修正拼写
                    println!("> 微信：geantendormi"); // 已更新为您的微信
                    println!("> 发送给我们。");
                    println!("> 您的每一次反馈，都是我变得更智能的动力！");
                    println!(); // 增加一个空行，保持格式美观
                    continue; // 处理完反馈后，跳过本次循环的后续部分
                }
                // --- 反馈指令处理结束 ---

                let _ = rl.add_history_entry(input);

                let command = Command::ProcessText(input.to_string());
                println!("[CLI] Sending command to orchestrator...");

                let result = orchestrator.dispatch(&command).await;

                println!("\n[助理]:");
                match result {
                    Ok(response) => {
                        match response {
                            Response::Text(text) => {
                                println!("{}", text);
                            }
                            Response::FileToOpen(path) => {
                                println!("请求打开文件: {:?}", path);
                            }
                            Response::Stream(mut receiver) => {
                                let mut is_thinking = false;
                                let mut full_response = String::new();

                                while let Some(token) = receiver.recv().await {
                                    if token.is_empty() {
                                        break;
                                    }
                                    full_response.push_str(&token);

                                    loop {
                                        if !is_thinking {
                                            if let Some(start_pos) = full_response.find("<think>") {
                                                print!("{}", &full_response[..start_pos]);
                                                is_thinking = true;
                                                if let Some(end_pos) = full_response[start_pos..].find("</think>") {
                                                    let end_in_slice = start_pos + end_pos + "</think>".len();
                                                    full_response.drain(..end_in_slice);
                                                    is_thinking = false;
                                                    continue;
                                                } else {
                                                    full_response.drain(..);
                                                    break;
                                                }
                                            }
                                        }

                                        if is_thinking {
                                            if let Some(end_pos) = full_response.find("</think>") {
                                                is_thinking = false;
                                                full_response.drain(..(end_pos + "</think>".len()));
                                                continue;
                                            } else {
                                                full_response.drain(..);
                                                break;
                                            }
                                        }

                                        if !is_thinking && !full_response.is_empty() {
                                            print!("{}", full_response);
                                            full_response.clear();
                                        }
                                        break;
                                    }

                                    use std::io::{self, Write};
                                    io::stdout().flush().unwrap();
                                }

                                if !is_thinking && !full_response.is_empty() {
                                    print!("{}", full_response);
                                }
                                println!();
                            }
                        }
                    },
                    Err(e) => {
                        eprintln!("发生错误: {}", e);
                    }
                }
                println!();
            }
            Err(_) => {
                break;
            }
        }
    }

    println!("感谢使用，再见！");
    Ok(())
}
use orchestrator::Orchestrator;
use agent_memos::MemosAgent;
use memos_core::{Agent, Command, Response};
use rustyline::DefaultEditor;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    println!("--- Memos Agent CLI Initializing ---");

    // 定义服务 URL
    let qdrant_url = "http://localhost:6334";
    // 在 V3.0 任务分解架构中，我们只需要一个 LLM 服务 URL
    let llm_url = "http://localhost:8282";

    // 初始化 MemosAgent
    let memos_agent = MemosAgent::new(qdrant_url).await?;
    let agents: Vec<Box<dyn Agent>> = vec![Box::new(memos_agent)];
    println!("Agents loaded: {} agent(s)", agents.len());

    // 使用新的、简化的签名初始化 Orchestrator
    let orchestrator = Orchestrator::new(agents, llm_url);
    println!("Orchestrator created.");
    println!("\n欢迎使用 Memos 智能助理 (CLI版)");
    println!("请输入您的指令 (例如: '帮我记一下明天要开会'), 输入 'exit' 或按 Ctrl+C 退出。");

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
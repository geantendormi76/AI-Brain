use orchestrator::Orchestrator;
use agent_memos::MemosAgent;
use memos_core::{Agent, Command, Response};
use rustyline::DefaultEditor;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    println!("--- Memos Agent CLI Initializing ---");

    // 我们需要两个URL：一个给Agent（Qdrant），一个给分类器（LLM）
    let qdrant_url = "http://localhost:6334";
    // URL现在是服务器的根地址，不包含具体的端点
    let classifier_llm_url = "http://localhost:8282"; 

    let memos_agent = MemosAgent::new(qdrant_url).await?;
    let agents: Vec<Box<dyn Agent>> = vec![Box::new(memos_agent)];
    println!("Agents loaded: {} agent(s)", agents.len());

    let orchestrator = Orchestrator::new(agents, classifier_llm_url);
    println!("Orchestrator created.");
    println!("\n欢迎使用 Memos 智能助理 (CLI版)");
    println!("请输入您的指令 (例如: '帮我记一下明天要开会'), 输入 'exit' 或按 Ctrl+C 退出。");

    let mut rl = DefaultEditor::new()?;

    loop {
        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                let input = line.trim();
                let _ = rl.add_history_entry(input);

                if input.is_empty() {
                    continue;
                }
                if input.eq_ignore_ascii_case("exit") {
                    break;
                }

                let command = Command::ProcessText(input.to_string());
                println!("[CLI] Sending command to orchestrator...");

                let result = orchestrator.dispatch(&command).await;

                println!("\n[助理]:");
                match result {
                    Ok(response) => {
                        match response {
                            Response::Text(text) => {
                                // 如果是普通文本响应，直接打印
                                println!("{}", text);
                            }
                            Response::FileToOpen(path) => {
                                // 处理打开文件请求
                                println!("请求打开文件: {:?}", path);
                            }
                            Response::Stream(mut receiver) => {
                                // 新增：用于过滤 <think> 标签的状态
                                let mut is_thinking = false;
                                let mut full_response = String::new(); // 用于拼接完整响应，以便处理标签

                                while let Some(token) = receiver.recv().await {
                                    if token.is_empty() { // 流结束信号
                                        break;
                                    }
                                    full_response.push_str(&token);

                                    // 持续处理，直到无法再找到完整的 <think> 或 </think> 标签
                                    loop {
                                        if !is_thinking {
                                            if let Some(start_pos) = full_response.find("<think>") {
                                                // 打印标签前的内容
                                                print!("{}", &full_response[..start_pos]);
                                                // 更新状态，并移除已处理的部分（包括标签）
                                                is_thinking = true;
                                                if let Some(end_pos) = full_response[start_pos..].find("</think>") {
                                                    // 如果在同一批次中找到了结束标签
                                                    let end_in_slice = start_pos + end_pos + "</think>".len();
                                                    full_response.drain(..end_in_slice);
                                                    is_thinking = false;
                                                    continue; // 继续循环，处理剩余的字符串
                                                } else {
                                                    // 如果只找到了开始标签
                                                    full_response.drain(..); // 清空已处理部分
                                                    break; // 等待更多token
                                                }
                                            }
                                        }
                                        
                                        if is_thinking {
                                            if let Some(end_pos) = full_response.find("</think>") {
                                                // 找到了结束标签，更新状态并移除已处理部分
                                                is_thinking = false;
                                                full_response.drain(..(end_pos + "</think>".len()));
                                                continue; // 继续循环，处理剩余的字符串
                                            } else {
                                                // 仍在思考中，清空buffer，不打印
                                                full_response.drain(..);
                                                break; // 等待更多token
                                            }
                                        }

                                        // 如果没有在思考，打印剩余内容并清空
                                        if !is_thinking && !full_response.is_empty() {
                                            print!("{}", full_response);
                                            full_response.clear();
                                        }
                                        break; // 没有更多标签可以处理，跳出内部循环
                                    }
                                    
                                    // 刷新标准输出
                                    use std::io::{self, Write};
                                    io::stdout().flush().unwrap();
                                }

                                // 打印最后剩余的内容（如果有）
                                if !is_thinking && !full_response.is_empty() {
                                    print!("{}", full_response);
                                }
                                println!(); // 流结束时打印一个换行符
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
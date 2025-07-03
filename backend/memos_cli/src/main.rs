use orchestrator::Orchestrator;
use agent_memos::MemosAgent;
use memos_core::{Agent, Command, Response};
use rustyline::DefaultEditor;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    println!("--- Memos Agent CLI Initializing ---");

    // 定义Qdrant服务的地址
    let qdrant_url = "http://localhost:6334";

    // MemosAgent::new 现在只接收qdrant_url
    let memos_agent = MemosAgent::new(qdrant_url).await?;
    let agents: Vec<Box<dyn Agent>> = vec![Box::new(memos_agent)];
    println!("Agents loaded: {} agent(s)", agents.len());

    let orchestrator = Orchestrator::new(agents);
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
                    Ok(response) => match response {
                        Response::Text(text) => {
                            println!("{}", text);
                        }
                        Response::FileToOpen(path) => {
                            println!("请求打开文件: {:?}", path);
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
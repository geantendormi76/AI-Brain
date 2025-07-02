use orchestrator::Orchestrator;
use agent_memos::MemosAgent;
use memos_core::{Agent, Command, Response};
// 导入 rustyline 的 Editor 和它的默认 Helper
use rustyline::DefaultEditor; 

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    println!("--- Memos Agent CLI Initializing ---");

    let memos_agent = MemosAgent::new()?;
    let agents: Vec<Box<dyn Agent>> = vec![Box::new(memos_agent)];
    println!("Agents loaded: {} agent(s)", agents.len());

    let orchestrator = Orchestrator::new(agents);
    println!("Orchestrator created.");
    println!("\n欢迎使用 Memos 智能助理 (CLI版)");
    println!("请输入您的指令 (例如: '帮我记一下明天要开会'), 输入 'exit' 或按 Ctrl+C 退出。");

    // 创建一个 rustyline 的默认编辑器实例，这是最简单的方式
    let mut rl = DefaultEditor::new()?;

    // 使用 rustyline 的循环
    loop {
        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                let input = line.trim();
                // 将输入添加到历史记录中
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
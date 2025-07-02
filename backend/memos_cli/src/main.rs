use orchestrator::Orchestrator;
use memos_core::{Command, Response};
use std::io::{self, Write};

// 使用 tokio::main 宏，让我们的 main 函数可以支持 async/await
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    println!("欢迎使用 Memos 智能助理 (CLI版)");
    println!("请输入您的指令 (例如: '帮我记一下明天要开会')，输入 'exit' 退出。");

    // 1. 创建我们的中枢调度器实例
    let orchestrator = Orchestrator::new();

    // 2. 进入一个无限循环，持续接收用户输入
    loop {
        print!("> ");
        // 确保提示符立即显示出来
        io::stdout().flush()?;

        let mut input = String::new();
        // 读取用户在命令行输入的一行
        io::stdin().read_line(&mut input)?;

        let input = input.trim(); // 去掉输入内容前后的空格和换行符

        // 如果用户输入 "exit"，就退出程序
        if input.eq_ignore_ascii_case("exit") {
            println!("感谢使用，再见！");
            break;
        }

        // 如果用户输入为空，就继续下一次循环
        if input.is_empty() {
            continue;
        }

        // 3. 将用户的输入文本包装成一个 Command
        let command = Command::ProcessText(input.to_string());

        // 4. 调用调度器来处理这个指令，并等待结果
        println!("[CLI] Sending command to orchestrator...");
        match orchestrator.process_command(command).await {
            Ok(response) => {
                // 5. 根据调度器返回的 Response，以友好的方式显示给用户
                match response {
                    Response::Text(text) => {
                        println!("[助理]: {}", text);
                    }
                    Response::FileToOpen(path) => {
                        println!("[助理]: 我需要为您打开这个文件: {:?}", path);
                    }
                }
            }
            Err(e) => {
                // 如果处理过程中发生任何错误，就打印出来
                eprintln!("[错误]: 处理指令时发生错误: {}", e);
            }
        }
    }

    Ok(())
}
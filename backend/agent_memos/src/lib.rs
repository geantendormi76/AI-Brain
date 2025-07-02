// 从我们新命名的 memos_core crate 中导入类型
// 现在，"core" 这个名字是自由的，async_trait 可以不受干扰地使用它了
use memos_core::{Agent, Command, Response};
use async_trait::async_trait;

// 这是Memos Agent的结构体
pub struct MemosAgent {}

impl MemosAgent {
    pub fn new() -> Self {
        Self {}
    }
}

// 实现我们从 memos_core 导入的 Agent trait
#[async_trait]
impl Agent for MemosAgent {
    fn name(&self) -> &'static str {
        "memos_agent"
    }

    fn interests(&self) -> &[&'static str] {
        &["提醒", "记一下", "别忘了", "帮我记"]
    }

    async fn handle_command(&self, command: &Command) -> Result<Response, anyhow::Error> {
        match command {
            Command::ProcessText(text) => {
                println!("[MemosAgent] Received text: {}", text);
                todo!("Implement MemosAgent logic here!");
            }
        }
    }
}
use memos_core::{Agent, Command, Response};
use agent_memos::MemosAgent;
use std::sync::Arc;

/// 中枢调度器
/// 它的职责是管理所有的Agent，并根据指令分发任务
pub struct Orchestrator {
    agents: Vec<Arc<dyn Agent>>,
}

impl Orchestrator {
    /// 创建一个新的调度器实例
    pub fn new() -> Self {
        let memos_agent = Arc::new(MemosAgent::new());

        Self {
            agents: vec![memos_agent],
        }
    }

    /// 处理指令的核心方法
    pub async fn process_command(&self, command: Command) -> Result<Response, anyhow::Error> {
        match command {
            Command::ProcessText(text) => {
                for agent in &self.agents {
                    if agent.interests().iter().any(|interest| text.contains(interest)) {
                        println!("[Orchestrator] Found interested agent: {}. Routing command...", agent.name());
                        // 注意：这里我们将 text 的所有权转移给新的 Command
                        return agent.handle_command(&Command::ProcessText(text)).await;
                    }
                }
                
                println!("[Orchestrator] No agent found for this command.");
                Ok(Response::Text("抱歉，我暂时无法理解您的指令。".to_string()))
            }
        }
    }
} // <--- 在这里补上缺失的右花括号！
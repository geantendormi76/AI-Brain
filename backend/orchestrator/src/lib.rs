use memos_core::{Agent, Command, Response};

// 重新导出依赖，方便上层模块（如CLI）使用
// 这是一个可选的、方便性的做法
pub use agent_memos;
pub use memos_core;

pub struct Orchestrator {
    agents: Vec<Box<dyn Agent>>,
}

impl Orchestrator {
    pub fn new(agents: Vec<Box<dyn Agent>>) -> Self {
        Self { agents }
    }

    // 使用 match 语句重构 dispatch 方法
    pub async fn dispatch(&self, command: &Command) -> Result<Response, anyhow::Error> {
        match command {
            // 分支1：处理文本指令
            Command::ProcessText(text) => {
                // 遍历所有 agent
                for agent in &self.agents {
                    // 检查 agent 的兴趣列表
                    for interest in agent.interests() {
                        // 如果指令包含了 agent 感兴趣的关键词...
                        if text.contains(interest) {
                            println!(
                                "[Orchestrator] Found interested agent: '{}'. Routing command...",
                                agent.name()
                            );
                            // ...就把任务交给他，并返回他的处理结果
                            return agent.handle_command(command).await;
                        }
                    }
                }
                
                // 如果循环结束都没有找到感兴趣的 agent
                println!("[Orchestrator] No interested agent found for text command.");
                Ok(Response::Text("抱歉，我无法理解您的指令。".to_string()))
            }
            
            // 分支2：处理未来可能出现的其他指令类型
            // 使用 `_` 通配符来捕获所有其他未明确处理的 Command 成员
            // _ => {
            //     println!("[Orchestrator] Received a non-text command, which is not supported yet.");
            //     Err(anyhow::anyhow!("Currently, only text commands are supported."))
            // }
        }
    }
}
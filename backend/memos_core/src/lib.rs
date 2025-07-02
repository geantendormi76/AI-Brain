use std::path::PathBuf;
use async_trait::async_trait;

// 1. 标准化的指令：UI层发给调度器的唯一入口
#[derive(Debug, Clone)] // 添加Debug和Clone派生
pub enum Command {
    ProcessText(String),
    // 未来可以扩展: ProcessAudioChunk(Vec<f32>), etc.
}

// 2. 标准化的响应：调度器返回给UI层的唯一出口
#[derive(Debug, Clone)] // 添加Debug和Clone派生
pub enum Response {
    Text(String),
    FileToOpen(PathBuf),
    // ... etc.
}

// 3. 所有Agent必须遵守的行为准则 (Trait)
// 我们需要async_trait来在trait中使用async fn
#[async_trait]
pub trait Agent: Send + Sync {
    // 每个Agent的名字，用于调试和路由
    fn name(&self) -> &'static str;

    // 每个Agent告诉调度器，它对哪些关键词感兴趣
    // 这是实现高效路由的关键
    fn interests(&self) -> &[&'static str];

    // 每个Agent处理指令的核心方法
    async fn handle_command(&self, command: &Command) -> Result<Response, anyhow::Error>;
}
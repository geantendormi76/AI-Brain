// memos_core/src/lib.rs
use std::path::PathBuf;
use async_trait::async_trait;
use tokio::sync::mpsc; // 新增：用于Stream变体
use std::any::Any;

// 1. 标准化的指令：UI层发给调度器的唯一入口
#[derive(Debug, Clone)] // Command 仍然可以 Clone
pub enum Command {
    ProcessText(String),
    // 未来可以扩展: ProcessAudioChunk(Vec<f32>), etc.
}

// 2. 标准化的响应：调度器返回给UI层的唯一出口
// 移除 Clone 派生，因为 mpsc::Receiver 不能 Clone
#[derive(Debug)] // 只有 Debug，没有 Clone
pub enum Response {
    Text(String),
    FileToOpen(PathBuf),
    Stream(mpsc::Receiver<String>), // 新增：用于流式文本
    // ... etc.
}

// 3. 所有Agent必须遵守的行为准则 (Trait)
// 我们需要async_trait来在trait中使用async fn
#[async_trait]
pub trait Agent: Send + Sync {
    fn name(&self) -> &'static str;
    fn interests(&self) -> &[&'static str];
    async fn handle_command(&self, command: &Command) -> Result<Response, anyhow::Error>;
    
    // 新增: 用于支持向下转型 (downcasting)
    fn as_any(&self) -> &dyn Any;
}
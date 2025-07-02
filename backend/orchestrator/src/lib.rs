// 声明一个名为 orchestrator 的模块，并将其设为公共
pub mod orchestrator;

// 将 orchestrator 模块中的 Orchestrator 结构体重新导出，
// 这样其他 crate 就可以直接通过 `use orchestrator::Orchestrator` 来使用它了。
// 这是Rust库设计的标准实践。
pub use crate::orchestrator::Orchestrator;
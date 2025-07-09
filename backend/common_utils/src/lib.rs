// In C:\zhzai\AI\backend\common_utils\src\lib.rs

use serde::Deserialize;
use sysinfo::System;

// 定义一个枚举来表示不同的性能模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceMode {
    QualityFirst,     // 质量优先 (高配)
    PerformanceFirst, // 性能优先 (低配)
}

// 硬件检测函数
pub fn detect_performance_mode() -> PerformanceMode {
    // 优先检查环境变量，作为手动覆盖开关
    if std::env::var("FORCE_PERFORMANCE_MODE").is_ok() {
        println!("[HardwareDetector] FORCE_PERFORMANCE_MODE is set. Forcing Performance-First mode.");
        return PerformanceMode::PerformanceFirst;
    }

    // 如果没有手动覆盖，则执行自动检测
    let mut sys = System::new_all();
    sys.refresh_memory();
    let total_memory_gb = sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    println!("[HardwareDetector] Total system memory: {:.2} GB", total_memory_gb);

    const MEMORY_THRESHOLD_GB: f64 = 12.0;

    if total_memory_gb >= MEMORY_THRESHOLD_GB {
        println!("[HardwareDetector] Auto-detected: Memory is sufficient. Enabling Quality-First mode.");
        PerformanceMode::QualityFirst
    } else {
        println!("[HardwareDetector] Auto-detected: Memory is limited. Enabling Performance-First mode.");
        PerformanceMode::PerformanceFirst
    }
}

// 定义一个统一的配置结构体
#[derive(Debug, Clone, Deserialize)]
pub struct ServiceUrls {
    pub llm_url: String,
    pub embedding_url: String,
    pub reranker_url: Option<String>, 
    pub qdrant_url: String,
}

// 提供一个加载默认配置的函数
pub fn load_default_urls() -> ServiceUrls {
    ServiceUrls {
        llm_url: "http://localhost:8282".to_string(),
        embedding_url: "http://localhost:8181".to_string(),
        reranker_url: None,
        qdrant_url: "http://localhost:6334".to_string(),
    }
}
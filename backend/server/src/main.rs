// In ~/AI/backend/server/src/main.rs

use axum::{
    debug_handler,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use orchestrator::Orchestrator; 
use memos_core::{Command, Response as CoreResponse};
use agent_memos::MemosAgent;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tower_http::{cors::{Any, CorsLayer}, trace::TraceLayer};
use std::sync::Arc;
use tokio::task; // 导入 tokio::task
use common_utils::{detect_performance_mode, PerformanceMode, load_default_urls};

// API 层 DTOs (保持不变)
#[derive(Serialize)] #[serde(rename_all = "PascalCase")] struct ApiResponse { text: String }
#[derive(Deserialize)] struct ApiCommand { #[serde(rename = "ProcessText")] process_text: String }

// 专业的错误处理 (保持不变)
#[derive(Debug, Error)]
enum ApiError {
    #[error("Orchestrator task failed")]
    TaskJoin(#[from] task::JoinError),
    #[error("Orchestrator dispatch failed")]
    Dispatch(anyhow::Error),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let error_message = self.to_string();
        eprintln!("[Server Error] {}", error_message);
        let body = Json(serde_json::json!({ "Text": "An internal server error occurred." }));
        (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
    }
}

// Axum Handler (核心修正)
#[debug_handler]
async fn dispatch_handler(
    State(orchestrator): State<Arc<Orchestrator>>,
    Json(payload): Json<ApiCommand>,
) -> Result<Json<ApiResponse>, ApiError> {
    let command = Command::ProcessText(payload.process_text);

    // 使用 spawn_blocking 将业务逻辑移到另一个线程执行
    // 这保证了我们的 handler 本身返回的 Future 是 Send
    let core_response = task::spawn_blocking(move || {
        // 创建一个新的、小型的同步运行时来驱动我们的异步业务逻辑
        // 这是在同步上下文中运行异步代码的标准模式
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        
        rt.block_on(orchestrator.dispatch(&command))
    })
    .await? // .await 在这里等待 spawn_blocking 的任务完成，并处理 JoinError
    .map_err(ApiError::Dispatch)?; // 将 dispatch 内部的 anyhow::Error 转换为 ApiError

    let api_response = match core_response {
        CoreResponse::Text(text) => ApiResponse { text },
        _ => ApiResponse { text: "[Info] Backend returned a non-text response.".to_string() },
    };
    
    Ok(Json(api_response))
}

// 主函数 (保持不变)
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("[Server] Initializing...");

    // 使用 common_utils 加载默认的服务URL
    let mut service_urls = load_default_urls();

    println!("[Server] Detecting hardware and setting performance mode...");
    let mode = detect_performance_mode();
    if mode == PerformanceMode::QualityFirst {
        // 如果是高质量模式，则启用 reranker
        service_urls.reranker_url = Some("http://localhost:8080".to_string());
        println!("[Server] Quality-First mode enabled. Reranker URL set.");
    } else {
        println!("[Server] Performance-First mode enabled. Reranker will not be used.");
    }

    println!("[Server] Initializing MemosAgent...");
    // 注意：这里的 MemosAgent::new 调用可能需要根据WSL版的定义来调整
    // 我们先假设WSL版的 new 函数需要 qdrant_url 和 embedding_url
    let memos_agent = MemosAgent::new(
        &service_urls.qdrant_url, 
        &service_urls.embedding_url
    ).await?;
    let agents: Vec<Box<dyn memos_core::Agent>> = vec![Box::new(memos_agent)];
    
    println!("[Server] Initializing Orchestrator...");
    let orchestrator = Orchestrator::new(
        agents, 
        &service_urls.llm_url, 
        service_urls.reranker_url.as_deref() // 使用 as_deref 传递 Option<&str>
    );

    let shared_state = Arc::new(orchestrator);
    println!("[Server] Orchestrator initialized.");

    let app = Router::new()
        .route("/api/v1/dispatch", post(dispatch_handler))
        .with_state(shared_state)
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any))
        .layer(TraceLayer::new_for_http());

    // 监听端口保持和Tauri前端一致
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8383").await?;
    println!("[Server] API Gateway listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    
    Ok(())
}
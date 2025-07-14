// In ~/AI/backend/server/src/main.rs

use axum::{
    debug_handler,
    extract::State,
    http::{StatusCode, HeaderMap, HeaderName, HeaderValue}, // 导入HeaderMap, HeaderName
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
use tokio::task;
use common_utils::{detect_performance_mode, PerformanceMode, load_default_urls};
use std::path::Path;

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
        
        let mut headers = HeaderMap::new();
        headers.insert(HeaderName::from_static("content-type"), HeaderValue::from_static("application/json; charset=utf-8"));

        (StatusCode::INTERNAL_SERVER_ERROR, headers, body).into_response()
    }
}
// Axum Handler (核心修正)
#[debug_handler]
async fn dispatch_handler(
    State(orchestrator): State<Arc<Orchestrator>>,
    Json(payload): Json<ApiCommand>,
) -> Result<(StatusCode, HeaderMap, Json<ApiResponse>), ApiError> {
    let command = Command::ProcessText(payload.process_text);

    let core_response = task::spawn_blocking(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        
        rt.block_on(orchestrator.dispatch(&command))
    })
    .await?
    .map_err(ApiError::Dispatch)?;

    let api_response = match core_response {
        CoreResponse::Text(text) => ApiResponse { text },
        _ => ApiResponse { text: "[Info] Backend returned a non-text response.".to_string() },
    };
    
    let mut headers = HeaderMap::new();
    headers.insert(HeaderName::from_static("content-type"), HeaderValue::from_static("application/json; charset=utf-8"));
    
    Ok((StatusCode::OK, headers, Json(api_response)))
}

// 主函数 (保持不变)
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("[Server] Initializing...");

    let models_path_str = "models"; // 从 server 目录出发，返回上一层到 backend，再进入 models
    let models_path = Path::new(models_path_str);
    let models_path = match models_path.canonicalize() {
        Ok(path) => {
            println!("[CLI] Found models path at: {:?}", path);
            path
        }
        Err(e) => {
            eprintln!("CRITICAL ERROR: Could not find models directory at '{}'.", models_path_str);
            eprintln!("Please ensure the 'models' directory exists relative to the 'backend' directory.");
            eprintln!("Underlying error: {}", e);
            std::process::exit(1); // 错误无法恢复，直接退出程序
        }
    };
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
    let memos_agent = MemosAgent::new(qdrant_url, embedding_url, &models_path).await?;
    let agents: Vec<Box<dyn memos_core::Agent>> = vec![Box::new(memos_agent)];
    
    println!("[Server] Initializing Orchestrator...");
    let orchestrator = Orchestrator::new(
        agents, 
        &service_urls.llm_url, 
        service_urls.reranker_url.as_deref(),
        &models_path // <-- 新增的参数
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
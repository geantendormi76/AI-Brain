// geantendormi76-ai-brain/backend/server/src/main.rs

use axum::{
    debug_handler,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response as AxumResponse}, // 明确 Axum 的 Response 类型
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

#[derive(Serialize)] #[serde(rename_all = "PascalCase")] struct ApiResponse { text: String }
#[derive(Deserialize)] struct ApiCommand { #[serde(rename = "ProcessText")] process_text: String }

#[derive(Debug, Error)]
enum ApiError {
    #[error("Orchestrator task failed")]
    TaskJoin(#[from] task::JoinError),
    #[error("Orchestrator dispatch failed")]
    Dispatch(#[from] anyhow::Error), // 直接从 anyhow::Error 转换
}

// ======================= 【核心修正开始】 =======================
// 重写错误处理，使其能够打印详细的、包含根本原因的错误链
impl IntoResponse for ApiError {
    fn into_response(self) -> AxumResponse {
        // 使用 {:#?} 格式化指令，它会递归地打印出整个错误链（chain of causes）。
        // 这将把隐藏在 "Orchestrator dispatch failed" 这句通用信息背后的、
        // 真正导致问题的具体错误（例如“JSON解析错误在第X行”）暴露出来。
        eprintln!("[Server Error] Detailed error: {:#?}", self); 
        
        let body = Json(serde_json::json!({ "Text": "An internal server error occurred. Check the backend console for detailed logs." }));
        (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
    }
}
// ======================= 【核心修正结束】 =======================

#[debug_handler]
async fn dispatch_handler(
    State(orchestrator): State<Arc<Orchestrator>>,
    Json(payload): Json<ApiCommand>,
) -> Result<Json<ApiResponse>, ApiError> {
    let command = Command::ProcessText(payload.process_text);

    let core_response = orchestrator.dispatch(&command)
        .await
        .map_err(ApiError::Dispatch)?; // .map_err 现在可以正确地包裹 anyhow::Error

    let api_response = match core_response {
        CoreResponse::Text(text) => ApiResponse { text },
        _ => ApiResponse { text: "[Info] Backend returned a non-text response.".to_string() },
    };
    
    Ok(Json(api_response))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("[Server] Initializing...");

    let mut service_urls = load_default_urls();

    println!("[Server] Detecting hardware and setting performance mode...");
    let mode = detect_performance_mode();
    if mode == PerformanceMode::QualityFirst {
        service_urls.reranker_url = Some("http://localhost:8080".to_string());
        println!("[Server] Quality-First mode enabled. Reranker URL set.");
    } else {
        println!("[Server] Performance-First mode enabled. Reranker will not be used.");
    }

    println!("[Server] Initializing MemosAgent...");
    let memos_agent = MemosAgent::new(
        &service_urls.qdrant_url, 
        &service_urls.embedding_url
    ).await?;
    let agents: Vec<Box<dyn memos_core::Agent>> = vec![Box::new(memos_agent)];
    
    println!("[Server] Initializing Orchestrator...");
    let orchestrator = Orchestrator::new(
        agents, 
        &service_urls.llm_url, 
        service_urls.reranker_url.as_deref()
    );

    let shared_state = Arc::new(orchestrator);
    println!("[Server] Orchestrator initialized.");

    let app = Router::new()
        .route("/api/v1/dispatch", post(dispatch_handler))
        .with_state(shared_state)
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any))
        .layer(TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8383").await?;
    println!("[Server] API Gateway listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    
    Ok(())
}
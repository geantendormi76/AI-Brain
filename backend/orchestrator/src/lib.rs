// orchestrator/src/lib.rs

// 引入我们的专家组模块
mod experts;

use agent_memos::MemosAgent;
use memos_core::{Agent, Command, Response};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, Mutex};
// 引入专家组的具体成员
use experts::{
    router::{self, RoutingDecision, ToolToCall},
    save_expert,
    recall_expert,
};

// ---- LLMConfig: 用于持有 client 和 url ----
// 我们复用这个结构来统一管理LLM的配置
pub struct LLMConfig {
    client: Client,
    llm_url: String,
}

impl LLMConfig {
    pub fn new(llm_url: &str) -> Self {
        Self {
            client: Client::new(),
            llm_url: llm_url.to_string(),
        }
    }
}

// ---- Orchestrator struct and impl (V4.1) ----
pub struct Orchestrator {
    agents: Vec<Box<dyn Agent>>,
    llm_config: LLMConfig,
    conversation_history: Arc<Mutex<Vec<String>>>,
}

impl Orchestrator {
    pub fn new(agents: Vec<Box<dyn Agent>>, llm_url: &str) -> Self {
        println!("[Orchestrator] V4.1 Initializing in Router-Expert mode.");
        Self {
            agents,
            llm_config: LLMConfig::new(llm_url),
            conversation_history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    // --- 专家工具的具体实现 ---
    async fn handle_save(&self, text: &str) -> Result<String, anyhow::Error> {
        let memos_agent = self.agents.iter().find_map(|a| a.as_any().downcast_ref::<MemosAgent>()).ok_or_else(|| anyhow::anyhow!("MemosAgent not found"))?;

        println!("[SaveExpert] Extracting fact from: '{}'", text);
        let messages = save_expert::get_fact_extraction_prompt(text);
        // 修正：为 SaveExpert 添加 GBNF 约束
        let gbnf_schema = save_expert::get_fact_extraction_gbnf_schema();
        let request_body = json!({ "messages": messages, "temperature": 0.0, "grammar": gbnf_schema });
        let chat_url = format!("{}/v1/chat/completions", self.llm_config.llm_url);
        let response = self.llm_config.client.post(&chat_url).json(&request_body).send().await?;
        
        #[derive(Deserialize)] struct ChatChoice { message: ChatMessageContent }
        #[derive(Deserialize)] struct ChatMessageContent { content: String }
        #[derive(Deserialize)] struct ChatCompletionResponse { choices: Vec<ChatChoice> }
        let chat_response: ChatCompletionResponse = response.json().await?;
        let content_str = chat_response.choices.get(0).map(|c| c.message.content.trim()).unwrap_or("{}");
        
        // 修正：解析新的结构体
        let extracted_fact_obj: save_expert::ExtractedFact = serde_json::from_str(content_str)?;
        let fact_to_save = &extracted_fact_obj.fact;

        println!("[SaveExpert] Fact to save: '{}'", fact_to_save);
        memos_agent.save(fact_to_save).await?;
        Ok("好的，已经记下了。".to_string())
    }

    // 查询专家
    async fn handle_recall(&self, text: &str, history: &[String]) -> Result<String, anyhow::Error> {
        let memos_agent = self.agents.iter()
            .find_map(|a| a.as_any().downcast_ref::<MemosAgent>())
            .ok_or_else(|| anyhow::anyhow!("MemosAgent not found"))?;

        // 1. 调用LLM重写查询
        println!("[RecallExpert] Rewriting query: '{}'", text);
        let messages = recall_expert::get_query_rewrite_prompt(text, history);
        let gbnf_schema = recall_expert::get_query_rewrite_gbnf_schema();
        let request_body = json!({ "messages": messages, "temperature": 0.0, "grammar": gbnf_schema });
        let chat_url = format!("{}/v1/chat/completions", self.llm_config.llm_url);
        let response = self.llm_config.client.post(&chat_url).json(&request_body).send().await?;
        #[derive(Deserialize)] struct ChatChoice { message: ChatMessageContent }
        #[derive(Deserialize)] struct ChatMessageContent { content: String }
        #[derive(Deserialize)] struct ChatCompletionResponse { choices: Vec<ChatChoice> }
        let chat_response: ChatCompletionResponse = response.json().await?;
        let content_str = chat_response.choices.get(0).map(|c| c.message.content.trim()).unwrap_or("{}");
        let rewritten_query_obj: recall_expert::RewrittenQuery = serde_json::from_str(content_str)?;
        let rewritten_query = &rewritten_query_obj.rewritten_query;
        println!("[RecallExpert] Rewritten query: '{}'", rewritten_query);

        // 2. 使用重写后的查询进行召回
        let candidate_points = memos_agent.recall(rewritten_query).await?;
        if candidate_points.is_empty() {
            Ok(format!("关于“{}”，我好像没什么印象...", rewritten_query))
        } else {
            let top_point = candidate_points.get(0).unwrap();
            // 修正：使用 map_or 来优雅地处理 Option<&str>
            let content = top_point.payload.get("content")
                                   .and_then(|v| v.as_str())
                                   .map_or("".to_string(), |v| v.to_string());
            Ok(content)
        }
    }

    // --- 总指挥的核心调度逻辑 ---
    pub async fn dispatch(&self, command: &Command) -> Result<Response, anyhow::Error> {
        match command {
            Command::ProcessText(text) => {
                // 1. 路由决策
                let history_guard = self.conversation_history.lock().unwrap();
                let messages = router::get_routing_prompt(text, &history_guard);
                let gbnf_schema = router::get_routing_gbnf_schema();
                drop(history_guard); // 尽早释放锁

                let request_body = json!({ 
                    "messages": messages, 
                    "grammar": gbnf_schema,
                    "temperature": 0.0,
                });
                let chat_url = format!("{}/v1/chat/completions", self.llm_config.llm_url);
                
                println!("[Router] Sending routing request...");
                let response = self.llm_config.client.post(&chat_url).json(&request_body).send().await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_body = response.text().await?;
                    return Err(anyhow::anyhow!("Router LLM service returned an error. Status: {}. Body: {}", status, error_body));
                }

                // 修正：添加正确的、分层的响应解析逻辑
                #[derive(Deserialize)] struct ChatChoice { message: ChatMessageContent }
                #[derive(Deserialize)] struct ChatMessageContent { content: String }
                #[derive(Deserialize)] struct ChatCompletionResponse { choices: Vec<ChatChoice> }
                
                let chat_response: ChatCompletionResponse = response.json().await?;
                let content_str = chat_response.choices.get(0).map(|c| c.message.content.trim())
                    .ok_or_else(|| anyhow::anyhow!("Missing 'choices' in router LLM response"))?;

                println!("[Router] Decision JSON: {}", content_str);
                let decision: RoutingDecision = serde_json::from_str(content_str)?;

                // 2. 根据决策调用专家
                let final_response = match decision.tool_to_call {
                    ToolToCall::SaveTool => self.handle_save(text).await?,
                    ToolToCall::RecallTool => {
                        let history_guard = self.conversation_history.lock().unwrap();
                        self.handle_recall(text, &history_guard).await?
                    },
                    ToolToCall::MixedTool => {
                        // 简化处理：先保存再查询
                        let save_response = self.handle_save(text).await?;
                        let history_guard = self.conversation_history.lock().unwrap();
                        let recall_response = self.handle_recall(text, &history_guard).await?;
                        format!("{}\n{}", save_response, recall_response)
                    },
                    ToolToCall::NoTool => "抱歉，我不太明白您的意思，可以换个方式说吗？".to_string(),
                };

                // 3. 更新历史并返回
                let mut history = self.conversation_history.lock().unwrap();
                history.push(format!("User: {}", text));
                history.push(format!("Assistant: {}", final_response));
                const MAX_HISTORY_SIZE: usize = 8;
                let current_len = history.len();
                if current_len > MAX_HISTORY_SIZE {
                    let drain_count = current_len - MAX_HISTORY_SIZE;
                    history.drain(..drain_count);
                }
                println!("[Orchestrator] Updated history: {:?}", history);

                Ok(Response::Text(final_response))
            }
        }
    }
}
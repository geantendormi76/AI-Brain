// orchestrator/src/lib.rs

mod experts;
mod preprocessors;

use agent_memos::MemosAgent;
use memos_core::{Agent, Command, Response};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, Mutex};

// 【最终风格修正】使用 snake_case 路径
use experts::memos_agent::{
    router::{self, RoutingDecision, ToolToCall},
    save_expert,
    modify_expert,
    confirmation_expert,
    re_ranker::{ReRanker, ReRankRequest, DocumentToRank, ReRankStrategy},
};
use preprocessors::memos_agent::disambiguation;


#[derive(Debug, Clone)]
pub enum PendingActionType {
    ModifyConfirmation { memory_id: i64, original_content: String },
    DeleteConfirmation { memory_id: i64, content_to_delete: String },
}

#[derive(Debug, Clone)]
pub struct PendingAction {
    pub action_type: PendingActionType,
    pub original_user_request: String,
}

pub struct LLMConfig {
    client: Client,
    llm_url: String,
}

impl LLMConfig {
    pub fn new(llm_url: &str) -> Self {
        Self { client: Client::new(), llm_url: llm_url.to_string() }
    }
}

pub struct Orchestrator {
    agents: Vec<Box<dyn Agent>>,
    llm_config: LLMConfig,
    conversation_history: Arc<Mutex<Vec<String>>>,
    reranker: Option<ReRanker>,
    pending_action: Arc<Mutex<Option<PendingAction>>>,
}

impl Orchestrator {
    pub fn new(agents: Vec<Box<dyn Agent>>, llm_url: &str, reranker_llm_url: Option<&str>) -> Self {
        println!("[Orchestrator] V5.3 Initializing with Agent-centric module structure.");
        let reranker = if let Some(url) = reranker_llm_url {
            println!("[Orchestrator] ReRanker is ENABLED.");
            Some(ReRanker::new(url))
        } else {
            println!("[Orchestrator] ReRanker is DISABLED.");
            None
        };
        Self {
            agents,
            llm_config: LLMConfig::new(llm_url),
            conversation_history: Arc::new(Mutex::new(Vec::new())),
            reranker,
            pending_action: Arc::new(Mutex::new(None)),
        }
    }

    async fn handle_save(&self, disambiguated_text: &str) -> Result<String, anyhow::Error> {
        let memos_agent = self.agents.iter().find_map(|a| a.as_any().downcast_ref::<MemosAgent>()).ok_or_else(|| anyhow::anyhow!("MemosAgent not found"))?;
        println!("[SaveExpert] Extracting fact from already disambiguated text: '{}'", disambiguated_text);
        let messages = save_expert::get_fact_extraction_prompt(disambiguated_text);
        let gbnf_schema = save_expert::get_fact_extraction_gbnf_schema();
        let request_body = json!({ "messages": messages, "temperature": 0.0, "grammar": gbnf_schema });
        let chat_url = format!("{}/v1/chat/completions", self.llm_config.llm_url);
        let response = self.llm_config.client.post(&chat_url).json(&request_body).send().await?;
        #[derive(Deserialize)] struct ChatChoice { message: ChatMessageContent }
        #[derive(Deserialize)] struct ChatMessageContent { content: String }
        #[derive(Deserialize)] struct ChatCompletionResponse { choices: Vec<ChatChoice> }
        let chat_response: ChatCompletionResponse = response.json().await?;
        let content_str = chat_response.choices.get(0).map(|c| c.message.content.trim()).unwrap_or("{}");
        let extracted_fact_obj: save_expert::ExtractedFact = serde_json::from_str(content_str)?;
        let fact_to_save = &extracted_fact_obj.fact;
        println!("[SaveExpert] Fact to save: '{}'", fact_to_save);
        memos_agent.save(fact_to_save).await?;
        Ok("好的，已经记下了。".to_string())
    }

    async fn handle_recall(&self, text: &str) -> Result<String, anyhow::Error> {
        println!("[RecallExpert] Received recall request for: '{}'", text);
        let memos_agent = self.agents.iter().find_map(|a| a.as_any().downcast_ref::<MemosAgent>()).ok_or_else(|| anyhow::anyhow!("MemosAgent not found"))?;
        let candidate_points = memos_agent.recall(text).await?;
        if candidate_points.is_empty() {
            return Ok(format!("关于“{}”，我好像没什么印象...", text));
        }
        if let Some(reranker) = &self.reranker {
            println!("[RecallExpert] Re-ranking candidates...");
            let documents_to_rank: Vec<DocumentToRank> = candidate_points.iter()
                .filter_map(|p| p.payload.get("content").and_then(|v| v.as_str()).map(|s| DocumentToRank { text: s }))
                .collect();
            let rerank_request = ReRankRequest { query: text, documents: documents_to_rank };
            let strategy = ReRankStrategy::ValidateTopOne { threshold: 0.1 };
            let final_results = reranker.rank(rerank_request, strategy).await?;
            if let Some(top_doc) = final_results.get(0) {
                Ok(top_doc.text.clone())
            } else {
                let summary: Vec<String> = candidate_points.iter().take(3)
                    .filter_map(|p| p.payload.get("content").and_then(|v| v.as_str()).map(|s| format!("- {}", s)))
                    .collect();
                Ok(format!("关于“{}”，我没有找到直接答案，但发现一些可能相关的内容：\n{}", text, summary.join("\n")))
            }
        } else {
            println!("[RecallExpert] Skipping re-ranking.");
            let top_point = candidate_points.get(0).unwrap();
            let content = top_point.payload.get("content")
                                   .and_then(|v| v.as_str())
                                   .map_or("".to_string(), |v| v.to_string());
            Ok(content)
        }
    }

    async fn handle_modify(&self, text: &str) -> Result<String, anyhow::Error> {
        println!("[ModifyExpert-Phase1] Received request: '{}'", text);
        let memos_agent = self.agents.iter()
            .find_map(|a| a.as_any().downcast_ref::<MemosAgent>())
            .ok_or_else(|| anyhow::anyhow!("MemosAgent not found"))?;
        let candidate_points = memos_agent.recall(text).await?;
        if let Some(top_point) = candidate_points.get(0) {
            let memory_id = top_point.id.as_ref().and_then(|id| match &id.point_id_options {
                Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => Some(*num as i64),
                _ => None,
            }).ok_or_else(|| anyhow::anyhow!("Found a point without a numeric ID"))?;
            let content = top_point.payload.get("content")
                .and_then(|v| v.as_str())
                .map_or("无法解析内容".to_string(), |v| v.to_string());
            let pending_action = PendingAction {
                action_type: PendingActionType::ModifyConfirmation { 
                    memory_id, 
                    original_content: content.clone()
                },
                original_user_request: text.to_string(),
            };
            *self.pending_action.lock().unwrap() = Some(pending_action);
            println!("[Orchestrator] Pending action set: ModifyConfirmation for ID {}", memory_id);
            let response_text = format!("您是想修改这条记忆吗？\n\n---\n{}\n---", content);
            Ok(response_text)
        } else {
            Ok("抱歉，我没有找到与您描述相关的记忆。".to_string())
        }
    }

    async fn handle_delete(&self, text: &str) -> Result<String, anyhow::Error> {
        println!("[DeleteExpert-Phase1] Received request: '{}'", text);
        let memos_agent = self.agents.iter()
            .find_map(|a| a.as_any().downcast_ref::<MemosAgent>())
            .ok_or_else(|| anyhow::anyhow!("MemosAgent not found"))?;
        let candidate_points = memos_agent.recall(text).await?;
        if let Some(top_point) = candidate_points.get(0) {
            let memory_id = top_point.id.as_ref().and_then(|id| match &id.point_id_options {
                Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => Some(*num as i64),
                _ => None,
            }).ok_or_else(|| anyhow::anyhow!("Found a point without a numeric ID"))?;
            let content = top_point.payload.get("content")
                .and_then(|v| v.as_str())
                .map_or("无法解析内容".to_string(), |v| v.to_string());
            let pending_action = PendingAction {
                action_type: PendingActionType::DeleteConfirmation { 
                    memory_id, 
                    content_to_delete: content.clone() 
                },
                original_user_request: text.to_string(),
            };
            *self.pending_action.lock().unwrap() = Some(pending_action);
            println!("[Orchestrator] Pending action set: DeleteConfirmation for ID {}", memory_id);
            let response_text = format!("您确定要删除这条记忆吗？\n\n---\n{}\n---", content);
            Ok(response_text)
        } else {
            Ok("抱歉，我没有找到与您描述相关的记忆可以删除。".to_string())
        }
    }
    
    async fn handle_confirmation(&self, text: &str) -> Result<String, anyhow::Error> {
        let pending_action = self.pending_action.lock().unwrap().take();
        if let Some(action) = pending_action {
            match confirmation_expert::parse_confirmation(text) {
                confirmation_expert::ConfirmationDecision::Affirm => {
                    println!("[ConfirmationExpert] User affirmed. Executing pending action.");
                    self.execute_pending_action(action).await
                }
                confirmation_expert::ConfirmationDecision::Deny => {
                    println!("[ConfirmationExpert] User denied. Cancelling pending action.");
                    Ok("好的，已取消操作。".to_string())
                }
                confirmation_expert::ConfirmationDecision::Unclear => {
                    println!("[ConfirmationExpert] User response unclear. Re-instating pending action.");
                    *self.pending_action.lock().unwrap() = Some(action);
                    Ok("抱歉，我没太明白。请明确回复“是”或“否”。".to_string())
                }
            }
        } else {
            Ok("嗯？我们刚才有在讨论什么需要确认的事情吗？".to_string())
        }
    }

    async fn execute_pending_action(&self, action: PendingAction) -> Result<String, anyhow::Error> {
        match action.action_type {
            PendingActionType::ModifyConfirmation { memory_id, original_content } => {
                println!("[ModifyExpert-Phase2] Executing modification for ID: {}", memory_id);
                let messages = modify_expert::get_text_modification_prompt(&original_content, &action.original_user_request);
                let gbnf_schema = modify_expert::get_text_modification_gbnf_schema();
                let request_body = json!({ "messages": messages, "temperature": 0.0, "grammar": gbnf_schema });
                let chat_url = format!("{}/v1/chat/completions", self.llm_config.llm_url);
                let response = self.llm_config.client.post(&chat_url).json(&request_body).send().await?;
                #[derive(Deserialize)] struct ChatChoice { message: ChatMessageContent }
                #[derive(Deserialize)] struct ChatMessageContent { content: String }
                #[derive(Deserialize)] struct ChatCompletionResponse { choices: Vec<ChatChoice> }
                let chat_response: ChatCompletionResponse = response.json().await?;
                let content_str = chat_response.choices.get(0).map(|c| c.message.content.trim())
                    .ok_or_else(|| anyhow::anyhow!("Modify LLM response is empty"))?;
                let modified_text_obj: modify_expert::ModifiedText = serde_json::from_str(content_str)?;
                let new_content = &modified_text_obj.modified_text;
                println!("[ModifyExpert-Phase2] LLM generated new text: '{}'", new_content);
                let memos_agent = self.agents.iter()
                    .find_map(|a| a.as_any().downcast_ref::<MemosAgent>())
                    .ok_or_else(|| anyhow::anyhow!("MemosAgent not found"))?;
                memos_agent.update(memory_id, new_content).await?;
                Ok("好的，我已经更新了这条记忆。".to_string())
            }
            PendingActionType::DeleteConfirmation { memory_id, .. } => {
                println!("[DeleteExpert-Phase2] Executing deletion for ID: {}", memory_id);
                let memos_agent = self.agents.iter()
                    .find_map(|a| a.as_any().downcast_ref::<MemosAgent>())
                    .ok_or_else(|| anyhow::anyhow!("MemosAgent not found"))?;
                memos_agent.delete(memory_id).await?;
                Ok("好的，我已经删除了这条记忆。".to_string())
            }
        }
    }

    pub async fn dispatch(&self, command: &Command) -> Result<Response, anyhow::Error> {
        match command {
            Command::ProcessText(text) => {
                // 【逻辑修正】对于确认指令，我们应该跳过复杂的预处理流程
                if confirmation_expert::parse_confirmation(text) != confirmation_expert::ConfirmationDecision::Unclear {
                    let final_response = self.handle_confirmation(text).await?;
                    let mut history = self.conversation_history.lock().unwrap();
                    history.push(format!("User: {}", text));
                    history.push(format!("Assistant: {}", final_response));
                    return Ok(Response::Text(final_response));
                }

                // 任务链阶段 1: 消歧义 (预处理)
                let history_guard = self.conversation_history.lock().unwrap();
                println!("[Orchestrator] Step 1: Disambiguating user input: '{}'", text);
                let dis_messages = disambiguation::get_disambiguation_prompt(text, &history_guard);
                let dis_gbnf = disambiguation::get_disambiguation_gbnf_schema();
                let chat_url = format!("{}/v1/chat/completions", self.llm_config.llm_url);
                let dis_request_body = json!({ "messages": dis_messages, "grammar": dis_gbnf, "temperature": 0.0 });
                let dis_response = self.llm_config.client.post(&chat_url).json(&dis_request_body).send().await?;
                if !dis_response.status().is_success() {
                    return Err(anyhow::anyhow!("DisambiguationExpert LLM call failed: {}", dis_response.text().await?));
                }
                #[derive(Deserialize)] struct ChatChoice { message: ChatMessageContent }
                #[derive(Deserialize)] struct ChatMessageContent { content: String }
                #[derive(Deserialize)] struct ChatCompletionResponse { choices: Vec<ChatChoice> }
                let dis_chat_response: ChatCompletionResponse = dis_response.json().await?;
                let dis_content_str = dis_chat_response.choices.get(0).map(|c| c.message.content.trim())
                    .ok_or_else(|| anyhow::anyhow!("Disambiguation LLM response is empty"))?;
                let disambiguated_obj: disambiguation::DisambiguatedText = serde_json::from_str(dis_content_str)?;
                let disambiguated_text = &disambiguated_obj.disambiguated_text;
                println!("[Orchestrator] Step 1 Result: Disambiguated text is: '{}'", disambiguated_text);

                // ... (后续的路由和执行阶段代码完全不变) ...
                println!("[Orchestrator] Step 2: Routing with disambiguated text.");
                let router_messages = router::get_routing_prompt(disambiguated_text, &history_guard);
                drop(history_guard);
                let router_gbnf = router::get_routing_gbnf_schema();
                let router_request_body = json!({ "messages": router_messages, "grammar": router_gbnf, "temperature": 0.0 });
                let router_response = self.llm_config.client.post(&chat_url).json(&router_request_body).send().await?;
                if !router_response.status().is_success() {
                    return Err(anyhow::anyhow!("Router LLM call failed: {}", router_response.text().await?));
                }
                let router_chat_response: ChatCompletionResponse = router_response.json().await?;
                let router_content_str = router_chat_response.choices.get(0).map(|c| c.message.content.trim())
                    .ok_or_else(|| anyhow::anyhow!("Router LLM response is empty"))?;
                println!("[Router] Decision JSON: {}", router_content_str);
                let decision: RoutingDecision = serde_json::from_str(router_content_str)?;
                
                let mut pending_action_guard = self.pending_action.lock().unwrap();
                if decision.tool_to_call != ToolToCall::ConfirmationTool {
                    if pending_action_guard.is_some() {
                        println!("[Orchestrator] New command received, cancelling previous pending action.");
                        *pending_action_guard = None;
                    }
                }
                drop(pending_action_guard);

                println!("[Orchestrator] Step 3: Executing expert with disambiguated text.");
                let final_response = match decision.tool_to_call {
                    ToolToCall::SaveTool => self.handle_save(disambiguated_text).await?,
                    ToolToCall::RecallTool => self.handle_recall(disambiguated_text).await?,
                    ToolToCall::ModifyTool => self.handle_modify(disambiguated_text).await?,
                    ToolToCall::DeleteTool => self.handle_delete(text).await?,
                    ToolToCall::ConfirmationTool => self.handle_confirmation(text).await?,
                    ToolToCall::MixedTool => {
                        let save_response = self.handle_save(disambiguated_text).await?;
                        let recall_response = self.handle_recall(disambiguated_text).await?;
                        format!("{}\n{}", save_response, recall_response)
                    },
                    ToolToCall::NoTool => "抱歉，我不太明白您的意思，可以换个方式说吗？".to_string(),
                };

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
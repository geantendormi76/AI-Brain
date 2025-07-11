// backend/orchestrator/src/lib.rs
// 【V7.0 - 最终完整版】

mod experts;
// preprocessors 模块已无用，可以安全移除或保留为空
// mod preprocessors; 

use agent_memos::MemosAgent;
use memos_core::{Agent, Command, Response};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, Mutex};
use std::fs::OpenOptions;
use std::io::Write;
use qdrant_client::qdrant::point_id;

use experts::memos_agent::{
    router,
    save_expert,
    modify_expert,
    re_ranker::{ReRanker, ReRankRequest, DocumentToRank, ReRankStrategy},
};

// --- 核心状态与响应结构定义 (已确认无误) ---
#[derive(Debug, Clone)]
pub enum ContextualAction {
    Save { memory_id: i64 },
    Recall { memory_id: i64, content: String },
}
#[derive(Debug, Clone)]
pub struct InteractionContext {
    pub last_action: ContextualAction,
}
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
#[derive(Deserialize, Debug, PartialEq)]
enum PendingActionDecision {
    Affirm,
    Deny,
    ProvideInfo,
    Unrelated,
}
#[derive(Deserialize, Debug)]
struct PendingActionResponse {
    decision: PendingActionDecision,
    new_information: Option<String>,
}

// --- LLMConfig 与 Orchestrator 结构体定义 (已确认无误) ---
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
    last_interaction_context: Arc<Mutex<Option<InteractionContext>>>,
    last_full_interaction: Arc<Mutex<Option<(String, String)>>>,
}

// --- Orchestrator 实现 ---
impl Orchestrator {
    pub fn new(agents: Vec<Box<dyn Agent>>, llm_url: &str, reranker_llm_url: Option<&str>) -> Self {
        println!("[Orchestrator] V7.0-Final Initializing...");
        let reranker = reranker_llm_url.map(ReRanker::new);
        if reranker.is_some() {
            println!("[Orchestrator] ReRanker is ENABLED.");
        } else {
            println!("[Orchestrator] ReRanker is DISABLED.");
        }
        Self {
            agents,
            llm_config: LLMConfig::new(llm_url),
            conversation_history: Arc::new(Mutex::new(Vec::new())),
            reranker,
            pending_action: Arc::new(Mutex::new(None)),
            last_interaction_context: Arc::new(Mutex::new(None)),
            last_full_interaction: Arc::new(Mutex::new(None)),
        }
    }

    // =================================================================
    // == 核心 `dispatch` 方法，【V7.0 最终版】 ==
    // =================================================================
    pub async fn dispatch(&self, command: &Command) -> Result<Response, anyhow::Error> {
        match command {
            Command::ProcessText(text) => {
                // 阶段一：状态优先检查
                if let Some(pending_action) = self.pending_action.lock().unwrap().take() {
                    println!("[Orchestrator-State] Detected pending action. Routing to dedicated handler.");
                    let final_response = self.handle_pending_action_response(pending_action, text).await?;
                    self.update_history_and_cache(text, &final_response);
                    return Ok(Response::Text(final_response));
                }

                // 阶段二：确定性前置分流
                let chit_chat_keywords = ["天气", "笑话", "你好"];
                if chit_chat_keywords.iter().any(|&kw| text.contains(kw)) {
                    println!("[Orchestrator-Heuristic] Detected Chit-Chat via keywords.");
                    let response = "抱歉，我暂时无法处理这类请求，我的主要任务是记录和查询记忆。".to_string();
                    self.update_history_and_cache(text, &response);
                    return Ok(Response::Text(response));
                }

                let correction_keywords = ["不对", "错了", "不是", "应该是"];
                if correction_keywords.iter().any(|&kw| text.starts_with(kw)) {
                    if let Some(context) = self.last_interaction_context.lock().unwrap().take() {
                        if let ContextualAction::Save { memory_id } = context.last_action {
                            println!("[Orchestrator-DST] Contextual Route: Detected correction for last saved memory ID: {}", memory_id);
                            let final_response = self.handle_contextual_modify(memory_id, text).await?;
                            self.update_history_and_cache(text, &final_response);
                            return Ok(Response::Text(final_response));
                        }
                    }
                }

                let contextual_delete_keywords = ["删除", "忘掉", "去掉", "移除"];
                let pronoun_keywords = ["这条", "那条", "这个", "它"];
                if contextual_delete_keywords.iter().any(|&kw| text.contains(kw)) && pronoun_keywords.iter().any(|&kw| text.contains(kw)) {
                     if let Some(context) = self.last_interaction_context.lock().unwrap().take() {
                         let memory_id = match context.last_action {
                             ContextualAction::Save { memory_id } => memory_id,
                             ContextualAction::Recall { memory_id, .. } => memory_id,
                         };
                         println!("[Orchestrator-DST] Contextual Route: Detected contextual delete for memory ID: {}", memory_id);
                         let final_response = self.handle_delete_by_id(memory_id).await?;
                         self.update_history_and_cache(text, &final_response);
                         return Ok(Response::Text(final_response));
                     }
                }

                let modify_keywords = ["修改", "更新", "编辑"];
                if modify_keywords.iter().any(|&kw| text.contains(kw)) {
                    println!("[Orchestrator-Heuristic] Detected ModifyTool via keywords.");
                    let final_response = self.handle_modify(text).await?;
                    self.update_history_and_cache(text, &final_response);
                    return Ok(Response::Text(final_response));
                }
                let delete_keywords = ["删除", "忘掉", "去掉", "移除"];
                if delete_keywords.iter().any(|&kw| text.contains(kw)) {
                     println!("[Orchestrator-Heuristic] Detected DeleteTool via keywords.");
                     let final_response = self.handle_delete(text).await?;
                     self.update_history_and_cache(text, &final_response);
                     return Ok(Response::Text(final_response));
                }

                // 阶段三：LLM 兜底路由
                println!("[Orchestrator] No heuristic matched or state triggered. Downgrading to LLM Router.");
                let final_response = self.route_with_llm(text).await?;
                self.update_history_and_cache(text, &final_response);
                Ok(Response::Text(final_response))
            }
        }
    }

    // --- 新增的、高内聚的函数 ---

    async fn handle_pending_action_response(&self, pending_action: PendingAction, user_response: &str) -> Result<String, anyhow::Error> {
        let original_request_text = &pending_action.original_user_request;
        let system_prompt = format!(
            r#"你是一个对话理解专家。AI助理刚刚问用户是否要进行一个关于“{}”的操作，用户的回复是“{}”。你的任务是判断用户的回复属于以下哪一类：
1.  `Affirm`: 用户明确同意。
2.  `Deny`: 用户明确拒绝。
3.  `ProvideInfo`: 用户在同意的同时，提供了完成操作所需的额外信息。例如“是的，改成xxx”。
4.  `Unrelated`: 用户的回复与当前确认无关。
你的输出必须是严格的JSON对象，格式如下：{{"decision": "...", "new_information": "..."}}
- "decision" 必须是 "Affirm", "Deny", "ProvideInfo", "Unrelated" 之一。
- "new_information" 只在decision是"ProvideInfo"时填写，否则为null。"#,
            original_request_text, user_response
        );
        let gbnf_schema = r#"root ::= "{" ws "\"decision\"" ws ":" ws ("\"Affirm\"" | "\"Deny\"" | "\"ProvideInfo\"" | "\"Unrelated\"") ws "," ws "\"new_information\"" ws ":" ws (string | "null") ws "}"
string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}) )* "\""
ws ::= ([ \t\n\r])*"#;

        let messages = vec![json!({"role": "system", "content": system_prompt})];
        let request_body = json!({ "messages": messages, "temperature": 0.0, "grammar": gbnf_schema });
        let chat_url = format!("{}/v1/chat/completions", self.llm_config.llm_url);
        let response = self.llm_config.client.post(&chat_url).json(&request_body).send().await?;
        let response_text = response.text().await?;
        let parsed_response: PendingActionResponse = serde_json::from_str(&response_text)
            .map_err(|e| anyhow::anyhow!("Failed to parse pending action response: {}. Raw text: {}", e, response_text))?;

        match parsed_response.decision {
            PendingActionDecision::Affirm => self.execute_pending_action(pending_action, None).await,
            PendingActionDecision::ProvideInfo => self.execute_pending_action(pending_action, parsed_response.new_information).await,
            PendingActionDecision::Deny => {
                println!("[Orchestrator-State] User denied pending action.");
                Ok("好的，已取消操作。".to_string())
            }
            PendingActionDecision::Unrelated => {
                 println!("[Orchestrator-State] User response is unrelated. Re-instating pending action.");
                 *self.pending_action.lock().unwrap() = Some(pending_action);
                 Ok("抱歉，我没太明白您的意思。我们正在讨论修改或删除一个记忆，请问您是同意还是取消？".to_string())
            }
        }
    }

    async fn route_with_llm(&self, text: &str) -> Result<String, anyhow::Error> {
        let history_clone = self.conversation_history.lock().unwrap().clone();
        let decision = router::run(&self.llm_config.client, &self.llm_config.llm_url, text, &history_clone).await?;
        
        println!("[Orchestrator] LLM-based execution decision: {:?}", decision.tool_to_call);
        match decision.tool_to_call {
            router::ToolToCall::SaveTool => self.handle_save(text).await,
            router::ToolToCall::RecallTool => self.handle_recall(text).await,
            _ => Ok("抱歉，我不太明白您的意思，可以换个方式说吗？".to_string()),
        }
    }

    async fn execute_pending_action(&self, action: PendingAction, new_information: Option<String>) -> Result<String, anyhow::Error> {
        let memos_agent = self.get_memos_agent()?;
        match action.action_type {
            PendingActionType::ModifyConfirmation { memory_id, original_content } => {
                println!("[ModifyExpert-Phase2] Executing modification for ID: {}", memory_id);
                let modification_request = new_information.unwrap_or(action.original_user_request);
                let new_content = modify_expert::run(&self.llm_config.client, &self.llm_config.llm_url, &original_content, &modification_request).await?;
                memos_agent.update(memory_id, &new_content).await?;
                Ok(format!("好的，我已经将记忆更新为：{}", new_content))
            }
            PendingActionType::DeleteConfirmation { memory_id, .. } => {
                println!("[DeleteExpert-Phase2] Executing deletion for ID: {}", memory_id);
                memos_agent.delete(memory_id).await?;
                Ok("好的，我已经删除了这条记忆。".to_string())
            }
        }
    }

    // --- 重构后的 handle_* 方法 ---

    async fn handle_save(&self, text: &str) -> Result<String, anyhow::Error> {
        let memos_agent = self.get_memos_agent()?;
        let extracted_fact = save_expert::run(&self.llm_config.client, &self.llm_config.llm_url, text).await?;
        let new_memory_id = memos_agent.save(&extracted_fact.fact, extracted_fact.metadata).await?;
        let context = InteractionContext { last_action: ContextualAction::Save { memory_id: new_memory_id } };
        *self.last_interaction_context.lock().unwrap() = Some(context);
        Ok("好的，已经记下了。".to_string())
    }

    async fn handle_recall(&self, text: &str) -> Result<String, anyhow::Error> {
        let memos_agent = self.get_memos_agent()?;
        let candidate_points = memos_agent.recall(text).await?;
        if candidate_points.is_empty() { return Ok(format!("关于“{}”，我好像没什么印象...", text)); }

        let final_content = if let Some(reranker) = &self.reranker {
            let contents: Vec<String> = candidate_points.iter().filter_map(|p| p.payload.get("content")?.as_str().map(String::from)).collect();
            let documents_to_rank: Vec<DocumentToRank> = contents.iter().map(|s| DocumentToRank { text: s }).collect();
            let rerank_request = ReRankRequest { query: text, documents: documents_to_rank };
            let final_results = reranker.rank(rerank_request, ReRankStrategy::ValidateTopOne { threshold: 0.1 }).await?;
            final_results.get(0).map(|d| d.text.clone()).unwrap_or_else(|| {
                let summary: Vec<String> = candidate_points.iter().take(3).filter_map(|p| p.payload.get("content")?.as_str().map(|s| format!("- {}", s))).collect();
                format!("关于“{}”，我没有找到直接答案，但发现一些可能相关的内容：\n{}", text, summary.join("\n"))
            })
        } else {
            candidate_points.get(0).unwrap().payload.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string()
        };

        if let Some(top_point) = candidate_points.get(0) {
            if let Some(point_id::PointIdOptions::Num(id)) = top_point.id.as_ref().and_then(|p| p.point_id_options.as_ref()) {
                let context = InteractionContext { last_action: ContextualAction::Recall { memory_id: *id as i64, content: final_content.clone() } };
                *self.last_interaction_context.lock().unwrap() = Some(context);
            }
        }
        Ok(final_content)
    }

    async fn handle_modify(&self, text: &str) -> Result<String, anyhow::Error> {
        let memos_agent = self.get_memos_agent()?;
        let candidate_points = memos_agent.recall(text).await?;
        if let Some(top_point) = candidate_points.get(0) {
            let memory_id = top_point.id.as_ref().and_then(|id| match &id.point_id_options {
                Some(point_id::PointIdOptions::Num(num)) => Some(*num as i64), _ => None,
            }).ok_or_else(|| anyhow::anyhow!("Found a point without a numeric ID"))?;
            let content = top_point.payload.get("content")
                .and_then(|v| v.as_str())
                .map_or(String::new(), |s| s.to_string());
            let pending_action = PendingAction {
                action_type: PendingActionType::ModifyConfirmation { memory_id, original_content: content.clone() },
                original_user_request: text.to_string(),
            };
            *self.pending_action.lock().unwrap() = Some(pending_action);
            Ok(format!("您是想修改这条记忆吗？\n\n---\n{}\n---", content))
        } else {
            Ok("抱歉，我没有找到与您描述相关的记忆。".to_string())
        }
    }

    async fn handle_delete(&self, text: &str) -> Result<String, anyhow::Error> {
        let memos_agent = self.get_memos_agent()?;
        let candidate_points = memos_agent.recall(text).await?;
        if let Some(top_point) = candidate_points.get(0) {
            let memory_id = top_point.id.as_ref().and_then(|id| match &id.point_id_options {
                Some(point_id::PointIdOptions::Num(num)) => Some(*num as i64), _ => None,
            }).ok_or_else(|| anyhow::anyhow!("Found a point without a numeric ID"))?;
            let content = top_point.payload.get("content")
                .and_then(|v| v.as_str())
                .map_or(String::new(), |s| s.to_string());
            let pending_action = PendingAction {
                action_type: PendingActionType::DeleteConfirmation { memory_id, content_to_delete: content.clone() },
                original_user_request: text.to_string(),
            };
            *self.pending_action.lock().unwrap() = Some(pending_action);
            Ok(format!("您确定要删除这条记忆吗？\n\n---\n{}\n---", content))
        } else {
            Ok("抱歉，我没有找到与您描述相关的记忆可以删除。".to_string())
        }
    }
    
    async fn handle_delete_by_id(&self, memory_id: i64) -> Result<String, anyhow::Error> {
        let memos_agent = self.get_memos_agent()?;
        memos_agent.delete(memory_id).await?;
        *self.last_interaction_context.lock().unwrap() = None;
        Ok("好的，我已经删除了那条记忆。".to_string())
    }

    async fn handle_contextual_modify(&self, memory_id: i64, correction_text: &str) -> Result<String, anyhow::Error> {
        let memos_agent = self.get_memos_agent()?;
        let original_content = memos_agent.get_by_id(memory_id).await?.ok_or_else(|| anyhow::anyhow!("Memory ID not found"))?;
        let new_content = modify_expert::run(&self.llm_config.client, &self.llm_config.llm_url, &original_content, correction_text).await?;
        memos_agent.update(memory_id, &new_content).await?;
        Ok(format!("好的，我已经将记忆更新为：{}", new_content))
    }

    // --- 辅助函数 (保持不变) ---
    fn update_history_and_cache(&self, text: &str, final_response: &str) {
        let mut history = self.conversation_history.lock().unwrap();
        history.push(format!("User: {}", text));
        history.push(format!("Assistant: {}", final_response));
        const MAX_HISTORY_SIZE: usize = 10;
        let current_len = history.len(); // 先计算长度，释放不可变借用
        if current_len > MAX_HISTORY_SIZE {
            let drain_count = current_len - MAX_HISTORY_SIZE;
            history.drain(..drain_count); // 然后再进行可变借用
        }
        *self.last_full_interaction.lock().unwrap() = Some((text.to_string(), final_response.to_string()));
    }
    fn get_memos_agent(&self) -> Result<&MemosAgent, anyhow::Error> {
        self.agents.iter().find_map(|a| a.as_any().downcast_ref::<MemosAgent>()).ok_or_else(|| anyhow::anyhow!("MemosAgent not found"))
    }
    pub fn handle_feedback(&self) {
        if let Some((user_input, assistant_response)) = self.last_full_interaction.lock().unwrap().as_ref() {
            let feedback_data = json!({
                "instruction": user_input,
                "input": "",
                "output": format!("// TODO: Add the expected correct tool call JSON here.\n// Assistant's wrong response was: {}", assistant_response),
            });
            if let Ok(mut file) = OpenOptions::new().create(true).write(true).append(true).open("feedback.jsonl") {
                if let Ok(line) = serde_json::to_string(&feedback_data) {
                    if writeln!(file, "{}", line).is_ok() { println!("[Feedback] Successfully saved feedback to feedback.jsonl"); }
                }
            }
        } else { println!("[Feedback] No last interaction found to provide feedback on."); }
    }
}
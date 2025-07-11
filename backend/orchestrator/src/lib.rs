// backend/orchestrator/src/lib.rs
// 【黄金标准对齐版 - V1】

// 声明模块
mod experts;
mod preprocessors;

// 导入所需工具
use agent_memos::MemosAgent;
use memos_core::{Agent, Command, Response};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, Mutex};
use std::fs::OpenOptions;
use std::io::Write;

// 从专家模块中导入所需定义
use experts::memos_agent::{
    router::{self, RoutingDecision, ToolToCall},
    save_expert,
    modify_expert,
    confirmation_expert,
    re_ranker::{ReRanker, ReRankRequest, DocumentToRank, ReRankStrategy},
};

// =================================================================
// == 1. 对话状态追踪 (DST) 与 待处理动作 (PendingAction) 的核心定义 ==
// =================================================================
// 这些结构体是实现精确上下文理解和安全操作的关键

/// 定义了上一次交互可能是什么类型的操作
#[derive(Debug, Clone)]
pub enum ContextualAction {
    /// 上一次是保存操作，我们记录了被保存记忆的ID
    Save { memory_id: i64 },
    /// 上一次是召回操作，我们记录了被召回记忆的ID和内容
    Recall { memory_id: i64, content: String },
}

/// 定义了“短期上下文”这个小记事本本身
#[derive(Debug, Clone)]
pub struct InteractionContext {
    pub last_action: ContextualAction,
}

/// 定义了需要用户确认的待处理操作的类型
#[derive(Debug, Clone)]
pub enum PendingActionType {
    ModifyConfirmation { memory_id: i64, original_content: String },
    DeleteConfirmation { memory_id: i64, content_to_delete: String },
}

/// 定义了待处理操作的完整结构
#[derive(Debug, Clone)]
pub struct PendingAction {
    pub action_type: PendingActionType,
    pub original_user_request: String,
}
// ======================= 核心定义结束 =======================

// LLM配置，保持不变
pub struct LLMConfig {
    client: Client,
    llm_url: String,
}

impl LLMConfig {
    pub fn new(llm_url: &str) -> Self {
        Self { client: Client::new(), llm_url: llm_url.to_string() }
    }
}

// =================================================================
// == 2. Orchestrator 结构体的“黄金标准”定义 ==
// =================================================================
// 我们加入了新的状态变量来支持DST和安全操作

pub struct Orchestrator {
    agents: Vec<Box<dyn Agent>>,
    llm_config: LLMConfig,
    conversation_history: Arc<Mutex<Vec<String>>>,
    reranker: Option<ReRanker>,
    // 新增的状态变量
    pending_action: Arc<Mutex<Option<PendingAction>>>,
    last_interaction_context: Arc<Mutex<Option<InteractionContext>>>,
    last_full_interaction: Arc<Mutex<Option<(String, String)>>>,
}

// ======================= 结构体定义结束 =======================


impl Orchestrator {
    // Orchestrator的构造函数，初始化所有状态
    pub fn new(agents: Vec<Box<dyn Agent>>, llm_url: &str, reranker_llm_url: Option<&str>) -> Self {
        println!("[Orchestrator] V6.1-Gold-Standard Initializing...");
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
            // 初始化新的状态变量
            pending_action: Arc::new(Mutex::new(None)),
            last_interaction_context: Arc::new(Mutex::new(None)),
            last_full_interaction: Arc::new(Mutex::new(None)),
        }
    }

    // =================================================================
    // == 3. 各种 handle_* 和辅助方法 (大部分是新增或重写的) ==
    // =================================================================

    async fn handle_save(&self, text: &str) -> Result<String, anyhow::Error> {
        let memos_agent = self.get_memos_agent()?;
        println!("[SaveExpert] Extracting fact from raw text: '{}'", text);
        
        let messages = save_expert::get_fact_extraction_prompt(text);
        let gbnf_schema = save_expert::get_fact_extraction_gbnf_schema();
        let request_body = json!({ "messages": messages, "temperature": 0.0, "grammar": gbnf_schema });
        let chat_url = format!("{}/v1/chat/completions", self.llm_config.llm_url);
        
        #[derive(Deserialize)] struct ChatChoice { message: ChatMessageContent }
        #[derive(Deserialize)] struct ChatMessageContent { content: String }
        #[derive(Deserialize)] struct ChatCompletionResponse { choices: Vec<ChatChoice> }
        
        let response = self.llm_config.client.post(&chat_url).json(&request_body).send().await?;
        let chat_response: ChatCompletionResponse = response.json().await?;
        let content_str = chat_response.choices.get(0).map(|c| c.message.content.trim()).unwrap_or("{}");
        let extracted_fact_obj: save_expert::ExtractedFact = serde_json::from_str(content_str)?;
        let fact_to_save = &extracted_fact_obj.fact;
        
        println!("[SaveExpert] Fact to save: '{}'", fact_to_save);
        
        let new_memory_id = memos_agent.save(fact_to_save, extracted_fact_obj.metadata).await?;

        // 更新DST
        let context = InteractionContext {
            last_action: ContextualAction::Save { memory_id: new_memory_id },
        };
        *self.last_interaction_context.lock().unwrap() = Some(context);
        println!("[Orchestrator-DST] Updated context: Last action was Save with ID {}", new_memory_id);

        Ok("好的，已经记下了。".to_string())
    }

    async fn handle_recall(&self, text: &str) -> Result<String, anyhow::Error> {
        println!("[RecallExpert] Received recall request for: '{}'", text);
        let memos_agent = self.get_memos_agent()?;
        let candidate_points = memos_agent.recall(text).await?;

        if candidate_points.is_empty() {
            return Ok(format!("关于“{}”，我好像没什么印象...", text));
        }

        if let Some(reranker) = &self.reranker {
            println!("[RecallExpert] Re-ranking candidates...");

            // 【核心修正】: 我们需要让 documents_to_rank 活得和 rerank_request 一样久
            // 所以我们先提取出所有内容的 String 副本
            let contents: Vec<String> = candidate_points.iter()
                .filter_map(|p| p.payload.get("content").and_then(|v| v.as_str()).map(|s| s.to_string()))
                .collect();
            
            // 然后，我们创建 DocumentToRank 的引用，其生命周期与 contents 绑定
            let documents_to_rank: Vec<DocumentToRank> = contents.iter()
                .map(|s| DocumentToRank { text: s.as_str() })
                .collect();

            // 最后，我们创建请求，其生命周期与 text 和 documents_to_rank 绑定
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
        let memos_agent = self.get_memos_agent()?;
        
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
            
            Ok(format!("您是想修改这条记忆吗？\n\n---\n{}\n---", content))
        } else {
            Ok("抱歉，我没有找到与您描述相关的记忆。".to_string())
        }
    }

    async fn handle_delete(&self, text: &str) -> Result<String, anyhow::Error> {
        println!("[DeleteExpert-Phase1] Received request: '{}'", text);
        let memos_agent = self.get_memos_agent()?;
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
            Ok(format!("您确定要删除这条记忆吗？\n\n---\n{}\n---", content))
        } else {
            Ok("抱歉，我没有找到与您描述相关的记忆可以删除。".to_string())
        }
    }
    
    
    async fn handle_delete_by_id(&self, memory_id: i64) -> Result<String, anyhow::Error> {
        let memos_agent = self.get_memos_agent()?;
        
        // 直接调用 agent 的 delete 方法
        memos_agent.delete(memory_id).await?;
        
        println!("[Orchestrator-DST] Deleted memory with ID: {}", memory_id);
        
        // 清空上下文，因为被引用的记忆已经不存在了
        *self.last_interaction_context.lock().unwrap() = None;
        
        Ok("好的，我已经删除了那条记忆。".to_string())
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
        let memos_agent = self.get_memos_agent()?;
        match action.action_type {
            PendingActionType::ModifyConfirmation { memory_id, original_content } => {
                println!("[ModifyExpert-Phase2] Executing modification for ID: {}", memory_id);
                #[derive(Deserialize)] struct ChatChoice { message: ChatMessageContent }
                #[derive(Deserialize)] struct ChatMessageContent { content: String }
                #[derive(Deserialize)] struct ChatCompletionResponse { choices: Vec<ChatChoice> }

                let messages = modify_expert::get_text_modification_prompt(&original_content, &action.original_user_request);
                let gbnf_schema = modify_expert::get_text_modification_gbnf_schema();
                let request_body = json!({ "messages": messages, "temperature": 0.0, "grammar": gbnf_schema });
                let chat_url = format!("{}/v1/chat/completions", self.llm_config.llm_url);
                let response = self.llm_config.client.post(&chat_url).json(&request_body).send().await?;
                
                let chat_response: ChatCompletionResponse = response.json().await?;
                let content_str = chat_response.choices.get(0).map(|c| c.message.content.trim())
                    .ok_or_else(|| anyhow::anyhow!("Modify LLM response is empty"))?;
                
                let modified_text_obj: modify_expert::ModifiedText = serde_json::from_str(content_str)?;
                let new_content = &modified_text_obj.modified_text;
                
                println!("[ModifyExpert-Phase2] LLM generated new text: '{}'", new_content);
                
                memos_agent.update(memory_id, new_content).await?;
                Ok("好的，我已经更新了这条记忆。".to_string())
            }
            PendingActionType::DeleteConfirmation { memory_id, .. } => {
                println!("[DeleteExpert-Phase2] Executing deletion for ID: {}", memory_id);
                memos_agent.delete(memory_id).await?;
                Ok("好的，我已经删除了这条记忆。".to_string())
            }
        }
    }

    async fn handle_contextual_modify(&self, memory_id: i64, correction_text: &str) -> Result<String, anyhow::Error> {
        println!("[Orchestrator-DST] Handling contextual modification for memory ID: {}", memory_id);
        let memos_agent = self.get_memos_agent()?;
        let original_content = match memos_agent.get_by_id(memory_id).await? {
            Some(content) => content,
            None => return Ok(format!("抱歉，我找不到刚才那条ID为 {} 的记忆了，无法为您修正。", memory_id)),
        };
        
        println!("[Orchestrator-DST] Found original content for ID {}: '{}'", memory_id, original_content);

        #[derive(Deserialize)] struct ChatChoice { message: ChatMessageContent }
        #[derive(Deserialize)] struct ChatMessageContent { content: String }
        #[derive(Deserialize)] struct ChatCompletionResponse { choices: Vec<ChatChoice> }

        let messages = modify_expert::get_text_modification_prompt(&original_content, correction_text);
        let gbnf_schema = modify_expert::get_text_modification_gbnf_schema();
        let request_body = json!({ "messages": messages, "temperature": 0.0, "grammar": gbnf_schema });
        let chat_url = format!("{}/v1/chat/completions", self.llm_config.llm_url);
        
        let response = self.llm_config.client.post(&chat_url).json(&request_body).send().await?;
        let chat_response: ChatCompletionResponse = response.json().await?;
        let content_str = chat_response.choices.get(0).map(|c| c.message.content.trim())
            .ok_or_else(|| anyhow::anyhow!("Contextual Modify LLM response is empty"))?;
        
        let modified_text_obj: modify_expert::ModifiedText = serde_json::from_str(content_str)?;
        let new_content = &modified_text_obj.modified_text;

        println!("[Orchestrator-DST] LLM generated new text: '{}'", new_content);

        memos_agent.update(memory_id, new_content).await?;
        Ok(format!("好的，我已经将记忆更新为：{}", new_content))
    }

    // 辅助函数：更新对话历史和缓存
    fn update_history_and_cache(&self, text: &str, final_response: &str) {
        let mut history = self.conversation_history.lock().unwrap();
        history.push(format!("User: {}", text));
        history.push(format!("Assistant: {}", final_response));
        const MAX_HISTORY_SIZE: usize = 10; // 适当增加历史记录长度以支持更长上下文
        let current_len = history.len();
        if current_len > MAX_HISTORY_SIZE {
            let drain_count = current_len - MAX_HISTORY_SIZE;
            history.drain(..drain_count);
        }
        println!("[Orchestrator] Updated history: {:?}", history);
        *self.last_full_interaction.lock().unwrap() = Some((text.to_string(), final_response.to_string()));
    }

    // 辅助函数：获取 MemosAgent 实例
    fn get_memos_agent(&self) -> Result<&MemosAgent, anyhow::Error> {
        self.agents.iter()
            .find_map(|a| a.as_any().downcast_ref::<MemosAgent>())
            .ok_or_else(|| anyhow::anyhow!("MemosAgent not found in Orchestrator"))
    }

    // 辅助函数：处理反馈（保持不变）
    pub fn handle_feedback(&self) {
        if let Some((user_input, assistant_response)) = self.last_full_interaction.lock().unwrap().as_ref() {
            // 定义我们希望的训练数据格式
            let feedback_data = json!({
                "instruction": user_input,
                "input": "", // 对于工具调用微调，这个字段通常为空
                "output": format!("// TODO: Add the expected correct tool call JSON here.\n// Assistant's wrong response was: {}", assistant_response),
            });

            // 打开或创建一个名为 feedback.jsonl 的文件，以追加模式写入
            if let Ok(mut file) = OpenOptions::new()
                .create(true)
                .write(true)
                .append(true)
                .open("feedback.jsonl")
            {
                // 将JSON对象转换为字符串，并在末尾加上换行符
                if let Ok(line) = serde_json::to_string(&feedback_data) {
                    if writeln!(file, "{}", line).is_ok() {
                        println!("[Feedback] Successfully saved feedback to feedback.jsonl");
                    }
                }
            }
        } else {
            println!("[Feedback] No last interaction found to provide feedback on.");
        }
    }

    // =================================================================
    // == 4. 核心 `dispatch` 方法，实现混合路由架构 ==
    // =================================================================
    pub async fn dispatch(&self, command: &Command) -> Result<Response, anyhow::Error> {
        match command {
            Command::ProcessText(text) => {
                // --- 阶段一：确定性前置分流 (V2 - 完全体) ---
                
                // 规则0【新增】：处理明确的闲聊指令，最高优先级
                let chit_chat_keywords = ["天气", "笑话"];
                if chit_chat_keywords.iter().any(|&kw| text.contains(kw)) {
                    println!("[Orchestrator] Heuristic Route: Detected Chit-Chat via keywords.");
                    let response = "抱歉，我暂时无法处理这类请求，我的主要任务是记录和查询记忆。".to_string();
                    self.update_history_and_cache(text, &response);
                    return Ok(Response::Text(response));
                }

                // 规则1：检查是否为“上下文修正”指令
                let correction_keywords = ["不对", "错了", "不是", "应该是"];
                if correction_keywords.iter().any(|&kw| text.starts_with(kw)) {
                    let context_id_to_correct = {
                        let mut context_guard = self.last_interaction_context.lock().unwrap();
                        if let Some(context) = context_guard.take() {
                            if let ContextualAction::Save { memory_id } = context.last_action {
                                Some(memory_id)
                            } else { None }
                        } else { None }
                    }; 

                    if let Some(memory_id) = context_id_to_correct {
                        println!("[Orchestrator-DST] Contextual Route: Detected correction for last saved memory ID: {}", memory_id);
                        let final_response = self.handle_contextual_modify(memory_id, text).await?;
                        self.update_history_and_cache(text, &final_response);
                        return Ok(Response::Text(final_response));
                    }
                }

                // 规则2：检查是否为“确认/取消”指令
                if confirmation_expert::parse_confirmation(text) != confirmation_expert::ConfirmationDecision::Unclear {
                    println!("[Orchestrator] Heuristic Route: Detected ConfirmationTool.");
                    let final_response = self.handle_confirmation(text).await?;
                    self.update_history_and_cache(text, &final_response);
                    return Ok(Response::Text(final_response));
                }
                
                // 规则3【升级】：处理上下文删除
                let contextual_delete_keywords = ["刚才那条", "上一条", "这个"];
                if text.contains("删") && contextual_delete_keywords.iter().any(|&kw| text.contains(kw)) {
                    let context_id_to_delete = {
                        let context_guard = self.last_interaction_context.lock().unwrap();
                        // 这里我们用 .as_ref() 来“偷看一下”上下文，而不消耗它
                        context_guard.as_ref().and_then(|context| {
                            match context.last_action {
                                ContextualAction::Save { memory_id } => Some(memory_id),
                                ContextualAction::Recall { memory_id, .. } => Some(memory_id),
                            }
                        })
                    };

                    if let Some(memory_id) = context_id_to_delete {
                        println!("[Orchestrator-DST] Contextual Route: Detected contextual delete for memory ID: {}", memory_id);
                        let final_response = self.handle_delete_by_id(memory_id).await?; // 调用一个新的、通过ID直接删除的函数
                        self.update_history_and_cache(text, &final_response);
                        return Ok(Response::Text(final_response));
                    }
                }

                // 规则4：检查是否为“修改”指令 (非上下文)
                let modify_keywords = ["修改", "改成", "更新", "编辑"];
                if modify_keywords.iter().any(|&kw| text.contains(kw)) {
                    println!("[Orchestrator] Heuristic Route: Detected ModifyTool via keywords.");
                    *self.last_interaction_context.lock().unwrap() = None;
                    let final_response = self.handle_modify(text).await?;
                    self.update_history_and_cache(text, &final_response);
                    return Ok(Response::Text(final_response));
                }

                // 规则5：检查是否为“删除”指令 (非上下文)
                let delete_keywords = ["删除", "忘掉", "去掉", "移除"];
                if delete_keywords.iter().any(|&kw| text.contains(kw)) {
                    println!("[Orchestrator] Heuristic Route: Detected DeleteTool via keywords.");
                    *self.last_interaction_context.lock().unwrap() = None;
                    let final_response = self.handle_delete(text).await?;
                    self.update_history_and_cache(text, &final_response);
                    return Ok(Response::Text(final_response));
                }

                // --- 阶段二：LLM 兜底路由 (模型辅助) ---
                // ... (此部分代码完全不变，从 println! 开始) ...
                println!("[Orchestrator] No heuristic matched. Downgrading to LLM Router.");
                *self.last_interaction_context.lock().unwrap() = None;

                #[derive(Deserialize)] struct ChatChoice { message: ChatMessageContent }
                #[derive(Deserialize)] struct ChatMessageContent { content: String }
                #[derive(Deserialize)] struct ChatCompletionResponse { choices: Vec<ChatChoice> }

                let history_clone = self.conversation_history.lock().unwrap().clone();
                
                println!("[Orchestrator] Step 2: Routing with raw user text: '{}'", text);
                
                let router_messages = router::get_routing_prompt(text, &history_clone);
                let chat_url = format!("{}/v1/chat/completions", self.llm_config.llm_url);
                
                let router_request_body = json!({ "messages": router_messages, "temperature": 0.0 });
                let router_response = self.llm_config.client.post(&chat_url).json(&router_request_body).send().await?;
                if !router_response.status().is_success() {
                    return Err(anyhow::anyhow!("Router LLM call failed: {}", router_response.text().await?));
                }
                
                let router_chat_response: ChatCompletionResponse = router_response.json().await?;
                let raw_content_str = router_chat_response.choices.get(0).map(|c| c.message.content.trim())
                    .ok_or_else(|| anyhow::anyhow!("Router LLM response is empty"))?;
                
                use regex::Regex;
                println!("[Router] Raw LLM Output: {}", raw_content_str);
                let re = Regex::new(r"\{[\s\S]*\}")?;
                let router_content_str = if let Some(mat) = re.find(raw_content_str) {
                    mat.as_str()
                } else {
                    return Err(anyhow::anyhow!("No valid JSON object found in LLM response. Raw output was: {}", raw_content_str));
                };

                println!("[Router] Cleaned Decision JSON: {}", router_content_str);
                let decision: RoutingDecision = serde_json::from_str(router_content_str)?;

                // --- 阶段三：专家执行 ---
                println!("[Orchestrator] Executing expert: {:?}", decision.tool_to_call);
                let final_response = match decision.tool_to_call {
                    ToolToCall::SaveTool => self.handle_save(text).await?,
                    ToolToCall::RecallTool => self.handle_recall(text).await?,
                    ToolToCall::NoTool => "抱歉，我不太明白您的意思，可以换个方式说吗？".to_string(),
                    _ => "我似乎理解错了您的意思，可以换一种方式告诉我吗？".to_string(),
                };

                self.update_history_and_cache(text, &final_response);
                Ok(Response::Text(final_response))
            }
        }
    }
}

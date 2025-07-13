// orchestrator/src/lib.rs

mod experts;
mod preprocessors;
use micromodels::{Classifier, Intent as MicroIntent}; // 使用别名避免与未来可能的内部Intent冲突
use std::path::Path;
use agent_memos::MemosAgent;
use memos_core::{Agent, Command, Response};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, Mutex};

// 【最终风格修正】使用 snake_case 路径
use experts::memos_agent::{
    save_expert,
    modify_expert,
    re_ranker::{ReRanker, ReRankRequest, DocumentToRank, ReRankStrategy},
};
use std::fs::OpenOptions;
use std::io::Write;

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
    // 可以在未来加入时间戳，让上下文在一段时间后自动失效
    // pub timestamp: std::time::Instant,
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
    is_question_classifier: Mutex<Classifier>,
    confirmation_classifier: Mutex<Classifier>,
}


impl Orchestrator {
    pub fn new(agents: Vec<Box<dyn Agent>>, llm_url: &str, reranker_llm_url: Option<&str>, models_path: &Path) -> Self {
        println!("[Orchestrator] Loading micromodels from path: {:?}", models_path);

        // 加载问题分类器
        let is_question_model_path = models_path.join("is_question_classifier.onnx");
        let is_question_data_path = models_path.join("is_question_preprocessor.bin");
        let is_question_classifier = Classifier::load(
            is_question_model_path,
            is_question_data_path, // <--- 修正为正确的变量名
            vec![MicroIntent::Question, MicroIntent::Statement, MicroIntent::Unknown]
        ).expect("CRITICAL: Failed to load is_question classifier.");
        println!("[Orchestrator] 'is_question_classifier' loaded successfully.");

        // 加载确认分类器
        let confirmation_model_path = models_path.join("confirmation_classifier.onnx");
        let confirmation_data_path = models_path.join("confirmation_preprocessor.bin");
        let confirmation_classifier = Classifier::load(
            confirmation_model_path,
            confirmation_data_path,
            vec![MicroIntent::Affirm, MicroIntent::Deny, MicroIntent::Unknown] // 确保包含所有可能的标签
        ).expect("CRITICAL: Failed to load confirmation classifier.");
        println!("[Orchestrator] 'confirmation_classifier' loaded successfully.");

        Self {
            agents,
            llm_config: LLMConfig::new(llm_url),
            conversation_history: Arc::new(Mutex::new(Vec::new())),
            reranker: reranker_llm_url.map(ReRanker::new),
            pending_action: Arc::new(Mutex::new(None)),
            last_interaction_context: Arc::new(Mutex::new(None)),
            last_full_interaction: Arc::new(Mutex::new(None)),
            is_question_classifier: Mutex::new(is_question_classifier),
            confirmation_classifier: Mutex::new(confirmation_classifier),
        }
    }


    async fn handle_save(&self, text: &str) -> Result<String, anyhow::Error> {
        let memos_agent = self.agents.iter().find_map(|a| a.as_any().downcast_ref::<MemosAgent>()).ok_or_else(|| anyhow::anyhow!("MemosAgent not found"))?;
        println!("[SaveExpert] Extracting fact from raw text: '{}'", text);
        
        let messages = save_expert::get_fact_extraction_prompt(text);
        let gbnf_schema = save_expert::get_fact_extraction_gbnf_schema();
        let request_body = json!({ "messages": messages, "temperature": 0.0, "grammar": gbnf_schema });
        let chat_url = format!("{}/v1/chat/completions", self.llm_config.llm_url);
        
        // 我们需要在这里定义解析LLM响应的结构体
        #[derive(Deserialize)] struct ChatChoice { message: ChatMessageContent }
        #[derive(Deserialize)] struct ChatMessageContent { content: String }
        #[derive(Deserialize)] struct ChatCompletionResponse { choices: Vec<ChatChoice> }
        
        let response = self.llm_config.client.post(&chat_url).json(&request_body).send().await?;
        let chat_response: ChatCompletionResponse = response.json().await?;
        let content_str = chat_response.choices.get(0).map(|c| c.message.content.trim()).unwrap_or("{}");
        let extracted_fact_obj: save_expert::ExtractedFact = serde_json::from_str(content_str)?;
        let fact_to_save = &extracted_fact_obj.fact;
        
        println!("[SaveExpert] Fact to save: '{}'", fact_to_save);
        
        // 调用修改后的save方法，并接收返回的ID
        let new_memory_id = memos_agent.save(fact_to_save).await?;

        // --- 新增：更新短期上下文 ---
        let context = InteractionContext {
            last_action: ContextualAction::Save { memory_id: new_memory_id },
        };
        *self.last_interaction_context.lock().unwrap() = Some(context);
        println!("[Orchestrator-DST] Updated context: Last action was Save with ID {}", new_memory_id);
        // --- 更新结束 ---

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
            let intent = self.confirmation_classifier.lock().unwrap().predict(text);
            
            match intent {
                MicroIntent::Affirm => {
                    println!("[ConfirmationExpert] Micromodel classified as 'Affirm'. Executing pending action.");
                    self.execute_pending_action(action).await
                }
                MicroIntent::Deny => {
                    println!("[ConfirmationExpert] Micromodel classified as 'Deny'. Cancelling pending action.");
                    Ok("好的，已取消操作。".to_string())
                }
                _ => { // 处理 Unclear 和其他所有情况
                    println!("[ConfirmationExpert] Micromodel response unclear. Re-instating pending action.");
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
                
                // 定义解析LLM响应所需的本地结构体
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



    pub async fn dispatch(&self, command: &Command) -> Result<Response, anyhow::Error> {
        match command {
            Command::ProcessText(text) => {
                // --- V10.2 最终版：具备状态管理的、三层分级路由策略 ---
                println!("[Orchestrator] V10.2 Routing with State-Aware strategy...");

                let final_response: String;

                // 1. "海马体"层：优先检查是否存在待处理的上下文动作 (PendingAction)。
                // 我们克隆一份，如果后续处理失败，可以再把它放回去。
                let pending_action = self.pending_action.lock().unwrap().clone();

                if pending_action.is_some() {
                    // 如果存在待处理动作，说明我们正处于一个多轮对话中。
                    // 此时用户的输入（如“是的”、“取消”）应被直接路由到确认处理器。
                    println!("[Orchestrator] Pending action found. Routing to ConfirmationHandler.");
                    final_response = self.handle_confirmation(text).await?;

                } else {
                    // 2. "脑干"层：如果没有待处理动作，再检查是否为高优先级核心指令。
                    let modify_keywords = ["修改", "改成", "更新", "编辑"];
                    let delete_keywords = ["删除", "忘掉", "去掉", "移除"];

                    if modify_keywords.iter().any(|&kw| text.contains(kw)) {
                        println!("[Orchestrator] Heuristic Route: Detected ModifyTool. Routing to ModifyExpert.");
                        final_response = self.handle_modify(text).await?;

                    } else if delete_keywords.iter().any(|&kw| text.contains(kw)) {
                        println!("[Orchestrator] Heuristic Route: Detected DeleteTool. Routing to DeleteExpert.");
                        final_response = self.handle_delete(text).await?;

                    } else {
                        // 3. "小脑"层：如果以上都不是，才将任务交给微模型进行常规分类。
                        println!("[Orchestrator] No pending action or heuristic hit. Falling back to 'is_question_classifier'...");
                        let intent = self.is_question_classifier.lock().unwrap().predict(text);

                        final_response = match intent {
                            MicroIntent::Question => {
                                println!("[Orchestrator] Micromodel classified as 'Question'. Routing to RecallExpert.");
                                self.handle_recall(text).await?
                            }
                            MicroIntent::Statement => {
                                println!("[Orchestrator] Micromodel classified as 'Statement'. Routing to SaveExpert.");
                                self.handle_save(text).await?
                            }
                            // 【关键修正】在这一层，Affirm/Deny 意图被视为无上下文的回答。
                            MicroIntent::Affirm | MicroIntent::Deny => {
                                "嗯？我们刚才有在讨论什么需要确认的事情吗？".to_string()
                            }
                            MicroIntent::Unknown => {
                                "抱歉，我不太明白您的意思，可以换个方式说吗？".to_string()
                            }
                        };
                    }
                }

                // --- 统一处理历史记录和上下文缓存 (保持不变) ---
                let mut history = self.conversation_history.lock().unwrap();
                history.push(format!("User: {}", text));
                history.push(format!("Assistant: {}", final_response));
                const MAX_HISTORY_SIZE: usize = 8;
                if history.len() > MAX_HISTORY_SIZE {
                    let drain_count = history.len() - MAX_HISTORY_SIZE;
                    history.drain(..drain_count);
                }
                println!("[Orchestrator] Updated history: {:?}", history);

                *self.last_full_interaction.lock().unwrap() = Some((text.to_string(), final_response.clone()));

                Ok(Response::Text(final_response))
            }
        }
    }}
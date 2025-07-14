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
    // --- 新增状态：等待用户从多个选项中澄清 ---
    Clarification { 
        // 存储候选记忆的ID和内容，用于向用户展示
        options: Vec<(i64, String)>, 
        // 存储原始意图，以便用户做出选择后，我们知道是该修改还是删除
        original_intent: ClarifiableIntent,
    },
}

// --- 新增枚举：定义哪些意图是需要澄清的 ---
#[derive(Debug, Clone, Copy)]
pub enum ClarifiableIntent {
    Modify,
    Delete,
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
    Recall { 
        memory_id: i64, 
        content: String,
        // --- 新增字段：存储从召回内容中提取出的核心实体 ---
        entities: Vec<String>,
    },
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
        
        let memos_agent = self.agents.iter()
            .find_map(|a| a.as_any().downcast_ref::<MemosAgent>())
            .ok_or_else(|| anyhow::anyhow!("MemosAgent not found"))?;
        
        // 初次召回，不带任何上下文
        let candidate_points = memos_agent.recall(text, None).await?;
        
        if candidate_points.is_empty() {
            return Ok(format!("关于“{}”，我好像没什么印象...", text));
        }

        let final_content: String;
        let top_point: &qdrant_client::qdrant::ScoredPoint;

        if let Some(reranker) = &self.reranker {
            println!("[RecallExpert] Re-ranking candidates...");
            let documents_to_rank: Vec<DocumentToRank> = candidate_points.iter()
                .filter_map(|p| p.payload.get("content").and_then(|v| v.as_str()).map(|s| DocumentToRank { text: s }))
                .collect();
            let rerank_request = ReRankRequest { query: text, documents: documents_to_rank };
            let strategy = ReRankStrategy::ValidateTopOne { threshold: 0.1 };
            let final_results = reranker.rank(rerank_request, strategy).await?;

            if let Some(top_doc) = final_results.get(0) {
                top_point = candidate_points.iter().find(|p| {
                    p.payload.get("content").and_then(|v| v.as_str()) == Some(&top_doc.text)
                }).unwrap();
                final_content = top_doc.text.clone();
            } else {
                let summary: Vec<String> = candidate_points.iter().take(3)
                    .filter_map(|p| p.payload.get("content").and_then(|v| v.as_str()).map(|s| format!("- {}", s)))
                    .collect();
                return Ok(format!("关于“{}”，我没有找到直接答案，但发现一些可能相关的内容：\n{}", text, summary.join("\n")));
            }
        } else {
            println!("[RecallExpert] Skipping re-ranking.");
            top_point = candidate_points.get(0).unwrap();
            final_content = top_point.payload.get("content")
                                .and_then(|v| v.as_str())
                                .map_or("".to_string(), |v| v.to_string());
        }

        let memory_id = top_point.id.as_ref().and_then(|id| match &id.point_id_options {
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => Some(*num as i64),
            _ => None,
        }).unwrap_or(-1);

        // --- 核心修复：不再依赖 payload，而是对成功召回的内容主动进行NER，以获取最准确的上下文实体 ---
        let memos_agent_for_ner = self.agents.iter()
            .find_map(|a| a.as_any().downcast_ref::<MemosAgent>())
            .ok_or_else(|| anyhow::anyhow!("MemosAgent not found for context NER"))?;
        
        // 对最终确认的召回内容，通过新的公共方法提取实体
        let entities = memos_agent_for_ner.extract_entities(&final_content)?;
        println!("[Orchestrator-DST] Extracted entities for context: {:?}", entities);
        // --- 修复结束 ---

        let context = InteractionContext {
            last_action: ContextualAction::Recall {
                memory_id,
                content: final_content.clone(),
                entities,
            },
        };
        *self.last_interaction_context.lock().unwrap() = Some(context);
        println!("[Orchestrator-DST] Updated context: Last action was Recall with ID {} and content '{}'", memory_id, final_content);

        Ok(final_content)
    }

    async fn handle_modify(&self, text: &str) -> Result<String, anyhow::Error> {
        println!("[ModifyExpert-Phase1] Received request: '{}'", text);
        
        // --- 指代消解逻辑 START (V2 - 意图驱动) ---
        println!("[Orchestrator-DST] Modify intent detected. Attempting to use context regardless of pronouns.");
        // 直接尝试从上一次交互中获取实体，不再需要判断输入中是否包含代词。
        let context_entities: Option<Vec<String>> = self.last_interaction_context.lock().unwrap()
            .as_ref()
            .and_then(|ctx| match &ctx.last_action {
                // 如果上次是召回操作，并且成功提取了实体，就使用它们
                ContextualAction::Recall { entities, .. } if !entities.is_empty() => {
                    println!("[Orchestrator-DST] Found context entities: {:?}", entities);
                    Some(entities.clone())
                },
                _ => None,
            });
        // --- 指代消解逻辑 END ---

        let memos_agent = self.agents.iter()
            .find_map(|a| a.as_any().downcast_ref::<MemosAgent>())
            .ok_or_else(|| anyhow::anyhow!("MemosAgent not found"))?;

        // 将原始用户输入和（可能存在的）上下文实体，分别传递给 recall 函数
        let candidate_points = memos_agent.recall(text, context_entities).await?;
        
        match candidate_points.len() {
            0 => Ok("抱歉，我没有找到与您描述相关的记忆。".to_string()),
            1 => {
                // 行为不变：只有一个匹配项，直接进入确认流程
                let top_point = candidate_points.get(0).unwrap();
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
            }
            _ => {
                // 核心改造：有多个匹配项，进入澄清流程
                println!("[Orchestrator] Multiple candidates found. Entering clarification mode.");
                let options: Vec<(i64, String)> = candidate_points.into_iter().filter_map(|p| {
                    let id = p.id.as_ref()?.point_id_options.as_ref()?;
                    let content = p.payload.get("content")?.as_str()?;
                    if let qdrant_client::qdrant::point_id::PointIdOptions::Num(num) = id {
                        Some((*num as i64, content.to_string()))
                    } else {
                        None
                    }
                }).collect();

                let pending_action = PendingAction {
                    action_type: PendingActionType::Clarification {
                        options: options.clone(),
                        original_intent: ClarifiableIntent::Modify,
                    },
                    original_user_request: text.to_string(),
                };
                *self.pending_action.lock().unwrap() = Some(pending_action);

                let options_text = options.iter().enumerate()
                    .map(|(i, (_, content))| format!("{}. {}", i + 1, content))
                    .collect::<Vec<_>>()
                    .join("\n");

                Ok(format!("我找到了多条相关记忆，您想修改哪一条？\n\n{}", options_text))
            }
        }
    }

    async fn handle_delete(&self, text: &str) -> Result<String, anyhow::Error> {
        println!("[DeleteExpert-Phase1] Received request: '{}'", text);

        // --- 指代消解逻辑 START (V2 - 意图驱动) ---
        println!("[Orchestrator-DST] Delete intent detected. Attempting to use context regardless of pronouns.");
        // 直接尝试从上一次交互中获取实体
        let context_entities: Option<Vec<String>> = self.last_interaction_context.lock().unwrap()
            .as_ref()
            .and_then(|ctx| match &ctx.last_action {
                ContextualAction::Recall { entities, .. } if !entities.is_empty() => {
                    println!("[Orchestrator-DST] Found context entities: {:?}", entities);
                    Some(entities.clone())
                },
                _ => None,
            });
        // --- 指代消解逻辑 END ---

        let memos_agent = self.agents.iter()
            .find_map(|a| a.as_any().downcast_ref::<MemosAgent>())
            .ok_or_else(|| anyhow::anyhow!("MemosAgent not found"))?;

        // 将原始用户输入和（可能存在的）上下文实体，分别传递给 recall 函数
        let candidate_points = memos_agent.recall(text, context_entities).await?;

        match candidate_points.len() {
            0 => Ok("抱歉，我没有找到与您描述相关的记忆可以删除。".to_string()),
            1 => {
                // 行为不变：只有一个匹配项
                let top_point = candidate_points.get(0).unwrap();
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
                
                Ok(format!("您确定要删除这条记忆吗？\n\n---\n{}\n---", content))
            }
            _ => {
                // 核心改造：有多个匹配项
                println!("[Orchestrator] Multiple candidates found. Entering clarification mode for deletion.");
                let options: Vec<(i64, String)> = candidate_points.into_iter().filter_map(|p| {
                    let id = p.id.as_ref()?.point_id_options.as_ref()?;
                    let content = p.payload.get("content")?.as_str()?;
                    if let qdrant_client::qdrant::point_id::PointIdOptions::Num(num) = id {
                        Some((*num as i64, content.to_string()))
                    } else {
                        None
                    }
                }).collect();

                let pending_action = PendingAction {
                    action_type: PendingActionType::Clarification {
                        options: options.clone(),
                        original_intent: ClarifiableIntent::Delete,
                    },
                    original_user_request: text.to_string(),
                };
                *self.pending_action.lock().unwrap() = Some(pending_action);

                let options_text = options.iter().enumerate()
                    .map(|(i, (_, content))| format!("{}. {}", i + 1, content))
                    .collect::<Vec<_>>()
                    .join("\n");

                Ok(format!("我找到了多条相关记忆，您想删除哪一条？\n\n{}", options_text))
            }
        }
    }
    
    async fn handle_confirmation(&self, text: &str) -> Result<String, anyhow::Error> {
        // 1. 先取出当前的待办事项
        let taken_action = self.pending_action.lock().unwrap().take();

        if let Some(action) = taken_action {
            match action.action_type {
                PendingActionType::Clarification { options, original_intent } => {
                    let choice = text.trim().parse::<usize>().ok();
                    if let Some(choice_idx) = choice.and_then(|c| c.checked_sub(1)) {
                        if let Some((chosen_id, chosen_content)) = options.get(choice_idx) {
                            // 用户做出了有效选择，我们创建一个新的、更具体的待办事项
                            let new_action_type = match original_intent {
                                ClarifiableIntent::Modify => PendingActionType::ModifyConfirmation {
                                    memory_id: *chosen_id,
                                    original_content: chosen_content.clone(),
                                },
                                ClarifiableIntent::Delete => PendingActionType::DeleteConfirmation {
                                    memory_id: *chosen_id,
                                    content_to_delete: chosen_content.clone(),
                                },
                            };
                            let new_pending_action = PendingAction {
                                action_type: new_action_type,
                                original_user_request: action.original_user_request,
                            };
                            
                            // --- 修复：重新获取锁，并将新的待办事项放回去 ---
                            *self.pending_action.lock().unwrap() = Some(new_pending_action);
                            
                            // 生成确认问题并返回
                            let confirmation_prompt = match original_intent {
                                ClarifiableIntent::Modify => format!("您是想修改这条记忆吗？\n\n---\n{}\n---", chosen_content),
                                ClarifiableIntent::Delete => format!("您确定要删除这条记忆吗？\n\n---\n{}\n---", chosen_content),
                            };
                            return Ok(confirmation_prompt);
                        }
                    }
                    println!("[ConfirmationExpert] User input '{}' is not a valid choice. Cancelling.", text);
                    Ok("好的，已取消操作。".to_string())
                }
                
                PendingActionType::ModifyConfirmation { .. } | PendingActionType::DeleteConfirmation { .. } => {
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
                        _ => {
                            println!("[ConfirmationExpert] Micromodel response unclear. Re-instating pending action.");
                            // 如果不清楚，把 action 放回去
                            *self.pending_action.lock().unwrap() = Some(action);
                            Ok("抱歉，我没太明白。请明确回复“是”或“否”，或者您想操作的选项编号。".to_string())
                        }
                    }
                }
            }
        } else {
            Ok("嗯？我们刚才有在讨论什么需要确认的事情吗？".to_string())
        }
    }

    async fn execute_pending_action(&self, action: PendingAction) -> Result<String, anyhow::Error> {
            match action.action_type {
                PendingActionType::ModifyConfirmation { memory_id, original_content } => {
                    // ... (此部分代码保持不变)
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
                    
                    let memos_agent = self.agents.iter()
                        .find_map(|a| a.as_any().downcast_ref::<MemosAgent>())
                        .ok_or_else(|| anyhow::anyhow!("MemosAgent not found"))?;
                    
                    memos_agent.update(memory_id, new_content).await?;
                    Ok("好的，我已经更新了这条记忆。".to_string())
                }
                PendingActionType::DeleteConfirmation { memory_id, .. } => {
                    // ... (此部分代码保持不变)
                    println!("[DeleteExpert-Phase2] Executing deletion for ID: {}", memory_id);
                    
                    let memos_agent = self.agents.iter()
                        .find_map(|a| a.as_any().downcast_ref::<MemosAgent>())
                        .ok_or_else(|| anyhow::anyhow!("MemosAgent not found"))?;
                    
                    memos_agent.delete(memory_id).await?;
                    Ok("好的，我已经删除了这条记忆。".to_string())
                }
                // --- 修复：处理被遗漏的 Clarification 分支 ---
                PendingActionType::Clarification { .. } => {
                    // 这是一个逻辑错误，澄清动作不应该被“执行”。
                    // 我们返回一个错误，而不是让程序崩溃。
                    Err(anyhow::anyhow!(
                        "[Logic Error] Attempted to execute a 'Clarification' action. This should not happen."
                    ))
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
                // --- V10.3 最终版：具备增强型启发式规则的、三层分级路由策略 ---
                println!("[Orchestrator] V10.3 Routing with Enhanced Heuristics...");

                let final_response: String;

                // 1. "海马体"层：优先检查是否存在待处理的上下文动作。
                if self.pending_action.lock().unwrap().is_some() {
                    println!("[Orchestrator] Pending action found. Routing to ConfirmationHandler.");
                    final_response = self.handle_confirmation(text).await?;
                } else {
                    // 2. "脑干"层：增强型启发式规则引擎。
                    let lower_text = text.to_lowercase();
                    let modify_keywords = ["修改", "改成", "更新", "编辑"];
                    let delete_keywords = ["删除", "忘掉", "去掉", "移除"];
                    let save_keywords = ["记一下", "记录", "帮我记"];

                    // --- 修复：补上被遗漏的 is_declarative 定义 ---
                    // 陈述性模式 (更智能的“保险丝”)
                    let is_declarative = (lower_text.contains("是") || lower_text.contains("为")) && !lower_text.contains('？') && !lower_text.contains('?');

                    if modify_keywords.iter().any(|&kw| lower_text.contains(kw)) {
                        println!("[Orchestrator] Heuristic Route: Detected ModifyTool.");
                        final_response = self.handle_modify(text).await?; // <-- 直接使用原始 text
                    } else if delete_keywords.iter().any(|&kw| lower_text.contains(kw)) {
                        println!("[Orchestrator] Heuristic Route: Detected DeleteTool.");
                        final_response = self.handle_delete(text).await?; // <-- 直接使用原始 text
                        
                    } else if save_keywords.iter().any(|&kw| lower_text.contains(kw)) || is_declarative {
                        println!("[Orchestrator] Heuristic Route: Detected SaveTool by keyword or declarative pattern.");
                        final_response = self.handle_save(text).await?;
                    } else {
                        // 3. "小脑"层：如果以上规则都未命中，才将任务交给微模型。
                        println!("[Orchestrator] No heuristic hit. Falling back to 'is_question_classifier'...");
                        let intent = self.is_question_classifier.lock().unwrap().predict(text);

                        final_response = match intent {
                            MicroIntent::Question => {
                                println!("[Orchestrator] Micromodel classified as 'Question'. Routing to RecallExpert.");
                                self.handle_recall(text).await?
                            }
                            // 如果微模型也认为是Statement，那就一定是Save
                            MicroIntent::Statement => {
                                println!("[Orchestrator] Micromodel classified as 'Statement'. Routing to SaveExpert.");
                                self.handle_save(text).await?
                            }
                            MicroIntent::Affirm | MicroIntent::Deny => {
                                "嗯？我们刚才有在讨论什么需要确认的事情吗？".to_string()
                            }
                            MicroIntent::Unknown => {
                                "抱歉，我不太明白您的意思，可以换个方式说吗？".to_string()
                            }
                        };
                    }
                }

                // --- 统一处理历史记录 (保持不变) ---
                let mut history = self.conversation_history.lock().unwrap();
                history.push(format!("User: {}", text));
                history.push(format!("Assistant: {}", final_response));
                const MAX_HISTORY_SIZE: usize = 8;
                if history.len() > MAX_HISTORY_SIZE {
                    // 1. 先进行不可变借用，计算出需要移除的数量，并将结果存到一个新变量中。
                    // 在这行代码结束后，对 history.len() 的不可变借用就结束了。
                    let drain_count = history.len() - MAX_HISTORY_SIZE;
                    
                    // 2. 然后，再对 history 进行可变借用，执行 drain 操作。
                    // 此时不存在任何不可变借用，操作是安全的。
                    history.drain(..drain_count);
                }
                println!("[Orchestrator] Updated history: {:?}", history);
                *self.last_full_interaction.lock().unwrap() = Some((text.to_string(), final_response.clone()));
                Ok(Response::Text(final_response))
            }
        }
    }}
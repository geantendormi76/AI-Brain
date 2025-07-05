// orchestrator/src/lib.rs

mod prompts;
use agent_memos::MemosAgent;
use memos_core::{Agent, Command, Response};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

// IntentClassifier 现在只负责持有 client 和 url
pub struct IntentClassifier {
    client: Client,
    llm_url: String,
}
impl IntentClassifier {
    pub fn new(llm_url: &str) -> Self {
        Self {
            client: Client::new(),
            llm_url: llm_url.to_string(),
        }
    }
    // 移除不再使用的 classify 方法
}

// ---- Orchestrator 结构体和实现 (V3.0 最终版) ----
pub struct Orchestrator {
    agents: Vec<Box<dyn Agent>>,
    // 复用 IntentClassifier 来持有 LLM 配置
    llm_config: IntentClassifier,
    // 移除 reranker 字段
}

// 用于解析任务列表的结构体
#[derive(Deserialize, Debug)]
struct Task {
    intent: String,
    text: String,
}

#[derive(Deserialize, Debug)]
struct DecomposedTasks {
    tasks: Vec<Task>,
}


impl Orchestrator {
    // 简化 new 方法，不再需要 reranker_llm_url
    pub fn new(agents: Vec<Box<dyn Agent>>, llm_url: &str) -> Self {
        println!("[Orchestrator] Initializing in Task Decomposition mode.");
        Self {
            agents,
            llm_config: IntentClassifier::new(llm_url),
        }
    }

    pub async fn dispatch(&self, command: &Command) -> Result<Response, anyhow::Error> {
        match command {
            Command::ProcessText(text) => {
                // 1. 调用 LLM 进行任务分解
                println!("[Orchestrator] Decomposing command: '{}'", text);
                let messages = prompts::get_intent_classification_messages(text);
                let gbnf_schema = prompts::get_intent_gbnf_schema();

                let request_body = json!({
                    "messages": messages,
                    "n_predict": 512,
                    "temperature": 0.0,
                    "grammar": gbnf_schema,
                });
                
                let chat_url = format!("{}/v1/chat/completions", self.llm_config.llm_url);
                let response = self.llm_config.client.post(&chat_url)
                    .json(&request_body)
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_body = response.text().await?;
                    return Err(anyhow::anyhow!("Task decomposition LLM service returned an error. Status: {}. Body: {}", status, error_body));
                }

                #[derive(Deserialize)] struct ChatChoice { message: ChatMessageContent }
                #[derive(Deserialize)] struct ChatMessageContent { content: String }
                #[derive(Deserialize)] struct ChatCompletionResponse { choices: Vec<ChatChoice> }
                
                let chat_response: ChatCompletionResponse = response.json().await?;
                let content_str = chat_response.choices.get(0).map(|c| &c.message.content)
                    .ok_or_else(|| anyhow::anyhow!("Missing 'choices' in LLM task decomposition response"))?;
                
                println!("[Orchestrator] Decomposed tasks JSON: {}", content_str);
                let decomposed_tasks: DecomposedTasks = serde_json::from_str(content_str)?;

                // 2. 遍历并执行任务列表
                let mut responses: Vec<String> = Vec::new();
                for task in decomposed_tasks.tasks {
                    if let Some(memos_agent) = self.agents.iter().find_map(|a| a.as_any().downcast_ref::<MemosAgent>()) {
                        match task.intent.as_str() {
                            "SaveIntent" => {
                                memos_agent.save(&task.text).await?;
                                responses.push("好的，已经记下了。".to_string());
                            },
                            "RecallIntent" => {
                                let candidate_points = memos_agent.recall(&task.text).await?;
                                if candidate_points.is_empty() {
                                    responses.push(format!("关于“{}”，我好像没什么印象...", task.text));
                                } else {
                                    let top_point = candidate_points.get(0).unwrap();
                                    if let Some(content) = top_point.payload.get("content").and_then(|v| v.as_str()) {
                                        responses.push(content.to_string());
                                    }
                                }
                            },
                            _ => {} // 忽略未知意图
                        }
                    }
                }

                // 3. 组合最终响应
                let final_response = responses.join("\n");
                if final_response.is_empty() {
                    Ok(Response::Text("抱歉，我暂时无法理解您的意图。".to_string()))
                } else {
                    Ok(Response::Text(final_response))
                }
            }
        }
    }
}
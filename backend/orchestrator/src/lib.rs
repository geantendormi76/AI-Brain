// orchestrator/src/lib.rs

mod prompts; // 引入我们新的prompts模块

use agent_memos::MemosAgent; 
use memos_core::{Agent, Command, Response};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

// ---- 新增：定义聊天API的响应结构 ----
#[derive(Deserialize, Debug)]
struct ChatCompletionChoice {
    message: ChatMessageContent,
}

#[derive(Deserialize, Debug)]
struct ChatMessageContent {
    content: String,
}

#[derive(Deserialize, Debug)]
struct ChatCompletionResponse {
    choices: Vec<ChatCompletionChoice>,
}

// ---- 定义分类器的响应结构 ----
#[derive(Deserialize, Debug)]
struct ClassificationResponse {
    intent: String,
}

// ---- 定义分类器的结构体 ----
pub struct IntentClassifier {
    client: Client,
    llm_url: String, // URL现在是服务器的根地址，如 "http://localhost:8282"
}

impl IntentClassifier {
    pub fn new(llm_url: &str) -> Self {
        Self {
            client: Client::new(),
            llm_url: llm_url.to_string(),
        }
    }

    // 分类方法 - 已适配聊天API
    pub async fn classify(&self, text: &str) -> Result<String, anyhow::Error> {
        println!("[IntentClassifier] Classifying text: '{}'", text);
        let messages = prompts::get_intent_classification_messages(text);
        let gbnf_schema = prompts::get_intent_gbnf_schema();

        // 构建符合 /v1/chat/completions 接口的请求体
        let request_body = json!({
            "messages": messages, // 使用 messages 数组
            "n_predict": 128,
            "temperature": 0.1,
            "grammar": gbnf_schema // GBNF 在聊天端点同样有效
        });
        
        // 构造完整的聊天API端点URL
        let chat_url = format!("{}/v1/chat/completions", self.llm_url);

        let response = self.client.post(&chat_url)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await?;
            return Err(anyhow::anyhow!("LLM service returned an error. Status: {}. Body: {}", status, error_body));
        }

        // 解析聊天API返回的JSON结构
        let chat_response: ChatCompletionResponse = response.json().await?;
        
        // 从响应中提取 content 字符串
        let content_str = chat_response.choices
            .get(0)
            .map(|c| &c.message.content)
            .ok_or_else(|| anyhow::anyhow!("Missing 'choices' in LLM chat response"))?;
        
        // 解析 content 字符串中的意图JSON
        let classification: ClassificationResponse = serde_json::from_str(content_str)?;

        println!("[IntentClassifier] Classified intent as: {}", classification.intent);
        Ok(classification.intent)
    }
}

// ---- Orchestrator 结构体和实现 ----
pub struct Orchestrator {
    agents: Vec<Box<dyn Agent>>,
    classifier: IntentClassifier,
}

impl Orchestrator {
    pub fn new(agents: Vec<Box<dyn Agent>>, classifier_llm_url: &str) -> Self {
        Self { 
            agents,
            classifier: IntentClassifier::new(classifier_llm_url),
        }
    }

    pub async fn dispatch(&self, command: &Command) -> Result<Response, anyhow::Error> {
        match command {
            Command::ProcessText(text) => {
                let intent = self.classifier.classify(text).await?;

                // 遍历所有 agents，找到能处理此意图的agent
                for agent in &self.agents {
                    // 使用 agent.interests() 来判断该agent是否能处理这个意图
                    if agent.interests().contains(&intent.as_str()) {
                        // 找到了，进行向下转型以调用具体方法
                        if let Some(memos_agent) = agent.as_any().downcast_ref::<MemosAgent>() {
                            println!("[Orchestrator] Routing intent '{}' to agent '{}'", intent, memos_agent.name());
                            return match intent.as_str() {
                                "SaveIntent" => {
                                    memos_agent.save(text).await?;
                                    Ok(Response::Text("好的，已经记下了。".to_string()))
                                },
                                "RecallIntent" => {
                                    memos_agent.recall(text).await
                                },
                                // 理论上不会到达这里，因为已经被 interests() 过滤
                                _ => unreachable!(), 
                            }
                        }
                    }
                }

                // 如果循环结束都没有找到能处理此意图的agent
                Ok(Response::Text("抱歉，我暂时无法理解您的意图。".to_string()))
            }
        }
    }
}
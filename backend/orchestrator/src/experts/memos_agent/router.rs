// backend/orchestrator/src/experts/memos_agent/router.rs
// 【V7.0 - 高内聚重构版】

use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use regex::Regex;

#[derive(Deserialize, Debug, PartialEq)]
pub enum ToolToCall {
    SaveTool,
    RecallTool,
    ModifyTool,
    DeleteTool,
    ConfirmationTool,
    NoTool,
}

#[derive(Deserialize, Debug)]
pub struct RoutingDecision {
    pub tool_to_call: ToolToCall,
}

// 将LLM调用和解析逻辑完全封装在此函数内
pub async fn run(
    client: &Client,
    llm_url: &str,
    user_query: &str,
    history: &[String],
) -> Result<RoutingDecision, anyhow::Error> {
    let history_context = if history.is_empty() {
        "无对话历史。".to_string()
    } else {
        history.join("\n")
    };

    let system_prompt = format!(
    r#"你是一个判断用户意图的专家。你的任务非常简单：判断用户的输入是一个“需要记录的陈述”还是一个“需要回答的问题”。
**工具描述:**
- `SaveTool`: 用于记录任何事实、想法、陈述或笔记。
- `RecallTool`: 用于回答任何明确或隐含的问题。
- `NoTool`: **仅用于**处理非常简短的、无意义的闲聊。
**关键决策逻辑 (二选一):**
1.  **判断是否为问题**: 如果用户的输入听起来像一个问题，那么**必须**选择 `RecallTool`。
2.  **默认为保存**: 如果用户的输入不是一个明确的问题，那么它就是一条需要记录的信息。**必须**选择 `SaveTool`。
**重要提醒**: 你不需要处理“修改”或“删除”的意图。你的任务不是回答问题，只是做出分类决策。
**输出格式:** 你的回复必须是一个只包含 "tool_to_call" 字段的JSON对象。
**对话历史 (用于参考):**
{}"#, history_context);

    let messages = vec![
        json!({"role": "system", "content": system_prompt}),
        json!({"role": "user", "content": user_query}),
    ];
    
    let request_body = json!({ "messages": messages, "temperature": 0.0 });
    let chat_url = format!("{}/v1/chat/completions", llm_url);

    #[derive(Deserialize)]
    struct ChatChoice { message: ChatMessageContent }
    #[derive(Deserialize)]
    struct ChatMessageContent { content: String }
    #[derive(Deserialize)]
    struct ChatCompletionResponse { choices: Vec<ChatChoice> }

    let response = client.post(&chat_url).json(&request_body).send().await?;
    if !response.status().is_success() {
        return Err(anyhow::anyhow!("Router LLM call failed: {}", response.text().await?));
    }
    
    let chat_response: ChatCompletionResponse = response.json().await?;
    let raw_content_str = chat_response.choices.get(0).map(|c| c.message.content.trim())
        .ok_or_else(|| anyhow::anyhow!("Router LLM response is empty"))?;

    println!("[Router] Raw LLM Output: {}", raw_content_str);
    let re = Regex::new(r"\{[\s\S]*\}")?;
    let router_content_str = if let Some(mat) = re.find(raw_content_str) {
        mat.as_str()
    } else {
        return Err(anyhow::anyhow!("No valid JSON object found in LLM response. Raw output was: {}", raw_content_str));
    };

    println!("[Router] Cleaned Decision JSON: {}", router_content_str);
    let decision: RoutingDecision = serde_json::from_str(router_content_str)?;
    
    Ok(decision)
}
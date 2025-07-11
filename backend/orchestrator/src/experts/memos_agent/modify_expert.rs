// backend/orchestrator/src/experts/memos_agent/modify_expert.rs
// 【V7.0 - 高内聚重构版】

use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize, Debug)]
pub struct ModifiedText {
    pub modified_text: String,
}

// 【新增】将LLM调用逻辑完全封装在此
pub async fn run(
    client: &Client,
    llm_url: &str,
    original_text: &str,
    user_request: &str,
) -> Result<String, anyhow::Error> {
    let system_prompt = format!(
        r#"You are a precise text editor. Your task is to take an original text and a user's modification request, then output the fully rewritten, new version of the text.
Your output MUST be a valid JSON object with a single field "modified_text".

Original Text:
---
{}
---

User's Modification Request:
---
{}
---

Now, generate the new, complete text based on the user's request."#,
        original_text, user_request
    );

    let gbnf_schema = r#"root ::= "{" ws "\"modified_text\"" ws ":" ws string ws "}"
string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}) )* "\""
ws     ::= ([ \t\n\r])*"#;

    let messages = vec![json!({ "role": "system", "content": system_prompt })];
    let request_body = json!({ "messages": messages, "temperature": 0.0, "grammar": gbnf_schema });
    let chat_url = format!("{}/v1/chat/completions", llm_url);
    
    #[derive(Deserialize)] struct ChatChoice { message: ChatMessageContent }
    #[derive(Deserialize)] struct ChatMessageContent { content: String }
    #[derive(Deserialize)] struct ChatCompletionResponse { choices: Vec<ChatChoice> }

    let response = client.post(&chat_url).json(&request_body).send().await?;
    let response_text = response.text().await?;

    let chat_response: ChatCompletionResponse = serde_json::from_str(&response_text)
         .map_err(|e| anyhow::anyhow!("Failed to parse modify_expert LLM choices: {}. Raw text: {}", e, response_text))?;
    
    let content_str = chat_response.choices.get(0).map(|c| c.message.content.trim())
        .ok_or_else(|| anyhow::anyhow!("ModifyExpert LLM response is empty"))?;
    
    let modified_text_obj: ModifiedText = serde_json::from_str(content_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse ModifiedText from LLM content: {}. Raw content: {}", e, content_str))?;

    Ok(modified_text_obj.modified_text)
}
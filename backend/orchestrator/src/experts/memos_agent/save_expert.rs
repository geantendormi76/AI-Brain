// backend/orchestrator/src/experts/memos_agent/save_expert.rs
// 【V7.0 - 高内聚重构版】

use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use memos_core::FactMetadata;

#[derive(Deserialize, Debug)]
pub struct ExtractedFact {
    pub fact: String,
    pub metadata: Option<FactMetadata>,
}

pub fn get_fact_extraction_gbnf_schema() -> &'static str {
    // 这个单行、无注释、依赖优先的GBNF语法是经过验证的最终版
    r#"
ws ::= ([ \t\n\r])*
string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}) )* "\""
topics-array ::= "[" ws (string ("," ws string)*)? ws "]"
metadata-object ::= "{" ws "\"topics\"" ws ":" ws topics-array ws "}"
fact-kv ::= "\"fact\"" ws ":" ws string
metadata-kv ::= "\"metadata\"" ws ":" ws metadata-object
extracted-fact-object ::= "{" ws fact-kv ("," ws metadata-kv)? ws "}"
root   ::= extracted-fact-object
"#
}

pub fn get_fact_extraction_prompt(user_input: &str) -> Vec<Value> {
    let system_prompt = r#"你的任务是从用户的输入中，提取出“核心事实”和相关的“主题标签”，并以一个严格的JSON对象格式输出。
**核心指令:**
1.  **提取事实 (fact)**: 将用户想要记录的核心信息，提炼成一个简洁、完整的陈述句。
2.  **提取标签 (topics)**: 提取出与事实最相关的1到3个核心关键词或主题，作为标签。
3.  **JSON格式**: 你的输出必须是严格的、不包含任何其他文本的JSON对象。
**输出格式:**
```json
{
  "fact": "提取出的核心事实。",
  "metadata": { "topics": ["标签一", "标签二"] }
}
```
---
**示例:**
<用户输入>
帮我记一下，我的个人网站是 example.com
</用户输入>
<你的输出>
```json
{
  "fact": "我的个人网站是 example.com。",
  "metadata": { "topics": ["个人网站", "网站地址"] }
}
```
---
现在，请处理以下用户输入。"#;

    vec![
        json!({"role": "system", "content": system_prompt}),
        json!({"role": "user", "content": user_input}),
    ]
}

// 【新增】将LLM调用逻辑完全封装在此
pub async fn run(
    client: &Client,
    llm_url: &str,
    text: &str,
) -> Result<ExtractedFact, anyhow::Error> {
    println!("[SaveExpert] Extracting fact from raw text: '{}'", text);
    
    let messages = get_fact_extraction_prompt(text);
    let gbnf_schema = get_fact_extraction_gbnf_schema();
    let request_body = json!({ "messages": messages, "temperature": 0.0, "grammar": gbnf_schema });
    let chat_url = format!("{}/v1/chat/completions", llm_url);
    
    #[derive(Deserialize)] struct ChatChoice { message: ChatMessageContent }
    #[derive(Deserialize)] struct ChatMessageContent { content: String }
    #[derive(Deserialize)] struct ChatCompletionResponse { choices: Vec<ChatChoice> }
    
    let response = client.post(&chat_url).json(&request_body).send().await?;
    let response_text = response.text().await?;

    let chat_response: ChatCompletionResponse = serde_json::from_str(&response_text)
        .map_err(|e| anyhow::anyhow!("Failed to parse save_expert LLM choices: {}. Raw text: {}", e, response_text))?;

    let content_str = chat_response.choices.get(0).map(|c| c.message.content.trim())
        .ok_or_else(|| anyhow::anyhow!("SaveExpert LLM response is empty"))?;

    let extracted_fact_obj: ExtractedFact = serde_json::from_str(content_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse ExtractedFact from LLM content: {}. Raw content: {}", e, content_str))?;
    
    println!("[SaveExpert] Fact to save: '{}'", extracted_fact_obj.fact);
    
    Ok(extracted_fact_obj)
}

// orchestrator/src/experts/router.rs

use serde::Deserialize;
use serde_json::Value;

// 1. 【核心改造】在 ToolToCall 枚举中增加新工具
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


pub fn get_routing_prompt(user_query: &str, history: &[String]) -> Vec<Value> {
    let history_context = if history.is_empty() {
        "No conversation history.".to_string()
    } else {
        history.join("\n")
    };

    // --- 新的、更严格的 System Prompt ---
    let system_prompt = format!(
    r#"You are a master router. Your only job is to analyze the user's query and decide which single tool to call.

**Tool Descriptions:**
- `SaveTool`: Use when the user wants to **save, remember, record, or take note** of information. This is your primary default tool.
- `RecallTool`: Use when the user wants to **ask a question, find, search, or recall** information. This includes implicit questions like "What are my plans for tomorrow?" or "What is...?.
- `NoTool`: Use ONLY for greetings ("hello", "hi"), meaningless single words, or if the intent is completely impossible to understand.

**CRITICAL DECISION LOGIC (in order of priority):**
1.  **Question First**: If the query is phrased as a question (e.g., ends with a question mark, starts with "what", "how", "who", "查询"), it is ALWAYS `RecallTool`.
2.  **Default to Save**: If the query is NOT a question, your default choice is ALWAYS `SaveTool`. Any declarative statement, observation, or thought ("Project Titan's core tech is...", "Today the weather is nice", "I feel like hotpot") should be classified as `SaveTool`.
3.  **Exception for NoTool**: Only if the input is a simple greeting like "你好" or complete gibberish, classify it as `NoTool`.

**Your Output MUST be a valid JSON object:**
```json
{{
  "tool_to_call": "string, one of [SaveTool, RecallTool, NoTool]"
}}
```

**Conversation History (for context):**
{}
"#, history_context);

    vec![
        serde_json::json!({"role": "system", "content": system_prompt}),
        serde_json::json!({"role": "user", "content": user_query}),
    ]
}



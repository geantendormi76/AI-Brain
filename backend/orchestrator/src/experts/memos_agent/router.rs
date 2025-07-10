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

    // --- 【V2 版优化】更智能、更严格的 System Prompt ---
    let system_prompt = format!(
    r#"You are a master router. Your only job is to analyze the user's query and decide which single tool to call.

**Tool Descriptions:**
- `SaveTool`: Use when the user wants to **save, remember, record, or take note** of information.
- `RecallTool`: Use when the user wants to **ask a question, find, search, recall, or get an explanation/definition**.
- `ModifyTool`: Use ONLY when the user explicitly asks to **change, update, or modify** a PREVIOUSLY saved memory. Requires context from conversation history.
- `DeleteTool`: Use ONLY when the user explicitly asks to **delete, remove, or forget** a PREVIOUSLY saved memory. Requires context.
- `ConfirmationTool`: Use ONLY for short, direct answers to a yes/no question asked by the assistant (e.g., "yes", "no", "confirm").
- `NoTool`: Use ONLY for greetings ("hello"), meaningless single words, or if the intent is completely impossible to understand.

**CRITICAL DECISION LOGIC (in order of priority):**
1.  **Confirmation First**: If the assistant just asked a yes/no question and the user gives a short confirmation/denial, it is ALWAYS `ConfirmationTool`.
2.  **Explicit Command**: If the query contains explicit modification/deletion keywords like "修改" or "删除", and refers to a past memory, use `ModifyTool` or `DeleteTool`.
3.  **Question / Definition Request**: If the query is a question (ends with "?", starts with "what", "how", etc.) OR asks for a definition/code (e.g., "什么是Rust", "完整get_embedding函数"), it is ALWAYS `RecallTool`.
4.  **Default to Save**: If none of the above match, the default choice is `SaveTool`. Any declarative statement, observation, or thought should be classified as `SaveTool`.

**Your Output MUST be a valid JSON object matching the GBNF schema.**

**Conversation History (for context):**
{}
"#, history_context);

    vec![
        serde_json::json!({"role": "system", "content": system_prompt}),
        serde_json::json!({"role": "user", "content": user_query}),
    ]
}



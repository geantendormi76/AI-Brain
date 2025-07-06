// orchestrator/src/experts/router.rs

use serde::Deserialize;
use serde_json::Value;

// 1. 【核心改造】在 ToolToCall 枚举中增加新工具
#[derive(Deserialize, Debug, PartialEq)]
pub enum ToolToCall {
    SaveTool,
    RecallTool,
    ModifyTool,       // 新增：用于修改记忆
    DeleteTool,       // 新增：用于删除记忆
    ConfirmationTool, // 新增：用于处理“是/否”的确认回复
    MixedTool,
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

    let system_prompt = format!(
r#"You are a master router. Your only job is to analyze the user's query and decide which single tool to call.

**Tool Descriptions:**
- `SaveTool`: Use when the user wants to **save, remember, record, or take note** of information.
- `RecallTool`: Use when the user wants to **ask a question, find, search, or recall** information. This includes implicit questions like "What are my plans for tomorrow?" or "What is...?".
- `ModifyTool`: Use when the user wants to **change, update, correct, or edit** a previous memory.
- `DeleteTool`: Use when the user wants to **delete, remove, or forget** a memory.
- `ConfirmationTool`: Use ONLY when the user's input is a direct confirmation or rejection like "yes", "no", "confirm", "cancel".
- `MixedTool`: Use ONLY when the query explicitly contains BOTH a save/modify command AND a recall command.
- `NoTool`: Use if the intent is completely unclear, conversational filler, or a greeting.

**CRITICAL DECISION LOGIC:**
1.  **Confirmation First**: Check for confirmation keywords (e.g., "是的", "确认", "取消"). If found, it's `ConfirmationTool`.
2.  **Explicit Commands**: Check for action keywords (e.g., "修改", "删除", "记一下"). If found, choose the corresponding tool.
3.  **Question Detection**: If the query is phrased as a question (e.g., ends with a question mark, starts with "what", "how", "who", "查询"), classify it as `RecallTool`.
4.  **Default to Save**: If the query is a declarative statement of fact (e.g., "Project Titan's core tech is..."), classify it as `SaveTool`.
5.  **Default to NoTool**: If none of the above match, it's likely casual chat. Classify as `NoTool`.

**Your Output MUST be a valid JSON object:**
```json
{{
  "tool_to_call": "string, one of [SaveTool, RecallTool, ModifyTool, DeleteTool, ConfirmationTool, MixedTool, NoTool]"
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

// 【最终正确版】使用正确的 GBNF 语法定义 JSON 结构
pub fn get_routing_gbnf_schema() -> &'static str {
    r#"
root ::= "{" ws "\"tool_to_call\"" ws ":" ws "\"" ("SaveTool" | "RecallTool" | "ModifyTool" | "DeleteTool" | "ConfirmationTool" | "MixedTool" | "NoTool") "\"" ws "}"
ws   ::= ([ \t\n\r])*
"#
}
// orchestrator/src/experts/router.rs

use serde::Deserialize;
use serde_json::Value;

// 定义路由决策的输出结构
#[derive(Deserialize, Debug, PartialEq)]
pub enum ToolToCall {
    SaveTool,
    RecallTool,
    MixedTool, // 用于处理混合意图
    NoTool,    // 当无法识别意图时
}

// 定义路由决策的完整JSON结构
#[derive(Deserialize, Debug)]
pub struct RoutingDecision {
    pub tool_to_call: ToolToCall,
}

// 获取路由决策的 Prompt
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
- `RecallTool`: Use when the user wants to **ask a question, find, search, or recall** information.
- `MixedTool`: Use ONLY when the query explicitly contains BOTH a save command AND a recall command.
- `NoTool`: Use if the intent is completely unclear.

**CRITICAL DECISION LOGIC:**
1.  First, check for explicit recall keywords (e.g., "what", "how", "who", "查询", "是什么"). If found, it's likely a `RecallTool` or `MixedTool`.
2.  Next, check for explicit save keywords (e.g., "remember", "record", "记一下"). If found, it's likely a `SaveTool` or `MixedTool`.
3.  **DEFAULT BEHAVIOR:** If the query does **NOT** contain any explicit command or question keywords and is a simple statement of fact (e.g., "Project Titan's core tech is..."), you **MUST** classify it as a `SaveTool`. This is the default action for declarative sentences.

**Your Output MUST be a valid JSON object:**
```json
{{
  "tool_to_call": "string, one of [SaveTool, RecallTool, MixedTool, NoTool]"
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
// 获取路由决策的 GBNF Schema
pub fn get_routing_gbnf_schema() -> &'static str {
    r#"root ::= "{" ws "\"tool_to_call\":" ws "\"" ("SaveTool" | "RecallTool" | "MixedTool" | "NoTool") "\"" ws "}"
ws ::= [ \t\n\r]*"#
}


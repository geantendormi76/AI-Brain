// orchestrator/src/experts/recall_expert.rs

use serde_json::Value;

// 新增：定义 RecallExpert 的输出结构
#[derive(serde::Deserialize, Debug)]
pub struct RewrittenQuery {
    pub rewritten_query: String,
}

pub fn get_query_rewrite_prompt(user_query: &str, history: &[String]) -> Vec<Value> {
    let history_context = if history.is_empty() { "No history.".to_string() } else { history.join("\n") };

    let system_prompt = format!(
r#"You are a query rewriting expert. Your job is to take a user query and, using the conversation history, rewrite it into a clear, self-contained question.

**CRITICAL INSTRUCTIONS:**
- If the user's query is already a clear question, return it as is.
- Use the history to resolve pronouns and ambiguity.
- Your output MUST be a valid JSON object with a single field "rewritten_query".

**Conversation History:**
{}

**Example:**
- History: "User: 什么是项目Titan？"
- User Query: "那它的核心技术呢？"
- Your Output: {{"rewritten_query": "项目Titan的核心技术是什么？"}}
"#, history_context);

    vec![
        serde_json::json!({"role": "system", "content": system_prompt}),
        serde_json::json!({"role": "user", "content": user_query}),
    ]
}

// 新增：为 RecallExpert 添加 GBNF Schema
pub fn get_query_rewrite_gbnf_schema() -> &'static str {
    r#"root ::= "{" ws "\"rewritten_query\":" ws string ws "}"
string ::= "\"" (
  [^"\\] |
  "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
)* "\""
ws ::= [ \t\n\r]*"#
}

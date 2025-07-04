// orchestrator/src/prompts.rs

use serde_json::Value;

// get_intent_classification_messages 函数本身是正确的，保持不变
pub fn get_intent_classification_messages(user_query: &str) -> Vec<Value> {
    let system_prompt = r#"You are a strict binary classifier. Your task is to classify the user's query into `SaveIntent` or `RecallIntent`.

**Decision Logic (Follow these steps in order):**

**Step 1: Check for Recall Intent.**
Does the query meet ANY of the following criteria?
- It is phrased as a direct question (ending with '？' or '?').
- It contains explicit query keywords like "查询", "查找", "搜索", "是什么", "如何", "为什么".
- It asks for an introduction or explanation (e.g., "介绍一下...", "解释一下...").

*   **If YES, the intent is `RecallIntent`. Stop here.**

**Step 2: Check for Save Intent.**
Does the query contain explicit command keywords like "记一下", "记录", "别忘了", "提醒我"?

*   **If YES, the intent is `SaveIntent`. Stop here.**

**Step 3: Default to Save Intent.**
If the query does not meet any of the criteria in Step 1 or Step 2, it is a statement of fact or an instruction to be remembered.

*   **The intent is `SaveIntent`.**

**Your Output MUST be a valid JSON object:**
```json
{
  "intent": "string, one of [SaveIntent, RecallIntent]"
}
```
**No other text or explanations.**
"#;

    vec![
        serde_json::json!({"role": "system", "content": system_prompt}),
        serde_json::json!({"role": "user", "content": user_query}),
    ]
}

// *** 这是最终的、极度精简的、单行 GBNF Schema ***
// 我们不再定义复杂的规则，而是直接将结构写死。
// 这是为了最大限度地保证解析成功。
pub fn get_intent_gbnf_schema() -> &'static str {
    "root ::= \"{\" [ \\t\\n\\r]* \"\\\"intent\\\"\" [ \\t\\n\\r]* \":\" [ \\t\\n\\r]* (\"\\\"SaveIntent\\\"\" | \"\\\"RecallIntent\\\"\") [ \\t\\n\\r]* \"}\""
}

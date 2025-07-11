use serde_json::Value;


pub fn get_intent_classification_messages(user_query: &str, history: &[String]) -> Vec<Value> {
    let system_prompt = r#"You are an expert task decomposer. Your job is to analyze the user's query and the conversation history, then break the query down into a precise, executable task list.

**CRITICAL INSTRUCTIONS:**
1.  **CONTEXT IS KING:** ALWAYS use the conversation history (Context) to resolve pronouns (like "it", "its", "that") and ambiguous phrases. The user's current query might only make sense with the history.
2.  **BE PRECISE:** When you use the context, create a SINGLE, complete task. Do NOT create a separate task for the context and another for the user's fragment.
3.  **STICK TO THE CURRENT QUERY:** Your output tasks must only correspond to the actions requested in the **current `user_input`**. Do not re-create tasks from the history.

**Your Output MUST be a valid JSON object:**
```json
{
  "tasks": [
    {
      "intent": "string, one of [SaveIntent, RecallIntent]",
      "text": "string, the specific, complete text for this task"
    }
  ]
}
```

**GOOD Example (Correct Context Usage):**
<Context>
User: 帮我查一下“项目Titan”的核心技术
Assistant: 项目Titan的核心技术是分布式图计算。
</Context>
<user_input>那它的主要应用场景呢？</user_input>
<assistant_response>
{
"tasks": [
{
"intent": "RecallIntent",
"text": "项目Titan的主要应用场景"
}
]
}
</assistant_response>


**BAD Example (Incorrect Decomposition):**
<Context>
User: 帮我查一下“项目Titan”的核心技术
Assistant: 项目Titan的核心技术是分布式图计算。
</Context>
<user_input>那它的主要应用场景呢？</user_input>
<assistant_response>
{
"tasks": [
{
"intent": "RecallIntent",
"text": "项目Titan的核心技术"
},
{
"intent": "RecallIntent",
"text": "主要应用场景"
}
]
}
</assistant_response>


Now, decompose the user's query into a JSON task list."#;

    // 构建发送给 LLM 的消息列表
    let mut messages = vec![
        serde_json::json!({"role": "system", "content": system_prompt}),
    ];

    if !history.is_empty() {
        let history_str = history.join("\n");
        let context_message = format!("<Context>\n{}\n</Context>", history_str);
        messages.push(serde_json::json!({"role": "user", "content": context_message}));
    }

    messages.push(serde_json::json!({"role": "user", "content": user_query}));

    messages
}

pub fn get_intent_gbnf_schema() -> &'static str {
    r#"root ::= "{" ws "\"tasks\":" ws "[" ws task (ws "," ws task)* ws "]" ws "}"
task ::= "{" ws "\"intent\":" ws "\"" ("SaveIntent" | "RecallIntent") "\"" ws "," ws "\"text\":" ws string ws "}"
string ::= "\"" (
  [^"\\] |
  "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
)* "\""
ws ::= [ \t\n\r]*"#
}

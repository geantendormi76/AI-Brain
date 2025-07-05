// orchestrator/src/prompts.rs

use serde_json::Value;

pub fn get_intent_classification_messages(user_query: &str) -> Vec<Value> {
    let system_prompt = r#"You are a task decomposer. Your job is to analyze the user's query and break it down into a list of one or more specific tasks: `SaveIntent` or `RecallIntent`. You must identify the core text associated with each task.

**Decision Logic:**
1.  Identify all parts of the query that ask to recall, find, or search for information. For each, create a `RecallIntent` task with the relevant query text.
2.  Identify all parts of the query that ask to save, remember, or take note of information. For each, create a `SaveIntent` task with the relevant fact.
3.  If a part of the query serves both purposes, create two separate tasks.

**Your Output MUST be a valid JSON object with a list of tasks:**
```json
{
  "tasks": [
    {
      "intent": "string, one of [SaveIntent, RecallIntent]",
      "text": "string, the specific text for this task"
    }
  ]
}
```

**Examples:**
<user_input>我最喜欢什么编程语言？</user_input>
<assistant_response>
{
"tasks": [
{
"intent": "RecallIntent",
"text": "我最喜欢什么编程语言？"
}
]
}
</assistant_response>

<user_input>帮我记一下：项目Alpha的截止日期是下周五。</user_input>
<assistant_response>
{
"tasks": [
{
"intent": "SaveIntent",
"text": "项目Alpha的截止日期是下周五。"
}
]
}
</assistant_response>

<user_input>帮我查一下“项目Alpha”的最新进度，并记一下我下周要跟进。</user_input>
<assistant_response>
{
"tasks": [
{
"intent": "RecallIntent",
"text": "“项目Alpha”的最新进度"
},
{
"intent": "SaveIntent",
"text": "我下周要跟进“项目Alpha”。"
}
]
}
</assistant_response>

Now, decompose the user's query into a JSON task list."#;

    vec![
        serde_json::json!({"role": "system", "content": system_prompt}),
        serde_json::json!({"role": "user", "content": user_query}),
    ]
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



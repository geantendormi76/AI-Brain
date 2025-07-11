// orchestrator/src/experts/save_expert.rs

use serde_json::Value;

// 新增：定义 SaveExpert 的输出结构
#[derive(serde::Deserialize, Debug)]
pub struct ExtractedFact {
    pub fact: String,
}

pub fn get_fact_extraction_prompt(user_input: &str) -> Vec<Value> {
    let system_prompt = r#"Your task is to extract the core fact from the user's input. Output ONLY the cleaned, pure fact in a JSON object.

**Your Output MUST be a valid JSON object:**
```json
{
  "fact": "The extracted fact goes here."
}
```

<example>
<user_input>帮我记一下：我最喜欢的编程语言是Rust。</user_input>
<assistant_response>
{
  "fact": "我最喜欢的编程语言是Rust。"
}
</assistant_response>
</example>

Now, extract the core fact from the user's input."#;

    vec![
        serde_json::json!({"role": "system", "content": system_prompt}),
        serde_json::json!({"role": "user", "content": user_input}),
    ]
}

// 新增：为 SaveExpert 添加 GBNF Schema
pub fn get_fact_extraction_gbnf_schema() -> &'static str {
    r#"root ::= "{" ws "\"fact\":" ws string ws "}"
string ::= "\"" (
  [^"\\] |
  "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
)* "\""
ws ::= [ \t\n\r]*"#
}

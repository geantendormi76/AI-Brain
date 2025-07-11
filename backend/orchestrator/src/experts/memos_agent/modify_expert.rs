// orchestrator/src/experts/modify_expert.rs

use serde::Deserialize;
use serde_json::Value;

// 定义 ModifyExpert 生成修改后文本的输出结构
#[derive(Deserialize, Debug)]
pub struct ModifiedText {
    pub modified_text: String,
}

// 获取生成修改后文本的 Prompt
pub fn get_text_modification_prompt(original_text: &str, user_request: &str) -> Vec<Value> {
    let system_prompt = format!(
r#"You are a precise text editor. Your task is to take an original text and a user's modification request, then output the fully rewritten, new version of the text.

**CRITICAL INSTRUCTIONS:**
- You must output the entire new text, not just the changed part.
- The new text must incorporate the user's requested change.
- Your output MUST be a valid JSON object with a single field "modified_text".

**Original Text:**
---
{}
---

**User's Modification Request:**
---
{}
---

Now, generate the new, complete text based on the user's request."#, original_text, user_request);

    vec![
        serde_json::json!({"role": "system", "content": system_prompt}),
        // 注意：这里没有 user role，因为所有信息都在 system prompt 里了
    ]
}



// 【最终正确版】使用正确的、通用的 GBNF 语法定义 JSON 结构
pub fn get_text_modification_gbnf_schema() -> &'static str {
    r#"
root   ::= "{" ws "\"modified_text\"" ws ":" ws string ws "}"
# 采用官方示例中的 string 定义，允许所有合法的 JSON 字符串字符
string ::= "\"" (
  [^"\\\\] |
  "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
)* "\""
ws     ::= ([ \t\n\r])*
"#
}
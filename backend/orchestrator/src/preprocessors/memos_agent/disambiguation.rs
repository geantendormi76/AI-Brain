// orchestrator/src/experts/disambiguation_expert.rs

use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize, Debug)]
pub struct DisambiguatedText {
    pub disambiguated_text: String,
}

pub fn get_disambiguation_prompt(user_input: &str, history: &[String]) -> Vec<Value> {
    let history_context = if history.is_empty() { 
        "No conversation history.".to_string() 
    } else { 
        history.join("\n") 
    };

    // 【最终修正】严格限定 DisambiguationExpert 的职责
    let system_prompt = format!(
r#"You are a text rewriting expert. Your ONLY job is to rewrite the user's input into a self-contained phrase by resolving pronouns (like "it", "that", "他", "它") based on the conversation history.

**CRITICAL INSTRUCTIONS:**
1.  **ONLY Resolve Pronouns:** Your primary task is to replace pronouns with the specific entities from the history.
2.  **Maintain Original Intent:** If the input is a question, the output MUST remain a question. If it's a statement, it MUST remain a statement. DO NOT answer the question or change the intent.
3.  **If No Pronouns:** If the input is already clear, return it IDENTICALLY.
4.  **Output JSON:** Your output MUST be a valid JSON object: {{"disambiguated_text": "..."}}

---
**Example 1: Question with Pronoun**
<History>
User: 我正在学习 MemosAgent 项目。
</History>
<User_Input>
它的优点是什么？
</User_Input>
<Your_Output>
{{"disambiguated_text": "MemosAgent项目的优点是什么？"}}
</Your_Output>
---
**Example 2: Statement with Pronoun**
<History>
User: 我正在学习 MemosAgent 项目。
</History>
<User_Input>
记一下：它的优点是高内聚低耦合。
</User_Input>
<Your_Output>
{{"disambiguated_text": "记一下：MemosAgent项目的优点是高内聚低耦合。"}}
</Your_Output>
---
**Example 3: Already Clear**
<History>
...
</History>
<User_Input>
周五下午有什么事？
</User_Input>
<Your_Output>
{{"disambiguated_text": "周五下午有什么事？"}}
</Your_Output>
---

**Current Task:**

<History>
{}
</History>
<User_Input>
{}
</User_Input>
<Your_Output>
"#, history_context, user_input);

    vec![
        serde_json::json!({"role": "system", "content": system_prompt}),
    ]
}

pub fn get_disambiguation_gbnf_schema() -> &'static str {
    r#"
root   ::= "{" ws "\"disambiguated_text\"" ws ":" ws string ws "}"
string ::= "\"" (
  [^"\\\\] |
  "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
)* "\""
ws     ::= ([ \t\n\r])*
"#
}
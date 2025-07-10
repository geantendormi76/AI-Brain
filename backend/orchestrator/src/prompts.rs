
pub fn get_intent_gbnf_schema() -> &'static str {
    r#"root ::= "{" ws "\"tasks\":" ws "[" ws task (ws "," ws task)* ws "]" ws "}"
task ::= "{" ws "\"intent\":" ws "\"" ("SaveIntent" | "RecallIntent") "\"" ws "," ws "\"text\":" ws string ws "}"
string ::= "\"" (
  [^"\\] |
  "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
)* "\""
ws ::= [ \t\n\r]*"#
}

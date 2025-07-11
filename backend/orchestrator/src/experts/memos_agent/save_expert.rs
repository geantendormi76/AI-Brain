
// backend/orchestrator/src/experts/memos_agent/save_expert.rs
// 【智能增强版 - V1】

use serde::Deserialize;
use serde_json::Value;
use memos_core::FactMetadata;

// =================================================================
// == 1. 升级输出结构体，增加 metadata 字段 ==
// =================================================================
// 这个结构体现在可以同时捕获事实和与之相关的元数据（标签）

#[derive(Deserialize, Debug)]
pub struct ExtractedFact {
    pub fact: String,
    pub metadata: Option<FactMetadata>,
}

// ======================= 结构体定义结束 =======================


// =================================================================
// == 2. 升级Prompt，教会LLM如何提取标签 ==
// =================================================================
// 我们使用了“多示例(Few-shot)”学习，为LLM提供了高质量的范例

pub fn get_fact_extraction_prompt(user_input: &str) -> Vec<Value> {
    let system_prompt = r#"你的任务是从用户的输入中，提取出“核心事实”和相关的“主题标签”，并以一个严格的JSON对象格式输出。

**核心指令:**
1.  **提取事实 (fact)**: 将用户想要记录的核心信息，提炼成一个简洁、完整的陈述句。
2.  **提取标签 (topics)**: 提取出与事实最相关的1到3个核心关键词或主题，作为标签。标签应该是名词或动名词。
3.  **JSON格式**: 你的输出必须是严格的、不包含任何其他文本的JSON对象。

**输出格式:**
```json
{
  "fact": "提取出的核心事实。",
  "metadata": {
    "topics": ["标签一", "标签二"]
  }
}
```

---
**示例 1:**
<用户输入>
帮我记一下，我的个人网站是 example.com
</用户输入>

<你的输出>
```json
{
  "fact": "我的个人网站是 example.com。",
  "metadata": {
    "topics": ["个人网站", "网站地址"]
  }
}
```
---
**示例 2:**
<用户输入>
我对于提升团队生产力的想法是，应该减少无效会议，多采用异步沟通。
</用户输入>

<你的输出>
```json
{
  "fact": "提升团队生产力的想法是减少无效会议，并多采用异步沟通。",
  "metadata": {
    "topics": ["生产力", "工作效率", "异步沟通"]
  }
}
```
---

现在，请处理以下用户输入。
"#;

    vec![
        serde_json::json!({"role": "system", "content": system_prompt}),
        serde_json::json!({"role": "user", "content": user_input}),
    ]
}
// ======================= Prompt定义结束 =======================


// =================================================================
// == 3. 升级GBNF语法，确保LLM输出的JSON结构100%正确 ==
// =================================================================
// backend/orchestrator/src/experts/memos_agent/save_expert.rs
// 请用这个“官方文档对齐版”的函数，替换掉旧的函数

pub fn get_fact_extraction_gbnf_schema() -> &'static str {
    // 【核心修正】：这个版本严格遵循了官方json.gbnf的模块化和递归定义风格，
    // 确保了最大的兼容性和稳定性。
    r#"
root   ::= object

# 定义一个通用的 object，可以包含任意键值对
object ::= "{" ws (
    string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

# 定义一个通用的 array，可以包含任意值
array  ::= "[" ws (
    value
    ("," ws value)*
  )? "]" ws

# 定义一个 value 可以是任何JSON基本类型
value  ::= object | array | string | number | ("true" | "false" | "null") ws

# 定义 string 和 number（从官方示例中简化，因为我们不需要那么复杂）
string ::= "\"" (
  [^"\\\\] |
  "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
)* "\"" ws

number ::= ("-"? [0-9]+ ("." [0-9]+)?) ws

# 定义空白字符
ws ::= ([ \t\n\r])*
"#
}

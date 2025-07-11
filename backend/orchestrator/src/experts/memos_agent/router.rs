
// backend/orchestrator/src/experts/memos_agent/router.rs
// 【黄金标准对齐版 - V2】

use serde::Deserialize;
use serde_json::Value;

// 结构体定义保持不变，它们是 Orchestrator 需要的公共接口
#[derive(Deserialize, Debug, PartialEq)]
pub enum ToolToCall {
    SaveTool,
    RecallTool,
    ModifyTool,
    DeleteTool,
    ConfirmationTool,
    NoTool,
}

#[derive(Deserialize, Debug)]
pub struct RoutingDecision {
    pub tool_to_call: ToolToCall,
}

// =================================================================
// == 核心：为混合路由架构定制的全新 get_routing_prompt 函数 ==
// =================================================================
pub fn get_routing_prompt(user_query: &str, history: &[String]) -> Vec<Value> {
    let history_context = if history.is_empty() {
        "无对话历史。".to_string()
    } else {
        history.join("\n")
    };

    // --- 【V6.1 黄金标准版】混合路由配套指令 ---
    // 这个Prompt的预设前提是：所有明确的修改/删除/确认指令已经被代码层的确定性规则拦截了。
    // 所以，它的任务被极大地简化了。
    let system_prompt = format!(
    r#"你是一个判断用户意图的专家。你的任务非常简单：判断用户的输入是一个“需要记录的陈述”还是一个“需要回答的问题”。

**工具描述:**
- `SaveTool`: 用于记录任何事实、想法、陈述或笔记。
- `RecallTool`: 用于回答任何明确或隐含的问题。
- `NoTool`: **仅用于**处理非常简短的、无意义的闲聊，如“你好”、“哈哈”。

**关键决策逻辑 (二选一):**
1.  **判断是否为问题**: 如果用户的输入听起来像一个问题（例如，以“是什么”、“怎么办”、“查询”开头，或以“？”结尾），或者它在寻求解释和信息，那么**必须**选择 `RecallTool`。
2.  **默认为保存**: 如果用户的输入不是一个明确的问题，那么它就是一条需要记录的信息。**必须**选择 `SaveTool`。这是你的主要默认选项。例如，“我明天要去北京”或“今天天气不错”都应被记录。

**重要提醒**:
- 你不需要处理“修改”或“删除”的意图，它们已经被其他程序处理了。
- 你的任务不是回答问题，只是做出分类决策。

**输出格式:**
你的回复必须是一个只包含 "tool_to_call" 字段的JSON对象。例如:
```json
{{
"tool_to_call": "SaveTool"
}}
```

**你的输出必须是严格遵守上述规则的、有效的JSON对象。**

**对话历史 (用于参考):**
{}
"#, history_context);

    vec![
        serde_json::json!({"role": "system", "content": system_prompt}),
        serde_json::json!({"role": "user", "content": user_query}),
    ]
}

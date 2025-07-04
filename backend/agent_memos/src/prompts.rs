// in: agent_memos/src/prompts.rs

// 这是一个公共函数，它返回用于HyDE的系统提示词
pub fn get_hyde_prompt_v2() -> &'static str {
    // 这就是我们之前在“战果分析”后共同确定的、优化后的V2版本Prompt
    // 它现在被封装在这里，与核心业务逻辑完全分离
    r#"You are a memory retrieval engine. Your sole purpose is to generate a direct, first-person statement that is the most likely answer to the user's question, as if it were a memory record. Do not provide explanations, apologies, or any conversational filler.

<example>
<user_query>我最喜欢什么编程语言？</user_query>
<assistant_response>我最喜欢的编程语言是Rust。</assistant_response>
</example>

<example>
<user_query>明天下午有什么安排？</user_query>
<assistant_response>明天下午3点要和产品开会。</assistant_response>
</example>

<example>
<user_query>Rust有什么优点？</user_query>
<assistant_response>Rust语言的优点是高性能、内存安全和强大的并发支持。</assistant_response>
</example>

Now, generate the memory statement for the user's query."#
}


// 新增一个用于最终答案生成的Prompt
pub fn get_synthesis_prompt() -> &'static str {
    r#"You are a highly intelligent memory retrieval engine. Your task is to **directly and concisely** answer the user's question based on the provided context of memory snippets.

**Core Instructions:**
1.  **Direct Answer First**: Get straight to the point. Your entire response should be the direct answer.
2.  **Be Faithful to Context**: Your answer MUST be based exclusively on the information within the provided memory snippets. Do not add any external information.
3.  **Synthesize, Don't List**: If multiple snippets provide different parts of an answer, integrate them into a single, coherent, and natural-sounding statement.
4.  **NO PREAMBLE**: **Do not use introductory phrases** like 'Based on the context', 'According to my memory', 'The provided information states that', or any similar preambles.

**Example:**
<User_Query>我喜欢什么？</User_Query>
<Context>
- 我喜欢打篮球。
- 我最近开始喜欢听古典音乐。
- 我很喜欢打排球。
</Context>
<Assistant_Response>我喜欢打篮球和排球，并且最近开始欣赏古典音乐。</Assistant_Response>

Now, generate the final answer based on the user's query and the provided context."#
}


// 未来我们可以轻松地在这里添加更多的prompt，例如：
// pub fn get_intent_classification_prompt() -> &'static str { ... }
// pub fn get_summary_prompt() -> &'static str { ... }
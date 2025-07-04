// agent_memos/src/query_expander.rs

use std::collections::HashMap;
use jieba_rs::Jieba;

// 定义查询扩展器的结构体
pub struct QueryExpander {
    jieba: Jieba,
    synonym_map: HashMap<String, Vec<String>>,
}

impl QueryExpander {
    // 创建一个新的 QueryExpander 实例
    pub fn new() -> Self {
        let mut synonym_map = HashMap::new();
        // 初始化一个简单的、硬编码的同义词词典
        // 未来这部分可以从外部文件加载
        synonym_map.insert("会议".to_string(), vec!["周会".to_string(), "讨论会".to_string()]);
        synonym_map.insert("喜欢".to_string(), vec!["偏好".to_string(), "最爱".to_string()]);
        synonym_map.insert("优点".to_string(), vec!["优势".to_string(), "好处".to_string()]);
        synonym_map.insert("如何".to_string(), vec!["怎样".to_string(), "怎么".to_string()]);
        synonym_map.insert("运作".to_string(), vec!["工作".to_string(), "运行".to_string()]);

        Self {
            jieba: Jieba::new(),
            synonym_map,
        }
    }

    // 核心的查询扩展方法
    pub fn expand(&self, original_query: &str) -> Vec<String> {
        println!("[QueryExpander] Expanding query: '{}'", original_query);
        let mut expansions = vec![original_query.to_string()];

        // --- 1. 同义词扩展 ---
        let tokens = self.jieba.cut(original_query, false); // 使用精确模式分词
        for token in &tokens {
            if let Some(synonyms) = self.synonym_map.get(*token) {
                for synonym in synonyms {
                    // 生成一个新的查询，其中一个词被同义词替换
                    let new_query = original_query.replace(token, synonym);
                    if !expansions.contains(&new_query) {
                        expansions.push(new_query);
                    }
                }
            }
        }

        // --- 2. 关键词重组扩展 ---
        // 我们复用 MemosAgent 中的 extract_keywords 逻辑，但在这里简化实现
        let keywords: Vec<&str> = self.jieba.cut_for_search(original_query, true)
            .into_iter()
            .filter(|k| !is_stop_word(k)) // 使用一个简单的停用词过滤函数
            .collect();
        
        if !keywords.is_empty() {
            let keyword_query = keywords.join(" ");
            if !expansions.contains(&keyword_query) {
                expansions.push(keyword_query);
            }
        }

        println!("[QueryExpander] Generated expansions: {:?}", expansions);
        expansions
    }
}

// 简单的停用词判断函数
fn is_stop_word(word: &str) -> bool {
    // 实际项目中，应该使用 stop-words 库或一个更完整的列表
    matches!(word, "的" | "是" | "了" | "吗" | "呢" | "吧" | "啊" | "我" | "你" | "他" | "她" | "它")
}
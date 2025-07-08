// in: agent_memos/src/memory_tier_manager.rs

#[derive(Debug, PartialEq)]
pub enum MemoryTier {
    Active,
    Archive,
}

/// 根据记忆内容，决定其应被放入哪个层级。
/// 这是未来可以持续优化的智能决策核心。
pub fn determine_tier(content: &str) -> MemoryTier {
    // 初版智能规则：基于关键词和内容长度的决策
    let archival_keywords = ["总结", "原理", "复盘", "思考", "报告", "长期规划"];
    
    let contains_archival_keyword = archival_keywords.iter().any(|&kw| content.contains(kw));
    let is_very_long = content.chars().count() > 500;

    if contains_archival_keyword || is_very_long {
        println!("[TierManager] Content classified as 'Archive' due to keywords or length.");
        MemoryTier::Archive
    } else {
        println!("[TierManager] Content classified as 'Active'.");
        MemoryTier::Active
    }
}
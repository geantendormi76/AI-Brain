// orchestrator/src/experts/confirmation_expert.rs

// 定义确认决策的枚举
#[derive(Debug, PartialEq)]
pub enum ConfirmationDecision {
    Affirm, // 肯定
    Deny,   // 否定
    Unclear,// 不明确
}

/// 解析用户的文本，判断其确认意图
pub fn parse_confirmation(text: &str) -> ConfirmationDecision {
    let lower_text = text.trim().to_lowercase();

    // 定义肯定和否定的关键词
    let affirm_keywords = ["yes", "ok", "confirm", "y", "是的", "确认", "好的", "对", "中", "是", "确定", "好"];
    let deny_keywords = ["no", "cancel", "n", "不是", "不用了", "取消", "否", "不"];

    if affirm_keywords.iter().any(|&s| lower_text == s) {
        return ConfirmationDecision::Affirm;
    }
    
    if deny_keywords.iter().any(|&s| lower_text == s) {
        return ConfirmationDecision::Deny;
    }

    ConfirmationDecision::Unclear
}
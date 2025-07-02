use memos_core::{Agent, Command, Response};
use async_trait::async_trait;
use r2d2_sqlite::SqliteConnectionManager;
use r2d2::Pool;
use chrono::Utc;

type DbPool = Pool<SqliteConnectionManager>;

pub struct MemosAgent {
    pool: DbPool,
}

// ====================================================================
//  为 MemosAgent 实现其自身的方法
//  所有 MemosAgent 的私有/公有方法都应该放在这个 impl 块里
// ====================================================================
impl MemosAgent {
    pub fn new() -> Result<Self, anyhow::Error> {
        let home_dir = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
        let db_dir = home_dir.join(".memos_agent");
        std::fs::create_dir_all(&db_dir)?;
        let db_path = db_dir.join("memos.db");

        let manager = SqliteConnectionManager::file(db_path);
        let pool = Pool::new(manager)?;

        let conn = pool.get()?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )",
            [],
        )?;
        println!("[MemosAgent-DB] Database initialized and 'facts' table checked/created.");

        Ok(Self { pool })
    }

    fn save_memo(&self, content: &str) -> Result<(), anyhow::Error> {
        let conn = self.pool.get()?;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO facts (content, created_at) VALUES (?1, ?2)",
            &[content, &now],
        )?;
        println!("[MemosAgent] Successfully saved memo to database.");
        Ok(())
    }

    fn query_memos(&self) -> Result<String, anyhow::Error> {
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare("SELECT id, content, created_at FROM facts ORDER BY id DESC")?;
        
        let memos = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let content: String = row.get(1)?;
            let created_at: String = row.get(2)?;
            Ok(format!("[ID: {}] [{}] {}", id, created_at, content))
        })?;

        let mut results = Vec::new();
        for memo in memos {
            results.push(memo?);
        }

        if results.is_empty() {
            Ok("您还没有记录任何备忘。".to_string())
        } else {
            Ok(results.join("\n"))
        }
    }

    fn delete_memo(&self, id: i64) -> Result<String, anyhow::Error> {
        let conn = self.pool.get()?;
        let rows_affected = conn.execute("DELETE FROM facts WHERE id = ?1", &[&id])?;

        if rows_affected > 0 {
            println!("[MemosAgent] Successfully deleted memo with id: {}", id);
            Ok(format!("好的，已经删除ID为 {} 的备忘。", id))
        } else {
            println!("[MemosAgent] Memo with id: {} not found for deletion.", id);
            Ok(format!("未找到ID为 {} 的备忘，无法删除。", id))
        }
    }

    // --- 将 update_memo 函数移动到这里！ ---
    fn update_memo(&self, id: i64, new_content: &str) -> Result<String, anyhow::Error> {
        let conn = self.pool.get()?;
        let now = Utc::now().to_rfc3339();
        
        let rows_affected = conn.execute(
            "UPDATE facts SET content = ?1, created_at = ?2 WHERE id = ?3",
            &[new_content, &now, &id.to_string()],
        )?;

        if rows_affected > 0 {
            println!("[MemosAgent] Successfully updated memo with id: {}", id);
            Ok(format!("好的，已经将ID为 {} 的备忘更新。", id))
        } else {
            println!("[MemosAgent] Memo with id: {} not found for update.", id);
            Ok(format!("未找到ID为 {} 的备忘，无法更新。", id))
        }
    }
} // <--- MemosAgent 自身方法的 impl 块在这里结束


// ====================================================================
//  为 MemosAgent 实现 Agent Trait
//  这是让 MemosAgent 成为一个合格“专家”的证明
// ====================================================================
#[async_trait]
impl Agent for MemosAgent {
    fn name(&self) -> &'static str {
        "memos_agent"
    }

    fn interests(&self) -> &[&'static str] {
        &[
            "提醒", "记一下", "别忘了", "帮我记",
            "查询", "查找", "我记了什么", "看看备忘",
            "删除", "忘了那条", "去掉",
            "更新", "修改", "改成"
        ]
    }

    async fn handle_command(&self, command: &Command) -> Result<Response, anyhow::Error> {
        match command {
            Command::ProcessText(text) => {
                println!("[MemosAgent] Received text: {}", text);

                let is_query_intent = self.interests()[4..8].iter().any(|&interest| text.contains(interest));
                let is_delete_intent = self.interests()[8..11].iter().any(|&interest| text.contains(interest));
                let is_update_intent = self.interests()[11..].iter().any(|&interest| text.contains(interest));

                if is_update_intent {
                    let id_str = text.chars().filter(|c| c.is_digit(10)).collect::<String>();
                    if let Ok(id) = id_str.parse::<i64>() {
                        if let Some(update_keyword) = self.interests()[11..].iter().find(|&kw| text.contains(kw)) {
                            if let Some(content_start_index) = text.find(update_keyword) {
                                let new_content = text[content_start_index..].trim();
                                let result_message = self.update_memo(id, new_content)?;
                                return Ok(Response::Text(result_message));
                            }
                        }
                        Ok(Response::Text("无法解析要更新的内容。请使用格式：'更新第X条为...'".to_string()))
                    } else {
                        Ok(Response::Text("请提供一个有效的备忘ID来更新，例如'更新第3条...'".to_string()))
                    }
                } else if is_delete_intent {
                    let id_str = text.chars().filter(|c| c.is_digit(10)).collect::<String>();
                    if let Ok(id) = id_str.parse::<i64>() {
                        let result_message = self.delete_memo(id)?;
                        Ok(Response::Text(result_message))
                    } else {
                        Ok(Response::Text("请提供一个有效的备V忘ID来删除，例如'删除第3条'".to_string()))
                    }
                } else if is_query_intent {
                    let memos_summary = self.query_memos()?;
                    Ok(Response::Text(format!("好的，这是您记录的备忘：\n---\n{}\n---", memos_summary)))
                } else {
                    self.save_memo(text)?;
                    Ok(Response::Text("好的，我已经帮您记下了。".to_string()))
                }
            }
        }
    }
}
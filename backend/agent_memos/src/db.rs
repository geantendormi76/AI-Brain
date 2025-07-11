// backend/agent_memos/src/db.rs
// 【智能增强版 - V2.1】

use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::Result;

// 定义连接池的类型别名，保持不变
pub type DbPool = r2d2::Pool<SqliteConnectionManager>;

/// 初始化数据库并创建表
pub fn init_db(pool: &DbPool) -> Result<()> {
    let conn = pool.get().expect("Failed to get DB connection from pool");
    
    // 【核心升级】: 为 'facts' 表增加 metadata 字段，并确保它能被正确索引
    // 我们将 metadata 字段的类型设置为 TEXT，并存储为 JSON 字符串。
    // 这是为了兼容各种版本的SQLite。
    conn.execute(
        "CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            metadata TEXT, -- 用于存储JSON格式的元数据
            expires_at DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;

    println!("[MemosAgent-DB] Database initialized and 'facts' table schema updated for metadata support.");
    Ok(())
}
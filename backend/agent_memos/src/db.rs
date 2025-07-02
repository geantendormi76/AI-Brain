use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::Result;

// 定义连接池的类型别名，方便使用
pub type DbPool = r2d2::Pool<SqliteConnectionManager>;

/// 初始化数据库并创建表
pub fn init_db(pool: &DbPool) -> Result<()> {
    // 从连接池中获取一个连接
    let conn = pool.get().expect("Failed to get DB connection from pool");
    
    // 创建 'facts' 表，如果它不存在的话
    // 我们在这里预留了未来需要的字段
    conn.execute(
        "CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;

    println!("[MemosAgent-DB] Database initialized and 'facts' table created.");
    Ok(())
}
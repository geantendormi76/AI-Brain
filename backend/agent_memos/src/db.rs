use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::Result;

// 定义连接池的类型别名，方便使用
pub type DbPool = r2d2::Pool<SqliteConnectionManager>;

/// 初始化数据库并创建表
pub fn init_db(pool: &DbPool) -> Result<()> {
    // 从连接池中获取一个连接
    let conn = pool.get().expect("Failed to get DB connection from pool");
    
    // V5.0 升级：创建 'facts' 表，增加 expires_at 和 metadata 字段
    // id 已经是主键，我们确保它自增
    conn.execute(
        "CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            metadata TEXT,
            expires_at DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;

    println!("[MemosAgent-DB] Database initialized and 'facts' table created/updated for V5.0.");
    Ok(())
}
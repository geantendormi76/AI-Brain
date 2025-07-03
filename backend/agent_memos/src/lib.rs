use memos_core::{Agent, Command, Response};
use async_trait::async_trait;
use r2d2_sqlite::SqliteConnectionManager;
use r2d2::Pool;
use chrono::Utc;

use qdrant_client::{
    Payload,
    Qdrant,
    qdrant::{
        CreateCollectionBuilder,
        Distance,
        PointStruct,
        VectorParamsBuilder,
        UpsertPointsBuilder,
        // --- 核心修正：精确导入正确的Vector和DenseVector ---
        Vector, // 直接从 qdrant_client::qdrant 导入 Vector 结构体
        vector::Vector as QdrantVectorEnums, // <-- 关键修正：从 `vector` 模块导入 `Vector` 枚举，并取别名
        DenseVector, // 从 qdrant_client::qdrant 导入 DenseVector
    }
};
use serde_json::json;

type DbPool = Pool<SqliteConnectionManager>;
const COLLECTION_NAME: &str = "memos";
const EMBEDDING_DIM: u64 = 384; 

pub struct MemosAgent {
    sql_pool: DbPool,
    qdrant_client: Qdrant, 
}

impl MemosAgent {
    pub async fn new(qdrant_url: &str) -> Result<Self, anyhow::Error> {
        // --- 初始化SQLite (不变) ---
        let home_dir = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
        let db_dir = home_dir.join(".memos_agent");
        std::fs::create_dir_all(&db_dir)?;
        let sql_db_path = db_dir.join("memos.db");
        let manager = SqliteConnectionManager::file(sql_db_path);
        let sql_pool = Pool::new(manager)?;
        let conn = sql_pool.get()?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, content TEXT NOT NULL, created_at TEXT NOT NULL)",
            [],
        )?;
        println!("[MemosAgent-DB] SQLite database initialized.");

        // --- Qdrant客户端初始化 (不变) ---
        let qdrant_client = Qdrant::from_url(qdrant_url)
            .build()?; 
        println!("[MemosAgent-DB] Qdrant client initialized.");
        
        // 检查Collection是否存在，如果不存在则创建 (不变)
        let collection_exists = qdrant_client.collection_exists(COLLECTION_NAME).await?;
        if !collection_exists {
            println!("[MemosAgent-DB] Collection '{}' not found. Creating...", COLLECTION_NAME);
            qdrant_client.create_collection(
                CreateCollectionBuilder::new(COLLECTION_NAME)
                    .vectors_config(VectorParamsBuilder::new(EMBEDDING_DIM, Distance::Cosine))
            ).await?;
            println!("[MemosAgent-DB] Qdrant collection '{}' created.", COLLECTION_NAME);
        } else {
            println!("[MemosAgent-DB] Found existing collection '{}'.", COLLECTION_NAME);
        }

        Ok(Self { sql_pool, qdrant_client })
    }

    fn get_mock_embedding(&self, text: &str) -> Result<Vec<f32>, anyhow::Error> {
        println!("[MemosAgent-MockEmbed] Generating MOCK vector for text: '{}'", text);
        let mock_vector = vec![0.0; EMBEDDING_DIM as usize];
        Ok(mock_vector)
    }

    async fn save_memo(&self, content: &str) -> Result<(), anyhow::Error> {
        let conn = self.sql_pool.get()?;
        let now = Utc::now().to_rfc3339();
        conn.execute("INSERT INTO facts (content, created_at) VALUES (?1, ?2)", &[content, &now])?;
        let memo_id = conn.last_insert_rowid();
        println!("[MemosAgent-DB] Saved to SQLite with ID: {}", memo_id);

        let vector_data = self.get_mock_embedding(content)?;
        println!("[MemosAgent-Embed] Generated {}d MOCK vector.", vector_data.len());

        // --- 确认这里的Vector和DenseVector使用正确 ---
        let qdrant_vector = Vector {
            // 这里的 `vector` 是 `qdrant_client::qdrant::Vector` 结构体内部的 `oneof` 字段
            vector: Some(QdrantVectorEnums::Dense(DenseVector { // 使用别名 `QdrantVectorEnums`
                data: vector_data,
            })),
            ..Default::default()
        };


        let payload: Payload = json!({
            "content": content,
            "created_at": now
        }).try_into().unwrap();

        let points = vec![PointStruct::new(memo_id as u64, qdrant_vector, payload)];
        
        self.qdrant_client.upsert_points(
            UpsertPointsBuilder::new(COLLECTION_NAME.to_string(), points)
        ).await?;
        println!("[MemosAgent-DB] Upserted point to Qdrant with ID: {}", memo_id);
        
        Ok(())
    }
}

#[async_trait]
impl Agent for MemosAgent {
    fn name(&self) -> &'static str { "memos_agent" }
    fn interests(&self) -> &[&'static str] { &[""] }

    async fn handle_command(&self, command: &Command) -> Result<Response, anyhow::Error> {
        match command {
            Command::ProcessText(text) => {
                self.save_memo(text).await?;
                Ok(Response::Text("已使用模拟数据，成功记录并向量化。".to_string()))
            }
        }
    }
}
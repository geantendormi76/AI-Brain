// agent_memos/src/lib.rs

// 1. 导入 db 模块
mod db; 
mod query_expander;
use query_expander::QueryExpander;
use memos_core::{Agent, Command, Response};
use async_trait::async_trait;
use r2d2_sqlite::SqliteConnectionManager;
use r2d2::Pool;
use chrono::Utc;
use std::collections::HashMap;
// 2. 导入 qdrant 删除点所需的新类型
use qdrant_client::qdrant::{PointStruct, Vector, VectorParamsBuilder, CreateCollectionBuilder, Distance, UpsertPointsBuilder, SearchPointsBuilder, ScoredPoint, point_id, vector::Vector as QdrantVectorEnums, DenseVector, PointsIdsList, DeletePointsBuilder};
use qdrant_client::{Payload, Qdrant};
use serde_json::json;
use std::any::Any;

// 移除所有与LLM相关的 struct 定义（EmbeddingRequest除外）
#[derive(serde::Serialize)]
struct EmbeddingRequest<'a> {
    content: &'a str,
}
#[derive(Debug, serde::Deserialize)]
struct EmbeddingData {
    embedding: Vec<Vec<f32>>,
}
#[derive(Debug, serde::Deserialize)]
struct EmbeddingResponse(Vec<EmbeddingData>);

type DbPool = Pool<SqliteConnectionManager>;
const COLLECTION_NAME: &str = "memos";
const EMBEDDING_DIM: u64 = 512; 

pub struct MemosAgent {
    sql_pool: DbPool,
    qdrant_client: Qdrant,
    query_expander: QueryExpander,
}

impl MemosAgent {
    pub async fn new(qdrant_url: &str) -> Result<Self, anyhow::Error> {
        let home_dir = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
        let db_dir = home_dir.join(".memos_agent");
        std::fs::create_dir_all(&db_dir)?;
        let sql_db_path = db_dir.join("memos.db");
        let manager = SqliteConnectionManager::file(sql_db_path);
        let sql_pool = Pool::new(manager)?;
        
        // 3. 【核心改造】调用统一的 db::init_db，移除旧的建表逻辑
        db::init_db(&sql_pool)?;
        println!("[MemosAgent-DB] SQLite database initialization delegated to db::init_db.");

        let qdrant_client = Qdrant::from_url(qdrant_url).build()?;
        println!("[MemosAgent-DB] Qdrant client initialized.");

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

        Ok(Self { 
            sql_pool, 
            qdrant_client,
            query_expander: QueryExpander::new(), // 初始化 QueryExpander
        })
    }

    // 新增：支持融合多路搜索结果的 RRF 函数
    fn reciprocal_rank_fusion_multi(
        &self, 
        ranked_lists: Vec<Vec<ScoredPoint>>,
        k: u32
    ) -> Vec<ScoredPoint> {
        println!("[MemosAgent-RRF] Fusing {} ranked lists...", ranked_lists.len());
        
        let mut fused_scores: HashMap<u64, f32> = HashMap::new();
        let mut point_data: HashMap<u64, ScoredPoint> = HashMap::new();

        for list in ranked_lists {
            for (rank, point) in list.into_iter().enumerate() {
                if let Some(point_id::PointIdOptions::Num(memo_id)) = point.id.as_ref().and_then(|p| p.point_id_options.as_ref()) {
                    let score = 1.0 / (k as f32 + (rank + 1) as f32);
                    *fused_scores.entry(*memo_id).or_insert(0.0) += score;
                    point_data.entry(*memo_id).or_insert(point);
                }
            }
        }

        let mut sorted_fused_results: Vec<(u64, f32)> = fused_scores.into_iter().collect();
        sorted_fused_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let final_ranked_list: Vec<ScoredPoint> = sorted_fused_results
            .into_iter()
            .filter_map(|(id, rrf_score)| {
                if let Some(mut point) = point_data.remove(&id) {
                    point.score = rrf_score;
                    Some(point)
                } else {
                    None
                }
            })
            .collect();

        println!("[MemosAgent-RRF] Fusion completed. Final ranked list has {} items.", final_ranked_list.len());
        final_ranked_list
    }


    // --- `save` 方法保持核心逻辑，但可以简化，因为它总是被调用
    pub async fn save(&self, content: &str) -> Result<(), anyhow::Error> {
        println!("[MemosAgent] Saving memo: '{}'", content);
        let conn = self.sql_pool.get()?;
        let now = Utc::now().to_rfc3339();
        conn.execute("INSERT INTO facts (content, created_at) VALUES (?1, ?2)", &[content, &now])?;
        let memo_id = conn.last_insert_rowid();
        println!("[MemosAgent-DB] Saved to SQLite with ID: {}", memo_id);
        let vector_data = self.get_embedding(content).await?;
        let qdrant_vector = Vector {
            vector: Some(QdrantVectorEnums::Dense(DenseVector { data: vector_data })),
            ..Default::default()
        };
        let payload: Payload = json!({"content": content,"created_at": now}).try_into().unwrap();
        let points = vec![PointStruct::new(memo_id as u64, qdrant_vector, payload)];
        self.qdrant_client.upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME.to_string(), points)).await?;
        println!("[MemosAgent-DB] Upserted point to Qdrant with ID: {}", memo_id);
        Ok(())
    }

    // --- 全新的 `recall` 方法，零LLM调用 ---
    pub async fn recall(&self, query_text: &str) -> Result<Vec<ScoredPoint>, anyhow::Error> {
        println!("[MemosAgent] Recalling V2.2 for: '{}'", query_text);

        let expansions = self.query_expander.expand(query_text);
        let original_query = expansions.get(0).unwrap_or(&query_text.to_string()).clone();
        let expanded_query_str = expansions.join(" ");

        // 修正：提高阈值以进行更严格的过滤。这是一个需要根据模型和数据进行调整的经验值。
        const VECTOR_SCORE_THRESHOLD: f32 = 0.5;

        // 步骤 2: 三路并行检索
        let (vec_original_res, vec_expanded_res, keyword_scroll_res) = tokio::try_join!(
            // 第一路: 原始查询的向量搜索 (高精度)
            async {
                let vector = self.get_embedding(&original_query).await?;
                self.qdrant_client.search_points(
                    SearchPointsBuilder::new(COLLECTION_NAME, vector, 5)
                        .with_payload(true)
                        // 修正：使用正确的 API 方法名 score_threshold
                        .score_threshold(VECTOR_SCORE_THRESHOLD)
                ).await.map_err(|e| anyhow::anyhow!("Original vector search failed: {}", e))
            },
            // 第二路: 扩展查询的向量搜索 (高召回)
            async {
                let vector = self.get_embedding(&expanded_query_str).await?;
                self.qdrant_client.search_points(
                    SearchPointsBuilder::new(COLLECTION_NAME, vector, 5)
                        .with_payload(true)
                        // 修正：使用正确的 API 方法名 score_threshold
                        .score_threshold(VECTOR_SCORE_THRESHOLD)
                ).await.map_err(|e| anyhow::anyhow!("Expanded vector search failed: {}", e))
            },
            // 第三路: 关键词搜索 (补充精确匹配)
            async {
                let keywords = self.extract_keywords(query_text);
                if keywords.is_empty() {
                    // 如果没有关键词，返回一个空的 Ok(None) 来匹配类型
                    return Ok(None);
                }
                use qdrant_client::qdrant::{r#match::MatchValue, Condition, Filter};
                let filter = Filter::must(keywords.iter().map(|k| Condition::matches("content", MatchValue::Text(k.clone()))));
                // 修正：返回原始的 ScrollResponse，而不是预处理它
                let scroll_response = self.qdrant_client.scroll(
                    qdrant_client::qdrant::ScrollPointsBuilder::new(COLLECTION_NAME).filter(filter).limit(5).with_payload(true)
                ).await.map_err(|e| anyhow::anyhow!("Keyword search failed: {}", e))?;
                // 使用 Some 包装，以便与向量搜索的 Option<ScrollResponse> 类型匹配（虽然向量搜索不会返回None，但这样类型更安全）
                Ok(Some(scroll_response))
            }
        )?;

        // 修正：在 try_join! 之后，统一处理所有召回结果
        let mut all_results: Vec<Vec<ScoredPoint>> = Vec::new();
        all_results.push(vec_original_res.result);
        all_results.push(vec_expanded_res.result);
        if let Some(scroll_res) = keyword_scroll_res {
            let keyword_points = scroll_res.result.into_iter().map(|p: qdrant_client::qdrant::RetrievedPoint| {
                ScoredPoint {
                    id: p.id,
                    payload: p.payload,
                    score: 1.0, // 关键词匹配结果的 RRF 基础分设为 1.0
                    version: 0,
                    vectors: p.vectors,
                    order_value: p.order_value,
                    shard_key: p.shard_key,
                }
            }).collect();
            all_results.push(keyword_points);
        }

        // 步骤 3: RRF融合三路结果
        let fused_points = self.reciprocal_rank_fusion_multi(all_results, 60);
        let filtered_points = self.apply_dynamic_threshold(fused_points);

        Ok(filtered_points)
    }

    // 4. 【新增】update 方法
    pub async fn update(&self, id: i64, new_content: &str) -> Result<(), anyhow::Error> {
        println!("[MemosAgent] Updating memo ID: {}", id);
        use rusqlite::params; // 导入 params! 宏

        // 更新 SQLite
        let conn = self.sql_pool.get()?;
        let now = Utc::now().to_rfc3339();
        // 修正 #1: 使用 rusqlite::params! 宏，更安全、更清晰
        conn.execute(
            "UPDATE facts SET content = ?1, updated_at = ?2 WHERE id = ?3",
            params![new_content, now, id],
        )?;
        println!("[MemosAgent-DB] Updated SQLite for ID: {}", id);

        // 更新 Qdrant (通过重新 Upsert)
        let vector_data = self.get_embedding(new_content).await?;
        let qdrant_vector = Vector {
            vector: Some(QdrantVectorEnums::Dense(DenseVector { data: vector_data })),
            ..Default::default()
        };
        let payload: Payload = json!({"content": new_content, "updated_at": now}).try_into()?;
        let point = PointStruct::new(id as u64, qdrant_vector, payload);
        self.qdrant_client.upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME.to_string(), vec![point])).await?;
        println!("[MemosAgent-DB] Re-upserted point to Qdrant for ID: {}", id);

        Ok(())
    }

    // 5. 【最终修正】delete 方法
    pub async fn delete(&self, id: i64) -> Result<(), anyhow::Error> {
        println!("[MemosAgent] Deleting memo ID: {}", id);
        use rusqlite::params;

        // 从 SQLite 删除
        let conn = self.sql_pool.get()?;
        conn.execute("DELETE FROM facts WHERE id = ?1", params![id])?;
        println!("[MemosAgent-DB] Deleted from SQLite for ID: {}", id);

        // 从 Qdrant 删除
        let point_id_to_delete = point_id::PointIdOptions::Num(id as u64);
        let points_list = PointsIdsList { ids: vec![point_id_to_delete.into()] };
        
        self.qdrant_client.delete_points(
            DeletePointsBuilder::new(COLLECTION_NAME.to_string())
                // 修正：直接传递 points_list，移除最后的 .into()
                .points(points_list) 
        ).await?;
        println!("[MemosAgent-DB] Deleted point from Qdrant for ID: {}", id);

        Ok(())
    }

    // --- 保持不变的辅助函数 ---
    async fn get_embedding(&self, text: &str) -> Result<Vec<f32>, anyhow::Error> {
        println!("[MemosAgent-Embed] Requesting vector for text: '{}'", text);
        let client = reqwest::Client::new();
        let embedding_url = "http://localhost:8181/embedding";
        let response = client.post(embedding_url).json(&EmbeddingRequest { content: text }).send().await?;
        if !response.status().is_success() {
            let error_body = response.text().await?;
            return Err(anyhow::anyhow!("Embedding service returned an error: {}", error_body));
        }
        let mut embedding_response = response.json::<EmbeddingResponse>().await?;
        if let Some(mut first_item) = embedding_response.0.pop() {
            if let Some(embedding_vector) = first_item.embedding.pop() {
                println!("[MemosAgent-Embed] Received {}d vector.", embedding_vector.len());
                Ok(embedding_vector)
            } else {
                Err(anyhow::anyhow!("Embedding service returned an item with an empty embedding list."))
            }
        } else {
            Err(anyhow::anyhow!("Embedding service returned an empty array."))
        }
    }

    fn extract_keywords(&self, query_text: &str) -> Vec<String> {
        println!("[MemosAgent-Keyword] Extracting keywords with Jieba...");
        use jieba_rs::Jieba;
        use stop_words::{get, LANGUAGE};

        let jieba = Jieba::new();
        let stop_words = get(LANGUAGE::Chinese);

        let keywords: Vec<String> = jieba.cut_for_search(query_text, true)
            .into_iter()
            .map(|s| s.to_lowercase())
            .filter(|word| !stop_words.contains(word))
            .collect();
        
        println!("[MemosAgent-Keyword] Extracted keywords: {:?}", keywords);
        keywords
    }
    

    fn apply_dynamic_threshold(&self, points: Vec<ScoredPoint>) -> Vec<ScoredPoint> {
        if points.is_empty() {
            return points;
        }
        let scores: Vec<f32> = points.iter().map(|p| p.score).collect();
        if scores.len() == 1 {
            return if scores[0] > 0.01 { points } else { vec![] }; // RRF分数很小，阈值也要小
        }
        let mut best_drop_index = 0;
        let mut max_drop = 0.0;
        for i in 1..scores.len() {
            let drop = scores[i-1] - scores[i];
            if drop > max_drop {
                max_drop = drop;
                best_drop_index = i;
            }
        }
        // RRF分数断崖会更明显，可以用一个更敏感的比例
        if max_drop > scores[best_drop_index - 1] * 0.3 && best_drop_index > 0 {
            println!("[MemosAgent-Threshold] Found score drop at index {}, truncating.", best_drop_index);
            return points.into_iter().take(best_drop_index).collect();
        }
        println!("[MemosAgent-Threshold] No significant drop found, returning all points.");
        points
    }
}


#[async_trait]
impl Agent for MemosAgent {
    fn name(&self) -> &'static str { "memos_agent" }
    
    // interests 现在不再重要，但为了满足 trait 定义，我们保留它
    fn interests(&self) -> &[&'static str] { &["SaveIntent", "RecallIntent"] }

    fn as_any(&self) -> &dyn Any {
    self}

    async fn handle_command(&self, command: &Command) -> Result<Response, anyhow::Error> {
        // 这个函数在V2.2架构中不应被直接调用。
        // 我们提供一个默认的回退行为以满足trait的编译要求。
        match command {
            Command::ProcessText(_) => {
                Ok(Response::Text("This agent should be called via its specific methods (save/recall), not handle_command.".to_string()))
            }
        }
    }
}

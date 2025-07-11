// backend/agent_memos/src/lib.rs
// 【头部最终修正版】

mod db;
mod query_expander;

use async_trait::async_trait;
use chrono::Utc;
use qdrant_client::{
    qdrant::{
        point_id, r#match::MatchValue, vector::Vector as QdrantVectorEnums, Condition,
        CreateCollectionBuilder, DeletePointsBuilder, DenseVector, Distance, Filter,
        PointStruct, PointsIdsList, RetrievedPoint, ScoredPoint, ScrollPointsBuilder,
        SearchPointsBuilder, UpsertPointsBuilder, Vector, VectorParamsBuilder,
    },
    Payload, Qdrant,
};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use serde_json::json;
use std::any::Any;
use std::collections::HashMap;

use crate::query_expander::QueryExpander;
use memos_core::{Agent, Command, FactMetadata, Response};


type DbPool = Pool<SqliteConnectionManager>;
const COLLECTION_NAME: &str = "memos";
const EMBEDDING_DIM: u64 = 512; 

pub struct MemosAgent {
    sql_pool: DbPool,
    qdrant_client: Qdrant,
    query_expander: QueryExpander,
    embedding_url: String,
}

#[derive(serde::Deserialize, Debug)]
struct LlamaCppEmbeddingItem {
    embedding: Vec<Vec<f32>>, 
}


impl MemosAgent {
    // ... new, reciprocal_rank_fusion_multi, save, recall, update, delete, get_by_id, extract_keywords, apply_dynamic_threshold 函数保持不变 ...
    pub async fn new(qdrant_url: &str, embedding_url: &str) -> Result<Self, anyhow::Error> {
        let home_dir = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
        let db_dir = home_dir.join(".memos_agent");
        std::fs::create_dir_all(&db_dir)?;
        let sql_db_path = db_dir.join("memos.db");
        let manager = SqliteConnectionManager::file(sql_db_path);
        let sql_pool = Pool::new(manager)?;
        
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
            query_expander: QueryExpander::new(),
            embedding_url: embedding_url.to_string(), 
        })
    }

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


    pub async fn save(&self, content: &str, metadata: Option<FactMetadata>) -> Result<i64, anyhow::Error> {
        println!("[MemosAgent] Saving memo with metadata: '{}'", content);
        let conn = self.sql_pool.get()?;
        let now = Utc::now().to_rfc3339();

        // 1. 【升级】处理元数据，将其序列化为JSON字符串以便存入SQLite
        let metadata_json_string = metadata.as_ref()
            .map(|m| serde_json::to_string(m).unwrap_or_else(|_| "{}".to_string()))
            .unwrap_or_else(|| "{}".to_string());

        // 2. 【升级】执行带元数据的SQL插入
        conn.execute(
            "INSERT INTO facts (content, metadata, created_at) VALUES (?1, ?2, ?3)",
            &[content, &metadata_json_string, &now],
        )?;
        let memo_id = conn.last_insert_rowid();
        println!("[MemosAgent-DB] Saved to SQLite with ID: {}", memo_id);

        // 3. 【升级】构建同时包含 content 和 metadata 的 Qdrant Payload
        let vector_data = self.get_embedding(content).await?;
        let qdrant_vector = Vector {
            vector: Some(QdrantVectorEnums::Dense(DenseVector { data: vector_data })),
            ..Default::default()
        };

        // 在Payload中加入topics，以便后续进行过滤搜索
        let payload_map = if let Some(meta) = metadata {
            json!({
                "content": content,
                "created_at": now,
                "metadata": {
                    "topics": meta.topics
                }
            })
        } else {
            json!({
                "content": content,
                "created_at": now
            })
        };
        
        let payload: Payload = payload_map.try_into()?;
        let points = vec![PointStruct::new(memo_id as u64, qdrant_vector, payload)];
        self.qdrant_client.upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME.to_string(), points)).await?;
        
        println!("[MemosAgent-DB] Upserted point to Qdrant with ID: {} and metadata.", memo_id);
        Ok(memo_id)
    }


    pub async fn recall(&self, query_text: &str) -> Result<Vec<ScoredPoint>, anyhow::Error> {
        println!("[MemosAgent] Recalling V2.2 for: '{}'", query_text);

        let expansions = self.query_expander.expand(query_text);
        let original_query = expansions.get(0).unwrap_or(&query_text.to_string()).clone();
        let expanded_query_str = expansions.join(" ");

        const VECTOR_SCORE_THRESHOLD: f32 = 0.5;

        // 【核心修正】: 明确地将所有块内的错误都转换为 anyhow::Error
        let (vec_original_res, vec_expanded_res, keyword_scroll_res) = tokio::try_join!(
            async {
                let vector = self.get_embedding(&original_query).await?; // 返回 anyhow::Error
                self.qdrant_client
                    .search_points(
                        SearchPointsBuilder::new(COLLECTION_NAME, vector, 5)
                            .with_payload(true)
                            .score_threshold(VECTOR_SCORE_THRESHOLD),
                    )
                    .await
                    .map_err(anyhow::Error::from) // 将 QdrantError 转换为 anyhow::Error
            },
            async {
                let vector = self.get_embedding(&expanded_query_str).await?; // 返回 anyhow::Error
                self.qdrant_client
                    .search_points(
                        SearchPointsBuilder::new(COLLECTION_NAME, vector, 5)
                            .with_payload(true)
                            .score_threshold(VECTOR_SCORE_THRESHOLD),
                    )
                    .await
                    .map_err(anyhow::Error::from) // 将 QdrantError 转换为 anyhow::Error
            },
            async {
                let keywords = self.extract_keywords(query_text);
                if keywords.is_empty() {
                    return Ok(None);
                }
                let filter = Filter::must(
                    keywords
                        .iter()
                        .map(|k| Condition::matches("content", MatchValue::Text(k.clone()))),
                );
                let scroll_response = self.qdrant_client
                    .scroll(
                        ScrollPointsBuilder::new(COLLECTION_NAME)
                            .filter(filter)
                            .limit(5)
                            .with_payload(true),
                    )
                    .await
                    .map_err(anyhow::Error::from)?; // 将 QdrantError 转换为 anyhow::Error
                Ok(Some(scroll_response))
            }
        )?;

        let mut all_results: Vec<Vec<ScoredPoint>> = Vec::new();
        
        // 【核心修正】: try_join! 成功后，返回的值不再是 Result，直接使用 .result
        all_results.push(vec_original_res.result);
        all_results.push(vec_expanded_res.result);
        
        // 【核心修正】: 对 Option<T> 类型的 keyword_scroll_res 进行正确处理
        if let Some(scroll_res) = keyword_scroll_res {
            let keyword_points = scroll_res.result.into_iter().map(|p: RetrievedPoint| {
                ScoredPoint {
                    id: p.id,
                    payload: p.payload,
                    score: 1.0,
                    version: 0,
                    vectors: p.vectors,
                    order_value: p.order_value,
                    shard_key: p.shard_key,
                }
            }).collect();
            all_results.push(keyword_points);
        }

        let fused_points = self.reciprocal_rank_fusion_multi(all_results, 60);
        let filtered_points = self.apply_dynamic_threshold(fused_points);

        Ok(filtered_points)
    }

    
    pub async fn update(&self, id: i64, new_content: &str) -> Result<(), anyhow::Error> {
        println!("[MemosAgent] Updating memo ID: {}", id);
        use rusqlite::params; 

        let conn = self.sql_pool.get()?;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE facts SET content = ?1, updated_at = ?2 WHERE id = ?3",
            params![new_content, now, id],
        )?;
        println!("[MemosAgent-DB] Updated SQLite for ID: {}", id);

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

    pub async fn delete(&self, id: i64) -> Result<(), anyhow::Error> {
        println!("[MemosAgent] Deleting memo ID: {}", id);
        use rusqlite::params;

        let conn = self.sql_pool.get()?;
        conn.execute("DELETE FROM facts WHERE id = ?1", params![id])?;
        println!("[MemosAgent-DB] Deleted from SQLite for ID: {}", id);

        let point_id_to_delete = point_id::PointIdOptions::Num(id as u64);
        let points_list = PointsIdsList { ids: vec![point_id_to_delete.into()] };
        
        self.qdrant_client.delete_points(
            DeletePointsBuilder::new(COLLECTION_NAME.to_string())
                .points(points_list) 
        ).await?;
        println!("[MemosAgent-DB] Deleted point from Qdrant for ID: {}", id);

        Ok(())
    }

    pub async fn get_by_id(&self, id: i64) -> Result<Option<String>, anyhow::Error> {
        let conn = self.sql_pool.get()?;
        let mut stmt = conn.prepare("SELECT content FROM facts WHERE id = ?1")?;
        let mut rows = stmt.query_map([id], |row| row.get(0))?;

        if let Some(content_result) = rows.next() {
            let content: String = content_result?;
            Ok(Some(content))
        } else {
            Ok(None)
        }
    }

    async fn get_embedding(&self, text: &str) -> Result<Vec<f32>, anyhow::Error> {
        println!("[MemosAgent-Embed] Requesting vector for text: '{}'", text);
        let client = reqwest::Client::new();
        let request_url = format!("{}/embedding", self.embedding_url);
        
        // 关键修正：确保发送的JSON字段是 "content"
        let request_body = serde_json::json!({ "content": text });

        let response = client.post(&request_url)
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_else(|e| format!("Could not read error body: {}", e));
            return Err(anyhow::anyhow!("Embedding service returned an error (status {}): {}", status, error_body));
        }

        let raw_text = response.text().await?;
        // println!("[MemosAgent-Embed] Raw response text: {}", raw_text);

        // 使用新的结构体进行解析
        // 我们期望得到一个包含单个元素的数组： Vec<LlamaCppEmbeddingItem>
        let mut parsed_response: Vec<LlamaCppEmbeddingItem> = match serde_json::from_str(&raw_text) {
            Ok(resp) => resp,
            Err(e) => {
                eprintln!("\n\n[ULTIMATE_DIAGNOSIS] Failed to parse JSON with new struct: {:#?}. Raw text was: '{}'\n\n", e, raw_text);
                return Err(e.into());
            }
        };
        
        // 从解析后的复杂结构中提取出我们需要的扁平向量 Vec<f32>
        if let Some(first_item) = parsed_response.pop() {
            if let Some(embedding_vector) = first_item.embedding.into_iter().next() {
                if !embedding_vector.is_empty() {
                    println!("[MemosAgent-Embed] Successfully extracted {}d vector.", embedding_vector.len());
                    return Ok(embedding_vector);
                }
            }
        }
        
        Err(anyhow::anyhow!("Failed to extract embedding vector from parsed response. The structure might be empty."))
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
            .filter(|word| !stop_words.contains(word) && !word.trim().is_empty())
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
            return if scores[0] > 0.01 { points } else { vec![] };
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
    
    fn interests(&self) -> &[&'static str] { &["SaveIntent", "RecallIntent"] }

    fn as_any(&self) -> &dyn Any {
    self}

    async fn handle_command(&self, command: &Command) -> Result<Response, anyhow::Error> {
        match command {
            Command::ProcessText(_) => {
                Ok(Response::Text("This agent should be called via its specific methods (save/recall), not handle_command.".to_string()))
            }
        }
    }
}
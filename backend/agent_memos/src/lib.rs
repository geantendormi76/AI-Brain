// agent_memos/src/lib.rs (已完成编译修复与NER能力植入)

mod db; 
mod query_expander;
use query_expander::QueryExpander;
use memos_core::{Agent, Command, Response};
use async_trait::async_trait;
use r2d2_sqlite::SqliteConnectionManager;
use r2d2::Pool;
use chrono::Utc;
use std::collections::HashMap;
// --- 核心修复 START ---
// 1. 移除未使用的 HashSet
// 2. 导入 Qdrant Filter 所需的类型
use qdrant_client::qdrant::{
    PointStruct, Vector, VectorParamsBuilder, CreateCollectionBuilder, Distance, 
    UpsertPointsBuilder, SearchPointsBuilder, ScoredPoint, point_id, 
    vector::Vector as QdrantVectorEnums, DenseVector, PointsIdsList, DeletePointsBuilder,
    Filter, Condition, r#match::MatchValue // <-- 编译器指示需要导入的类型
};
// --- 核心修复 END ---
use qdrant_client::{Payload, Qdrant};
use serde_json::json;
use std::any::Any;
use std::path::Path;
use std::sync::Mutex;

// 3. 导入 micromodels (依赖修复后，这里将能正常工作)
use micromodels::NerClassifier;

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
    embedding_url: String,
    ner_classifier: Mutex<NerClassifier>,
}


impl MemosAgent {
    pub async fn new(qdrant_url: &str, embedding_url: &str, models_path: &Path) -> Result<Self, anyhow::Error> {
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

        if !qdrant_client.collection_exists(COLLECTION_NAME).await? {
            qdrant_client.create_collection(
                CreateCollectionBuilder::new(COLLECTION_NAME)
                    .vectors_config(VectorParamsBuilder::new(EMBEDDING_DIM, Distance::Cosine))
            ).await?;
            println!("[MemosAgent-DB] Qdrant collection '{}' created.", COLLECTION_NAME);
        }

        let ner_model_path = models_path.join("ner_core_entity.onnx");
        let ner_preprocessor_path = models_path.join("ner_core_entity_preprocessor.bin");
        let ner_classifier = NerClassifier::load(ner_model_path, ner_preprocessor_path)?;

        Ok(Self { 
            sql_pool, 
            qdrant_client,
            query_expander: QueryExpander::new(),
            embedding_url: embedding_url.to_string(),
            ner_classifier: Mutex::new(ner_classifier),  
        })
    }

    fn reciprocal_rank_fusion_multi(&self, ranked_lists: Vec<Vec<ScoredPoint>>, k: u32) -> Vec<ScoredPoint> {
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
                } else { None }
            })
            .collect();
        println!("[MemosAgent-RRF] Fusion completed. Final ranked list has {} items.", final_ranked_list.len());
        final_ranked_list
    }

    // --- 【神经连接手术 - SAVE】 ---
    pub async fn save(&self, content: &str) -> Result<i64, anyhow::Error> {
        println!("[MemosAgent] Saving memo: '{}'", content);
        let conn = self.sql_pool.get()?;
        // 4. 移除行尾的非法 `\` 字符
        let now = Utc::now().to_rfc3339();
        conn.execute("INSERT INTO facts (content, created_at) VALUES (?1, ?2)", &[content, &now])?;
        let memo_id = conn.last_insert_rowid();
        println!("[MemosAgent-DB] Saved to SQLite with ID: {}", memo_id);

        // A. 调用 NER 分类器提取实体
        let entities: Vec<String> = self.ner_classifier.lock().unwrap().predict(content)?;
        println!("[MemosAgent-NER] Extracted entities: {:?}\n", entities);

        let vector_data = self.get_embedding(content).await?;
        let qdrant_vector = Vector { vector: Some(QdrantVectorEnums::Dense(DenseVector { data: vector_data })), ..Default::default() };
        
        // B. 将提取出的实体存入 Qdrant payload
        let payload: Payload = json!({
            "content": content,
            "created_at": now,
            "entities": entities // <-- 新增的字段
        }).try_into().unwrap();

        let points = vec![PointStruct::new(memo_id as u64, qdrant_vector, payload)];
        self.qdrant_client.upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME.to_string(), points)).await?;
        println!("[MemosAgent-DB] Upserted point to Qdrant with ID: {}", memo_id);
        Ok(memo_id)
    }

    // --- 【神经连接手术 - RECALL】 ---
    pub async fn recall(&self, query_text: &str) -> Result<Vec<ScoredPoint>, anyhow::Error> {
        println!("[MemosAgent] Recalling for: '{}'", query_text);

        // A. 判断是否为精确意图
        let is_precise_intent = query_text.contains("修改") || query_text.contains("删除") || query_text.contains("那条关于");

        if is_precise_intent {
            println!("[MemosAgent] Precise intent detected. Attempting NER-based entity linking.");
            
            // B. 从用户指令中提取实体
            let entities: Vec<String> = self.ner_classifier.lock().unwrap().predict(query_text)?;
            
            println!("[MemosAgent-NER] Extracted entities from query: {:?}", entities);

            if !entities.is_empty() {
                // C. 使用实体构建精确匹配的 Filter
                let filter = Filter::must(
                    entities.iter().map(|e| Condition::matches("entities", MatchValue::Text(e.clone())))
                );

                // D. 使用 scroll API 进行精确过滤查询
                let scroll_response = self.qdrant_client.scroll(
                    qdrant_client::qdrant::ScrollPointsBuilder::new(COLLECTION_NAME.to_string()).filter(filter).limit(5).with_payload(true)
                ).await?;

                // E. 如果找到结果，直接返回，绕过模糊搜索
                if !scroll_response.result.is_empty() {
                    println!("[MemosAgent] Entity linking found {} precise results. Returning immediately.", scroll_response.result.len());
                    let precise_points: Vec<ScoredPoint> = scroll_response.result.into_iter().map(|p| ScoredPoint {
                        id: p.id, payload: p.payload, score: 1.0, // 精确匹配，给满分
                        version: 0, // RetrievedPoint 没有 version，我们提供一个默认值
                        vectors: p.vectors, order_value: p.order_value, shard_key: p.shard_key,
                    }).collect();
                    return Ok(precise_points);
                }
            }
        }
        
        // F. 如果不是精确意图或实体链接失败，则回退到标准的三路模糊搜索
        println!("[MemosAgent] Precise search failed or not applicable. Falling back to standard fuzzy recall.");
        let expansions = self.query_expander.expand(query_text);
        let original_query = expansions.get(0).unwrap_or(&query_text.to_string()).clone();
        let expanded_query_str = expansions.join(" ");

        const VECTOR_SCORE_THRESHOLD: f32 = 0.5;

        let (vec_original_res, vec_expanded_res, keyword_scroll_res) = tokio::try_join!(
            async {
                let vector = self.get_embedding(&original_query).await?;
                self.qdrant_client.search_points(
                    SearchPointsBuilder::new(COLLECTION_NAME, vector, 5)
                        .with_payload(true)
                        .score_threshold(VECTOR_SCORE_THRESHOLD)
                ).await.map_err(|e| anyhow::anyhow!("Original vector search failed: {}", e))
            },
            async {
                let vector = self.get_embedding(&expanded_query_str).await?;
                self.qdrant_client.search_points(
                    SearchPointsBuilder::new(COLLECTION_NAME, vector, 5)
                        .with_payload(true)
                        .score_threshold(VECTOR_SCORE_THRESHOLD)
                ).await.map_err(|e| anyhow::anyhow!("Expanded vector search failed: {}", e))
            },
            async {
                let keywords = self.extract_keywords(query_text);
                if keywords.is_empty() { return Ok(None); }
                let filter = Filter::must(keywords.iter().map(|k| Condition::matches("content", MatchValue::Text(k.clone()))));
                let scroll_response = self.qdrant_client.scroll(
                    qdrant_client::qdrant::ScrollPointsBuilder::new(COLLECTION_NAME).filter(filter).limit(5).with_payload(true)
                ).await.map_err(|e| anyhow::anyhow!("Keyword search failed: {}", e))?;
                Ok(Some(scroll_response))
            }
        )?;

        let mut all_results: Vec<Vec<ScoredPoint>> = Vec::new();
        all_results.push(vec_original_res.result);
        all_results.push(vec_expanded_res.result);
        if let Some(scroll_res) = keyword_scroll_res {
            let keyword_points = scroll_res.result.into_iter().map(|p: qdrant_client::qdrant::RetrievedPoint| {
                ScoredPoint {
                    id: p.id, payload: p.payload, score: 1.0, 
                    version: 0, // RetrievedPoint 没有 version，我们提供一个默认值
                    vectors: p.vectors, order_value: p.order_value, shard_key: p.shard_key,
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
        conn.execute("UPDATE facts SET content = ?1, updated_at = ?2 WHERE id = ?3", params![new_content, now, id])?;
        println!("[MemosAgent-DB] Updated SQLite for ID: {}", id);
        let vector_data = self.get_embedding(new_content).await?;
        let qdrant_vector = Vector { vector: Some(QdrantVectorEnums::Dense(DenseVector { data: vector_data })), ..Default::default() };
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
        self.qdrant_client.delete_points(DeletePointsBuilder::new(COLLECTION_NAME.to_string()).points(points_list)).await?;
        println!("[MemosAgent-DB] Deleted point from Qdrant for ID: {}", id);
        Ok(())
    }

    pub async fn get_by_id(&self, id: i64) -> Result<Option<String>, anyhow::Error> {
        let conn = self.sql_pool.get()?;
        let mut stmt = conn.prepare("SELECT content FROM facts WHERE id = ?1")?;
        let mut rows = stmt.query_map([id], |row| row.get(0))?;
        if let Some(content_result) = rows.next() {
            Ok(Some(content_result?))
        } else {
            Ok(None)
        }
    }

    async fn get_embedding(&self, text: &str) -> Result<Vec<f32>, anyhow::Error> {
        println!("[MemosAgent-Embed] Requesting vector for text: '{}'", text);
        let client = reqwest::Client::new();
        let request_url = format!("{}/embedding", self.embedding_url); 
        let response = client.post(&request_url).json(&EmbeddingRequest { content: text }).send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Embedding service returned an error: {}", response.text().await?));
        }
        let mut embedding_response = response.json::<EmbeddingResponse>().await?;
        if let Some(mut first_item) = embedding_response.0.pop() {
            if let Some(embedding_vector) = first_item.embedding.pop() {
                println!("[MemosAgent-Embed] Received {}d vector.", embedding_vector.len());
                Ok(embedding_vector)
            } else { Err(anyhow::anyhow!("Embedding service returned empty embedding list.")) }
        } else { Err(anyhow::anyhow!("Embedding service returned empty array.")) }
    }

    fn extract_keywords(&self, query_text: &str) -> Vec<String> {
        use jieba_rs::Jieba;
        use stop_words::{get, LANGUAGE};
        let jieba = Jieba::new();
        let stop_words = get(LANGUAGE::Chinese);
        let keywords: Vec<String> = jieba.cut_for_search(query_text, true)
            .into_iter().map(|s| s.to_lowercase()).filter(|word| !stop_words.contains(word)).collect();
        println!("[MemosAgent-Keyword] Extracted keywords: {:?}", keywords);
        keywords
    }
    
    fn apply_dynamic_threshold(&self, points: Vec<ScoredPoint>) -> Vec<ScoredPoint> {
        if points.is_empty() { return points; }
        let scores: Vec<f32> = points.iter().map(|p| p.score).collect();
        if scores.len() == 1 { return if scores[0] > 0.01 { points } else { vec![] }; }
        let mut best_drop_index = 0;
        let mut max_drop = 0.0;
        for i in 1..scores.len() {
            let drop = scores[i-1] - scores[i];
            if drop > max_drop { max_drop = drop; best_drop_index = i; }
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
    fn as_any(&self) -> &dyn Any { self }
    async fn handle_command(&self, command: &Command) -> Result<Response, anyhow::Error> {
        match command {
            Command::ProcessText(_) => {
                Ok(Response::Text("This agent should be called via its specific methods (save/recall), not handle_command.".to_string()))
            }
        }
    }
}
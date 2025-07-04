// agent_memos/src/lib.rs

mod query_expander; // 声明新模块
use query_expander::QueryExpander; // 导入 QueryExpander
// use 声明也相应简化
use memos_core::{Agent, Command, Response};
use async_trait::async_trait;
use r2d2_sqlite::SqliteConnectionManager;
use r2d2::Pool;
use chrono::Utc;
use std::collections::HashMap;
use qdrant_client::qdrant::{PointStruct, Vector, VectorParamsBuilder, CreateCollectionBuilder, Distance, UpsertPointsBuilder, SearchPointsBuilder, ScoredPoint, point_id, vector::Vector as QdrantVectorEnums, DenseVector};
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
const EMBEDDING_DIM: u64 = 1024; 

pub struct MemosAgent {
    sql_pool: DbPool,
    qdrant_client: Qdrant,
    query_expander: QueryExpander, // 新增字段
}

impl MemosAgent {
    pub async fn new(qdrant_url: &str) -> Result<Self, anyhow::Error> {
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
    pub async fn recall(&self, query_text: &str) -> Result<Response, anyhow::Error> {
        println!("[MemosAgent] Recalling V2.1 for: '{}'", query_text);
        
        // 步骤 1: 使用 QueryExpander 生成查询变体
        let expansions = self.query_expander.expand(query_text);
        let original_query = expansions.get(0).unwrap_or(&query_text.to_string()).clone();
        // 将所有扩展（包括原始查询）连接成一个字符串，用于扩展向量搜索
        let expanded_query_str = expansions.join(" ");

        // 步骤 2: 三路并行检索
        let (vec_original_res, vec_expanded_res, keyword_res) = tokio::try_join!(
            // 第一路: 原始查询的向量搜索 (高精度)
            async {
                let vector = self.get_embedding(&original_query).await?;
                self.qdrant_client.search_points(
                    SearchPointsBuilder::new(COLLECTION_NAME, vector, 5).with_payload(true)
                ).await.map_err(|e| anyhow::anyhow!("Original vector search failed: {}", e))
            },
            // 第二路: 扩展查询的向量搜索 (高召回)
            async {
                let vector = self.get_embedding(&expanded_query_str).await?;
                self.qdrant_client.search_points(
                    SearchPointsBuilder::new(COLLECTION_NAME, vector, 5).with_payload(true)
                ).await.map_err(|e| anyhow::anyhow!("Expanded vector search failed: {}", e))
            },
            // 第三路: 关键词搜索 (补充精确匹配)
            async {
                let keywords = self.extract_keywords(query_text);
                if keywords.is_empty() { return Ok(vec![]); } 
                use qdrant_client::qdrant::{r#match::MatchValue, Condition, Filter};
                let filter = Filter::must(keywords.iter().map(|k| Condition::matches("content", MatchValue::Text(k.clone()))));
                let scroll_response = self.qdrant_client.scroll(
                    qdrant_client::qdrant::ScrollPointsBuilder::new(COLLECTION_NAME).filter(filter).limit(5).with_payload(true)
                ).await.map_err(|e| anyhow::anyhow!("Keyword search failed: {}", e))?;
                // 将关键词搜索结果包装成 ScoredPoint 以便融合
                Ok(scroll_response.result.into_iter().map(|p: qdrant_client::qdrant::RetrievedPoint| {
                    ScoredPoint {
                        id: p.id,
                        payload: p.payload,
                        score: 1.0,
                        version: 0, // 修正：RetrievedPoint中没有version字段，我们为其提供一个默认值0
                        vectors: p.vectors,
                        order_value: p.order_value,
                        shard_key: p.shard_key,
                    }
                }).collect())
            }
        )?;

        // 步骤 3: RRF融合三路结果
        let all_results: Vec<Vec<ScoredPoint>> = vec![
            vec_original_res.result.clone(), 
            vec_expanded_res.result, 
            keyword_res.clone()
        ];
        let fused_points = self.reciprocal_rank_fusion_multi(all_results, 60);
        
        // 步骤 4: 动态阈值过滤 (保持不变)
        let filtered_points = self.apply_dynamic_threshold(fused_points);

        if filtered_points.is_empty() {
            return Ok(Response::Text("关于这个，我好像没什么印象...".to_string()));
        }

        // 步骤 5: 高置信度判断 (保持不变)
        if !filtered_points.is_empty() {
            // 获取融合后的Top-1结果的ID
            if let Some(top_id) = filtered_points[0].id.as_ref().and_then(|p| p.point_id_options.as_ref()) {
                // 检查这个ID是否存在于原始向量搜索的结果中
                let in_vector_results = vec_original_res.result.iter().any(|p| p.id.as_ref().and_then(|pi| pi.point_id_options.as_ref()) == Some(top_id));
                // 检查这个ID是否存在于关键词搜索的结果中
                let in_keyword_results = keyword_res.iter().any(|p| p.id.as_ref().and_then(|pi| pi.point_id_options.as_ref()) == Some(top_id));

                // 如果Top-1结果同时被向量和关键词找到，则认为是高置信度匹配
                if in_vector_results && in_keyword_results {
                    if let Some(content) = filtered_points[0].payload.get("content").and_then(|v| v.as_str()) {
                        println!("[MemosAgent] High confidence (vector+keyword consensus) match found. Returning directly.");
                        return Ok(Response::Text(content.to_string()));
                    }
                }
            }
        }

        // 步骤 6: 低置信度时的处理 (保持不变)
        println!("[MemosAgent] Low confidence matches. Returning summary.");
        let top_results_summary: Vec<String> = filtered_points.iter().take(3).filter_map(|p| {
            p.payload.get("content").and_then(|v| v.as_str()).map(|s| format!("- {}", s))
        }).collect();

        let response_text = format!(
            "关于“{}”，我没有找到直接的记忆，但发现了一些可能相关的内容：\n{}",
            query_text,
            top_results_summary.join("\n")
        );

        Ok(Response::Text(response_text))
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
        // 这个函数现在只是一个简单的分发器，实际逻辑在 save 和 recall 中
        // Orchestrator 将直接调用 save 和 recall，所以这个函数可能不会被直接使用
        // 但为了满足Agent trait，我们保留一个简单的实现
        match command {
            Command::ProcessText(text) => {
                // 默认行为是回忆
                self.recall(text).await
            }
        }
    }
}

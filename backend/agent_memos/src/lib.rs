// --- 粘贴下面的全部代码到 agent_memos/src/lib.rs ---

use memos_core::{Agent, Command, Response};
use async_trait::async_trait;
use r2d2_sqlite::SqliteConnectionManager;
use r2d2::Pool;
use chrono::Utc;
use std::collections::HashMap;
use qdrant_client::qdrant::point_id;
use qdrant_client::{
    Payload,
    Qdrant,
    qdrant::{
        CreateCollectionBuilder,
        Distance,
        PointStruct,
        VectorParamsBuilder,
        UpsertPointsBuilder,
        SearchPointsBuilder,
        ScoredPoint,
        Vector, 
        vector::Vector as QdrantVectorEnums, 
        DenseVector,
    }
};
use serde_json::json;

// --- 数据结构定义 (保持不变) ---
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
#[derive(serde::Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}
#[derive(serde::Serialize)]
struct ChatCompletionRequest<'a> {
    messages: Vec<ChatMessage<'a>>,
    max_tokens: u32,
}
#[derive(serde::Deserialize, Debug)]
struct ChatChoice {
    message: ChatResponseMessage,
}
#[derive(serde::Deserialize, Debug)]
struct ChatResponseMessage {
    content: String,
}
#[derive(serde::Deserialize, Debug)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

type DbPool = Pool<SqliteConnectionManager>;
const COLLECTION_NAME: &str = "memos";
const EMBEDDING_DIM: u64 = 1024; 

pub struct MemosAgent {
    sql_pool: DbPool,
    qdrant_client: Qdrant, 
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

        Ok(Self { sql_pool, qdrant_client })
    }

    async fn generate_hypothetical_document(&self, query: &str) -> Result<String, anyhow::Error> {
        // ... 此函数内容保持不变 ...
        println!("[MemosAgent-HyDE] Generating hypothetical document for query: '{}'", query);
        let client = reqwest::Client::new();
        let chat_url = "http://localhost:8282/v1/chat/completions";
        let system_prompt = r#"You are a helpful assistant. Your task is to provide a direct, factual, and likely answer to the user's question. You must first think in a <think> block, and then provide the answer.

    <example>
    <user_query>What are the advantages of Rust?</user_query>
    <assistant_response><think>The user is asking about the advantages of the Rust programming language. I should list its key features like performance, memory safety, and concurrency.</think>Rust's main advantages are its high performance, strong memory safety guarantees without a garbage collector, and excellent support for concurrent programming.</assistant_response>
    </example>

    Now, answer the user's query.
    "#;
        let request_body = ChatCompletionRequest {
            messages: vec![
                ChatMessage { role: "system", content: system_prompt },
                ChatMessage { role: "user", content: query },
            ],
            max_tokens: 256,
        };
        let response = client.post(chat_url).json(&request_body).send().await?;
        if !response.status().is_success() {
            let error_body = response.text().await?;
            return Err(anyhow::anyhow!("HyDE generation service returned an error: {}", error_body));
        }
        let mut chat_response = response.json::<ChatCompletionResponse>().await?;
        if let Some(choice) = chat_response.choices.pop() {
            let raw_content = choice.message.content;
            println!("[MemosAgent-HyDE] Raw generated content: '{}'", raw_content);
            let clean_doc = if let Some(split_content) = raw_content.split("</think>").nth(1) {
                split_content.trim().to_string()
            } else {
                raw_content.trim().to_string()
            };
            if clean_doc.is_empty() {
                println!("[MemosAgent-HyDE] Generated document is empty after cleaning, falling back to original query.");
                return Ok(query.to_string());
            }
            println!("[MemosAgent-HyDE] Cleaned document for embedding: '{}'", clean_doc);
            Ok(clean_doc)
        } else {
            println!("[MemosAgent-HyDE] Could not generate document, falling back to original query.");
            Ok(query.to_string())
        }
    }

    async fn get_embedding(&self, text: &str) -> Result<Vec<f32>, anyhow::Error> {

        println!("[MemosAgent-Embed] Requesting REAL vector for text: '{}'", text);
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
                println!("[MemosAgent-Embed] Received REAL {}d vector.", embedding_vector.len());
                Ok(embedding_vector)
            } else {
                Err(anyhow::anyhow!("Embedding service returned an item with an empty embedding list."))
            }
        } else {
            Err(anyhow::anyhow!("Embedding service returned an empty array."))
        }
    }

    async fn save_memo(&self, content: &str) -> Result<(), anyhow::Error> {

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


    fn reciprocal_rank_fusion(
        &self, 
        vec_points: Vec<ScoredPoint>, 
        kw_points: Vec<ScoredPoint>,
        k: u32
    ) -> Vec<ScoredPoint> {
        println!("[MemosAgent-RRF] Starting Reciprocal Rank Fusion...");
        
        let mut fused_scores: HashMap<u64, f32> = HashMap::new();
        let mut point_data: HashMap<u64, ScoredPoint> = HashMap::new();

        // 处理向量搜索结果
        for (rank, point) in vec_points.into_iter().enumerate() {
            if let Some(point_id::PointIdOptions::Num(memo_id)) = point.id.as_ref().and_then(|p| p.point_id_options.as_ref()) {
                let score = 1.0 / (k as f32 + (rank + 1) as f32);
                *fused_scores.entry(*memo_id).or_insert(0.0) += score;
                point_data.entry(*memo_id).or_insert(point);
            }
        }

        // 处理关键词搜索结果
        for (rank, point) in kw_points.into_iter().enumerate() {
            if let Some(point_id::PointIdOptions::Num(memo_id)) = point.id.as_ref().and_then(|p| p.point_id_options.as_ref()) {
                let score = 1.0 / (k as f32 + (rank + 1) as f32); // RRF核心算法
                *fused_scores.entry(*memo_id).or_insert(0.0) += score;
                point_data.entry(*memo_id).or_insert(point);
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

    async fn query_memos(&self, query_text: &str) -> Result<String, anyhow::Error> {
        println!("[MemosAgent-Query V3.6-RRF] Received query: '{}'", query_text);

        // --- 步骤 1: 准备工作 ---
        let keywords = self.extract_keywords(query_text);
        let hypothetical_document = self.generate_hypothetical_document(query_text).await?;
        let query_vector = self.get_embedding(&hypothetical_document).await?;

        // --- 步骤 2: 并行检索 ---
        println!("[MemosAgent-Query V3.6-RRF] Starting parallel search...");

        // 任务1: 向量搜索
        let vector_search = async {
            self.qdrant_client.search_points(
                SearchPointsBuilder::new(COLLECTION_NAME, query_vector, 10) // query_vector的所有权在这里被移动
                    .with_payload(true)
            ).await.map_err(|e| anyhow::anyhow!("Vector search failed: {}", e))
        };

        // 任务2: 关键词搜索
        let keyword_search = async {
            if keywords.is_empty() { 
                // 如果没有关键词，返回一个OK的空Vec，以匹配返回类型
                return Ok(vec![]); 
            } 
            use qdrant_client::qdrant::{r#match::MatchValue, Condition, Filter};
            let conditions: Vec<Condition> = keywords.iter()
                .map(|k| Condition::matches("content", MatchValue::Text(k.clone())))
                .collect();
            let filter = Filter::must(conditions);
            let scroll_response = self.qdrant_client.scroll(
                qdrant_client::qdrant::ScrollPointsBuilder::new(COLLECTION_NAME)
                    .filter(filter).limit(10).with_payload(true)
            ).await.map_err(|e| anyhow::anyhow!("Keyword search failed: {}", e))?;
            
            // 将 RetrievedPoint 转换为 ScoredPoint
            Ok(scroll_response.result.into_iter().map(|point| {
                ScoredPoint {
                    id: point.id, 
                    payload: point.payload, 
                    score: 1.0, // 给关键词匹配结果一个基础分1.0
                    version: 0,
                    vectors: point.vectors,
                    order_value: point.order_value,
                    shard_key: point.shard_key,
                }
            }).collect())
        };

        // --- 步骤 3: 执行并等待结果 ---
        let (vector_result, keyword_result) = tokio::try_join!(vector_search, keyword_search)?;
        
        // --- 步骤 4: RRF 融合 ---
        println!("[MemosAgent-RRF] Fusing results...");
        let fused_points = self.reciprocal_rank_fusion(
            vector_result.result,
            keyword_result,
            60 // RRF的k值
        );
        
        let final_points = fused_points;

        if final_points.is_empty() {
            return Ok("关于这个，我好像没什么印象...".to_string());
        }

        // --- 步骤 5: 格式化输出 ---
        println!("[MemosAgent-RRF] Formatting final results...");
        let results: Vec<String> = final_points.iter().enumerate().map(|(index, point)| {
            let content = point.payload.get("content")
                .and_then(|v| v.as_str()).map(|s| s.to_string())
                .unwrap_or_else(|| "[无效内容]".to_string());
            // RRF分数很小，乘以1000方便观察
            format!("{}. {} (Fusion Score: {:.4})", index + 1, content, point.score * 1000.0)
        }).collect();

        Ok(format!("关于'{}'，我找到了这些相关记忆：\n{}", query_text, results.join("\n")))
    }
}

#[async_trait]
impl Agent for MemosAgent {
    fn name(&self) -> &'static str { "memos_agent" }
    fn interests(&self) -> &[&'static str] { &[""] }

    async fn handle_command(&self, command: &Command) -> Result<Response, anyhow::Error> {
        match command {
            Command::ProcessText(text) => {
                let is_query = text.starts_with("查询") || text.starts_with("查找") || text.starts_with("搜索") || text.contains("是什么");
                if is_query {
                    let query_content = if text.contains("是什么") {
                        text.to_string()
                    } else {
                        text.replace("查询", "").replace("查找", "").replace("搜索", "").trim().to_string()
                    };
                    if query_content.is_empty() {
                        return Ok(Response::Text("请提供要查询的内容。例如：查询我最喜欢的语言".to_string()));
                    }
                    let response_text = self.query_memos(&query_content).await?;
                    Ok(Response::Text(response_text))
                } else {
                    self.save_memo(text).await?;
                    Ok(Response::Text("好的，已经记下了。".to_string()))
                }
            }
        }
    }
}
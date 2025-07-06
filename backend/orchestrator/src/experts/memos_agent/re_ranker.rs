// orchestrator/src/experts/re_ranker.rs

use reqwest::Client;
use serde::{Deserialize, Serialize};

// --- 输入与输出结构 ---
#[derive(Debug)] // ReRankRequest 不再需要 Clone
pub struct ReRankRequest<'a> {
    pub query: &'a str,
    pub documents: Vec<DocumentToRank<'a>>,
}

#[derive(Debug, Clone)]
pub struct DocumentToRank<'a> {
    pub text: &'a str,
}

#[derive(Deserialize, Debug, Clone)]
pub struct RankedDocument {
    pub text: String,
    pub score: f32,
}

// --- 策略定义 ---
#[derive(Debug)] // ReRankStrategy 不再需要 Clone
pub enum ReRankStrategy {
    ValidateTopOne { threshold: f32 },
}

// --- 内部 LLM API 交互结构 ---
#[derive(Serialize)]
struct LlmReRankRequest<'a> {
    query: &'a str,
    documents: Vec<&'a str>,
}

#[derive(Deserialize, Debug)]
struct ReRankResult {
    relevance_score: f32,
}

#[derive(Deserialize, Debug)]
struct LlmReRankResponse {
    results: Vec<ReRankResult>,
}

// --- ReRanker 结构体 ---
pub struct ReRanker {
    client: Client,
    model_url: String,
}

impl ReRanker {
    pub fn new(model_url: &str) -> Self {
        println!("[ReRanker] Initializing with model URL: {}", model_url);
        Self {
            client: Client::new(),
            model_url: model_url.to_string(),
        }
    }

    pub async fn rank(
        &self,
        request: ReRankRequest<'_>,
        strategy: ReRankStrategy
    ) -> Result<Vec<RankedDocument>, anyhow::Error> {
        println!("[ReRanker] Ranking {} documents for query: '{}'", request.documents.len(), request.query);

        if request.documents.is_empty() {
            return Ok(vec![]);
        }

        let document_texts: Vec<&str> = request.documents.iter().map(|d| d.text).collect();
        let llm_request = LlmReRankRequest {
            query: request.query,
            documents: document_texts,
        };

        let rerank_endpoint = format!("{}/rerank", self.model_url);
        let response = self.client.post(&rerank_endpoint)
            .json(&llm_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await?;
            return Err(anyhow::anyhow!("Re-ranker service returned an error. Status: {}. Body: {}", status, error_body));
        }

        let llm_response: LlmReRankResponse = response.json().await?;
        let scores: Vec<f32> = llm_response.results.into_iter().map(|r| r.relevance_score).collect();

        if scores.len() != request.documents.len() {
            return Err(anyhow::anyhow!("Re-ranker returned {} scores for {} documents. Mismatch.", scores.len(), request.documents.len()));
        }

        let mut ranked_docs: Vec<RankedDocument> = request.documents.into_iter()
            .zip(scores.into_iter())
            .map(|(doc, score)| {
                RankedDocument {
                    text: doc.text.to_string(),
                    score,
                }
            })
            .collect();

        ranked_docs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        println!("[ReRanker] Ranking completed. Top score: {}", ranked_docs.get(0).map_or(0.0, |d| d.score));

        match strategy {
            ReRankStrategy::ValidateTopOne { threshold } => {
                if let Some(top_doc) = ranked_docs.get(0) {
                    if top_doc.score >= threshold {
                        println!("[ReRanker] Top document score {} exceeds threshold {}. Returning.", top_doc.score, threshold);
                        Ok(vec![top_doc.clone()])
                    } else {
                        println!("[ReRanker] Top document score {} did not exceed threshold {}.", top_doc.score, threshold);
                        Ok(vec![])
                    }
                } else {
                    Ok(vec![])
                }
            }
        }
    }
}
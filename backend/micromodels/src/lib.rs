// backend/micromodels/src/lib.rs (V17 - 绝对服从)

use anyhow::Result;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::{inputs, execution_providers::CPUExecutionProvider, value::{Value, Tensor}};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use ndarray::{Array1, Axis};

// --- 结构体定义 (保持不变) ---
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Intent { Question, Statement, Affirm, Deny, Unknown }
#[derive(Deserialize, Debug)]
struct PreprocessorData {
    word_vocabulary: HashMap<String, usize>,
    word_idf: Vec<f32>,
    char_vocabulary: HashMap<String, usize>,
    char_idf: Vec<f32>,
}
pub struct Classifier {
    session: Session,
    preprocessor_data: PreprocessorData,
    labels: Vec<Intent>,
    jieba: jieba_rs::Jieba,
}

// --- 核心实现 ---
impl Classifier {
    pub fn load(model_path: impl AsRef<Path>, data_path: impl AsRef<Path>, labels: Vec<Intent>) -> Result<Self> {
        let _ = ort::init()
            .with_name("zhzAI-micromodels")
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .commit();

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;
        
        let file = File::open(data_path)?;
        let preprocessor_data: PreprocessorData = serde_json::from_reader(file)?;
        
        Ok(Self { session, preprocessor_data, labels, jieba: jieba_rs::Jieba::new() })
    }

    pub fn predict(&mut self, text: &str) -> Intent {
        let features = match self.preprocess(text) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[micromodels] Preprocessing failed: {}", e);
                return Intent::Unknown;
            }
        };

        let input_shape = vec![1, features.len()];
        let owned_features: Vec<f32> = features.into_raw_vec();

        let input_tensor: Value = match Tensor::from_array((input_shape, owned_features)) {
            Ok(t) => t.into(),
            Err(e) => {
                eprintln!("[micromodels] Failed to create ort::Tensor: {}", e);
                return Intent::Unknown;
            }
        };

        // 【最终修正】: 严格遵循编译器指示，删除 .unwrap()
        let inputs = inputs!{ "float_input" => input_tensor };

        let extracted_label: Option<String> = {
            let outputs = match self.session.run(inputs) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("[micromodels] ONNX session run failed: {}", e);
                    return Intent::Unknown;
                }
            };

            let output_value = &outputs[0];
            if let Ok((_shape, strings_vec)) = output_value.try_extract_strings() {
                strings_vec.iter().next().map(|s| s.to_string())
            } else {
                None
            }
        };

        if let Some(label_str) = extracted_label {
            self.map_str_to_intent(&label_str)
        } else {
            eprintln!("[micromodels] Failed to extract string label from ONNX output.");
            Intent::Unknown
        }
    }

    // ... (preprocess, calculate_tfidf, map_str_to_intent 函数保持不变) ...
    fn preprocess(&self, text: &str) -> Result<Array1<f32>> {
        let word_vector = self.calculate_tfidf(text, &self.preprocessor_data.word_vocabulary, &self.preprocessor_data.word_idf, false)?;
        let char_vector = self.calculate_tfidf(text, &self.preprocessor_data.char_vocabulary, &self.preprocessor_data.char_idf, true)?;
        let combined_features = ndarray::concatenate(Axis(0), &[word_vector.view(), char_vector.view()])?;
        Ok(combined_features)
    }

    fn calculate_tfidf(&self, text: &str, vocab: &HashMap<String, usize>, idf: &[f32], is_char_ngram: bool) -> Result<Array1<f32>> {
        let mut term_counts = HashMap::new();
        let mut total_tokens = 0;
        if is_char_ngram {
            for word in self.jieba.cut(text, false) {
                let chars: Vec<char> = word.chars().collect();
                if chars.len() >= 2 {
                    for n in 2..=5 {
                        if chars.len() >= n {
                            for i in 0..=(chars.len() - n) {
                                let ngram: String = chars[i..i + n].iter().collect();
                                *term_counts.entry(ngram).or_insert(0) += 1;
                                total_tokens += 1;
                            }
                        }
                    }
                }
            }
        } else {
            for token in self.jieba.cut(text, false) {
                *term_counts.entry(token.to_string()).or_insert(0) += 1;
                total_tokens += 1;
            }
        }
        if total_tokens == 0 { return Ok(Array1::zeros(vocab.len())); }
        let mut vector = Array1::zeros(vocab.len());
        for (term, count) in term_counts {
            if let Some(&index) = vocab.get(&term) {
                let tf = 1.0 + (count as f32).ln();
                vector[index] = tf * idf[index];
            }
        }
        let norm = vector.dot(&vector).sqrt();
        if norm > 0.0 { vector /= norm; }
        Ok(vector)
    }

    fn map_str_to_intent(&self, label_str: &str) -> Intent {
        self.labels.iter().find(|&intent| {
            let intent_str = match intent {
                Intent::Question => "Question",
                Intent::Statement => "Statement",
                Intent::Affirm => "Affirm",
                Intent::Deny => "Deny",
                _ => ""
            };
            intent_str.eq_ignore_ascii_case(label_str)
        }).cloned().unwrap_or(Intent::Unknown)
    }
}
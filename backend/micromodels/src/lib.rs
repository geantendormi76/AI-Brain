// backend/micromodels/src/lib.rs (已根据最新编译器指示修复)

use anyhow::{Result, anyhow};
use ort::session::{Session, builder::GraphOptimizationLevel};
// --- 编译器指示修复 START (E0432) ---
// 1. 既然编译器无法在任何我们尝试过的路径下找到 TensorView，我们就不再尝试导入它。
//    我们将修改代码，使其不再需要显式地声明 TensorView 类型。
use ort::{inputs, value::{Value, TensorRef}};
// --- 编译器指示修复 END (E0432) ---
use ndarray::{Array1, Axis};
use std::collections::HashMap;
use std::path::Path;
use std::fs;
use prost::Message;
use std::io::BufReader;

// 包含由build.rs在OUT_DIR中生成的代码
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/micromodels.rs"));
}
use proto::{PreprocessorData, NerPreprocessorData};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Intent { Question, Statement, Affirm, Deny, Unknown }

// Classifier 结构体及其实现保持不变，因为其中没有编译错误。
pub struct Classifier {
    session: Session,
    preprocessor_data: PreprocessorData,
    word_vocab_map: HashMap<String, usize>,
    char_vocab_map: HashMap<String, usize>,
    labels: Vec<Intent>,
    jieba: jieba_rs::Jieba,
}

impl Classifier {
    pub fn load(model_path: impl AsRef<Path>, data_path: impl AsRef<Path>, labels: Vec<Intent>) -> Result<Self> {
        let _ = ort::init().with_name("zhzAI-micromodels").commit();

        let model_bytes = fs::read(&model_path)?;
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("Failed to set intra-threads: {}", e))?
            .commit_from_memory(&model_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to commit model from memory: {}", e))?;
        
        let data_bytes = fs::read(data_path)?;
        let preprocessor_data = PreprocessorData::decode(&data_bytes[..])?;
        
        let word_features = preprocessor_data.word_features.as_ref().ok_or_else(|| anyhow::anyhow!("Missing word_features"))?;
        let char_features = preprocessor_data.char_features.as_ref().ok_or_else(|| anyhow::anyhow!("Missing char_features"))?;

        let word_vocab_map = word_features.vocabulary.iter()
            .enumerate().map(|(i, word)| (word.clone(), i)).collect();
            
        let char_vocab_map = char_features.vocabulary.iter()
            .enumerate().map(|(i, word)| (word.clone(), i)).collect();

        let mut jieba = jieba_rs::Jieba::new();
        let dict_path = model_path.as_ref().parent().unwrap().join("dict.txt");
        if dict_path.exists() {
            let file = fs::File::open(&dict_path)?;
            let mut reader = BufReader::new(file);
            jieba.load_dict(&mut reader)?;
        } else {
        }

        Ok(Self { 
            session, 
            preprocessor_data, 
            word_vocab_map,
            char_vocab_map,
            labels, 
            jieba
        })
    }

    pub fn predict(&mut self, text: &str) -> Intent {
        let features = match self.preprocess(text) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[micromodels] Preprocessing failed: {}", e);
                return Intent::Unknown;
            }
        };
        
        let input_shape = (1, features.len());
        let input_tensor_ndarray = match features.into_shape(input_shape) {
            Ok(arr) => arr,
            Err(e) => {
                eprintln!("[micromodels] Failed to reshape features: {}", e);
                return Intent::Unknown;
            }
        };

        let input_tensor_ref = match TensorRef::from_array_view((input_tensor_ndarray.shape(), input_tensor_ndarray.as_slice().unwrap())) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[micromodels] Failed to create ort::TensorRef: {}", e);
                return Intent::Unknown;
            }
        };
        
        let inputs = inputs!{ "float_input" => input_tensor_ref };
        
        let extracted_label: Option<String> = {
            let outputs = match self.session.run(inputs) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("[micromodels] ONNX session run failed: {}", e);
                    return Intent::Unknown;
                }
            };

            if let Ok(strings_vec_tuple) = outputs[0].try_extract_strings() {
                strings_vec_tuple.1.iter().next().map(|s| s.to_string())
            } else {
                None
            }
        };

        if let Some(label_str) = extracted_label {
            return self.map_str_to_intent(&label_str);
        }
        Intent::Unknown
    }

    fn preprocess(&self, text: &str) -> Result<Array1<f32>> {
        let lowercased_text = text.to_lowercase();
        let word_features = self.preprocessor_data.word_features.as_ref().unwrap();
        let char_features = self.preprocessor_data.char_features.as_ref().unwrap();
        let word_vector = self.calculate_tfidf(&lowercased_text, &self.word_vocab_map, &word_features.idf_weights, false)?;
        let char_vector = self.calculate_tfidf(&lowercased_text, &self.char_vocab_map, &char_features.idf_weights, true)?;
        let combined_features = ndarray::concatenate(Axis(0), &[word_vector.view(), char_vector.view()])?;
        Ok(combined_features)
    }

    fn calculate_tfidf(&self, text: &str, vocab_map: &HashMap<String, usize>, idf: &[f32], is_char_ngram: bool) -> Result<Array1<f32>> {
        let mut term_counts: HashMap<String, i32> = HashMap::new();
        if is_char_ngram {
            let chars: Vec<char> = text.chars().collect();
            for n in 2..=5 {
                if chars.len() >= n {
                    for i in 0..=(chars.len() - n) {
                        let ngram: String = chars[i..i + n].iter().collect();
                        *term_counts.entry(ngram).or_insert(0) += 1;
                    }
                }
            }
        } else {
            for token in self.jieba.cut(text, false) {
                *term_counts.entry(token.to_string()).or_insert(0) += 1;
            }
        }
        if term_counts.is_empty() { return Ok(Array1::zeros(vocab_map.len())); }
        let mut vector = Array1::zeros(vocab_map.len());
        for (term, count) in &term_counts { 
            if let Some(&index) = vocab_map.get(term) {
                let tf = *count as f32;
                vector[index] = tf * idf[index];
            }
        }
        let norm = vector.dot(&vector).sqrt();
        if norm > 0.0 {
            vector /= norm;
        }
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


pub struct NerClassifier {
    session: Session,
    word_to_ix: HashMap<String, i32>,
    ix_to_tag: HashMap<i32, String>,
    unknown_token_index: i32,
}

impl NerClassifier {
    pub fn load(model_path: impl AsRef<Path>, preprocessor_path: impl AsRef<Path>) -> Result<Self> {
        println!("[NerClassifier] Loading NER model from: {:?}", model_path.as_ref());

        // --- 修复：恢复被错误删除的 model_bytes 定义 ---
        let model_bytes = fs::read(model_path)?;

        let session = Session::builder()
            .map_err(|e| anyhow!("Failed to create session builder: {}", e))?
            .commit_from_memory(&model_bytes)
            .map_err(|e| anyhow!("Failed to commit model from memory: {}", e))?;

        let preprocessor_bytes = fs::read(preprocessor_path)?;
        let preprocessor_data = NerPreprocessorData::decode(&preprocessor_bytes[..])?;
        
        let word_to_ix = preprocessor_data.word_to_ix;
        let ix_to_tag = preprocessor_data.tag_to_ix.into_iter().map(|(k, v)| (v, k)).collect();
        let unknown_token_index = *word_to_ix.get("<UNK>").ok_or_else(|| anyhow!("<UNK> token not found in vocabulary"))?;

        println!("[NerClassifier] NER model and preprocessor loaded successfully.");
        Ok(Self { session, word_to_ix, ix_to_tag, unknown_token_index })
    }

    pub fn predict(&mut self, text: &str) -> Result<Vec<String>> {
            // --- 恢复单字处理逻辑 ---
            let tokens: Vec<char> = text.chars().collect();
            
            let indices: Vec<i64> = tokens
                .iter()
                .map(|c| *self.word_to_ix.get(&c.to_string()).unwrap_or(&self.unknown_token_index) as i64)
                .collect();
            // --- 恢复结束 ---

            let shape = [1, indices.len()];
            let input_tensor = Value::from_array((shape, indices))
                .map_err(|e| anyhow!("Failed to create input tensor: {}", e))?;
        
            let inputs = inputs!["input" => input_tensor];

            let outputs = self.session.run(inputs)
                .map_err(|e| anyhow!("ONNX session run failed: {}", e))?;

            let output_value: &Value = &outputs[0];
            
            let (_shape, scores_data) = output_value.try_extract_tensor::<f32>()
                .map_err(|e| anyhow!("Failed to extract f32 tensor from ONNX output: {}", e))?;

            let num_tags = self.ix_to_tag.len();
            let sequence_length = tokens.len();
            
            if scores_data.len() != sequence_length * num_tags {
                return Err(anyhow!(
                    "Shape mismatch: ONNX output length ({}) does not match sequence_length * num_tags ({} * {} = {}).",
                    scores_data.len(), sequence_length, num_tags, sequence_length * num_tags
                ));
            }
            
            let tag_indices: Vec<usize> = scores_data
                .chunks(num_tags)
                .map(|scores_for_token| {
                    scores_for_token
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(index, _)| index)
                        .unwrap_or(0)
                })
                .collect();

            let token_tag_pairs: Vec<(String, String)> = tokens
                .into_iter()
                .zip(tag_indices.into_iter())
                .map(|(token, tag_ix)| {
                    let tag = self.ix_to_tag.get(&(tag_ix as i32)).cloned().unwrap_or_else(|| "O".to_string());
                    (token.to_string(), tag)
                })
                .collect();

            // --- 新增：实体合并逻辑 ---
            // 将 [("泰", "B-LOC"), ("坦", "I-LOC"), ("计", "O")] 这样的结果合并成 ["泰坦"]
            let mut entities = Vec::new();
            let mut current_entity = String::new();

            for (token, tag) in token_tag_pairs {
                if tag.starts_with("B-") {
                    if !current_entity.is_empty() {
                        entities.push(current_entity.clone());
                    }
                    current_entity = token;
                } else if tag.starts_with("I-") {
                    if !current_entity.is_empty() {
                        current_entity.push_str(&token);
                    }
                } else { // tag is "O" or something else
                    if !current_entity.is_empty() {
                        entities.push(current_entity.clone());
                        current_entity.clear();
                    }
                }
            }
            // Don't forget the last entity
            if !current_entity.is_empty() {
                entities.push(current_entity);
            }
            // --- 实体合并逻辑结束 ---

            // 函数的返回值也需要修改
            Ok(entities)
    }
}
// backend/micromodels/src/lib.rs (V25 - 借用检查最终修正版)

use anyhow::Result;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::{inputs, value::TensorRef};
use std::collections::HashMap;
use std::path::Path;
use ndarray::{Array1, Axis};
use std::fs;
use prost::Message;
use std::io::BufReader;

// 包含由build.rs在OUT_DIR中生成的代码
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/micromodels.rs"));
}
use proto::PreprocessorData;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Intent { Question, Statement, Affirm, Deny, Unknown }

pub struct Classifier {
    session: Session,
    preprocessor_data: PreprocessorData,
    word_vocab_map: HashMap<String, usize>,
    char_vocab_map: HashMap<String, usize>,
    labels: Vec<Intent>,
    jieba: jieba_rs::Jieba,
}

impl Classifier {
    // load 函数已验证无误，保持不变
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
            println!("[micromodels] Successfully loaded custom jieba dictionary from: {:?}", dict_path);
        } else {
            println!("[micromodels] WARNING: dict.txt not found at {:?}. Using default embedded dictionary. This may cause minor inconsistencies with Python.", dict_path);
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
        
        // --- 【E0502 修正】: 将生命周期冲突的操作分离开 ---
        
        // 1. 先执行可变借用，并提取出需要的数据。
        // 我们将结果提取到一个新的变量 `extracted_label` 中。
        // `session.run` 返回的 `outputs` 在这个 `let` 语句结束时就会被销毁，
        // 从而释放对 `self` 的可变借用。
        let extracted_label: Option<String> = {
            let outputs = match self.session.run(inputs) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("[micromodels] ONNX session run failed: {}", e);
                    return Intent::Unknown;
                }
            };

            if let Ok(strings_vec_tuple) = outputs[0].try_extract_strings() {
                // 将找到的字符串克隆一份，使其生命周期与`outputs`脱钩
                strings_vec_tuple.1.iter().next().map(|s| s.to_string())
            } else {
                None
            }
        }; // `outputs` 在这里被销毁，可变借用结束

        // 2. 现在 `self` 没有被借用，我们可以安全地进行不可变借用。
        if let Some(label_str) = extracted_label {
            return self.map_str_to_intent(&label_str);
        }
        
        eprintln!("[micromodels] Failed to extract string label from ONNX output.");
        Intent::Unknown
    }


    fn preprocess(&self, text: &str) -> Result<Array1<f32>> {
        // 1. 精确复刻 `lowercase=True` 行为
        let lowercased_text = text.to_lowercase();
        println!("[micromodels-debug] Input to preprocess: '{}'", lowercased_text);

        let word_features = self.preprocessor_data.word_features.as_ref().unwrap();
        let char_features = self.preprocessor_data.char_features.as_ref().unwrap();

        // 2. 将处理后的小写文本传递给 TF-IDF 计算函数
        let word_vector = self.calculate_tfidf(&lowercased_text, &self.word_vocab_map, &word_features.idf_weights, false)?;
        let char_vector = self.calculate_tfidf(&lowercased_text, &self.char_vocab_map, &char_features.idf_weights, true)?;
        
        let combined_features = ndarray::concatenate(Axis(0), &[word_vector.view(), char_vector.view()])?;
        Ok(combined_features)
    }


    fn calculate_tfidf(&self, text: &str, vocab_map: &HashMap<String, usize>, idf: &[f32], is_char_ngram: bool) -> Result<Array1<f32>> {
        let mut term_counts: HashMap<String, i32> = HashMap::new(); // <-- 【修正1】明确指定类型
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
            if let Some(&index) = vocab_map.get(term) { // `term`现在是 &String，但 get 接受 &str，自动解引用
                let tf = *count as f32; // `count` 现在是 &i32，需要解引用
                vector[index] = tf * idf[index];
            }
        }
        let norm = vector.dot(&vector).sqrt();
        if norm > 0.0 {
            vector /= norm;
        }
        // 现在 term_counts 的所有权还在，可以安全地调用 .is_empty()
        if !term_counts.is_empty() {
            println!("[micromodels-debug] Vector for '{}' (is_char_ngram: {}):", text, is_char_ngram);
            for (i, &v) in vector.iter().enumerate() {
                if v != 0.0 {
                    // 注意：这里我们迭代的是最终的vector，而不是term_counts
                    println!("  (索引: {}, 值: {:.6})", i, v);
                }
            }
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
// in agent_memos/src/embedding.rs
use llama_cpp_2::{
    context::params::LlamaContextParams, llama_backend::LlamaBackend, llama_batch::LlamaBatch,
    model::params::LlamaModelParams, model::AddBos, model::LlamaModel,
};

pub struct EmbeddingProvider {
    model: LlamaModel,
    backend: LlamaBackend, // 我们需要持有backend以创建context
}

impl EmbeddingProvider {
    pub fn new(model_path: &str) -> Result<Self, anyhow::Error> {
        println!("[EmbeddingProvider] Initializing...");
        
        // 1. 初始化后端 (必须)
        let backend = LlamaBackend::init()?;
        
        // 2. 设置模型参数
        let model_params = LlamaModelParams::default();

        // 3. 加载模型
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;
        println!("[EmbeddingProvider] Embedding model loaded successfully from: {}", model_path);

        Ok(Self { model, backend })
    }

    pub fn get_embedding(&self, text: &str) -> Result<Vec<f32>, anyhow::Error> {
        // 1. 设置上下文参数，最关键的是 with_embeddings(true)
        let ctx_params = LlamaContextParams::default()
            .with_embeddings(true);
        
        // 2. 创建上下文
        let mut ctx = self.model.new_context(&self.backend, ctx_params)?;

        // 3. 词元化
        let tokens = self.model.str_to_token(text, AddBos::Always)?;

        // 4. 创建批处理
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        batch.add_sequence(&tokens, 0, true)?;

        // 5. 解码
        ctx.decode(&mut batch)?;

        // 6. 获取嵌入向量
        // 对于单序列批处理，我们总是获取第0个序列的嵌入
        let embeddings = ctx.embeddings_seq_ith(0)?;

        Ok(embeddings.to_vec())
    }
}